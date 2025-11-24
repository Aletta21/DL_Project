"""Training entrypoint for gene-to-isoform prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data import (
    GeneIsoformDataLoaders,
    make_loader,
    train_val_test_split,
    aggregate_by_gene,
    align_anndata,
    build_transcript_gene_index,
    densify,
    normalize_inputs,
    prepare_targets,
    summarise_gene_isoforms,
)
from models import IsoformPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNN to map genes -> isoforms.")
    parser.add_argument(
        "--gene-h5ad",
        default=DEFAULT_GENE_H5AD,
        help="Path to gene-level AnnData file.",
    )
    parser.add_argument(
        "--isoform-h5ad",
        default=DEFAULT_ISOFORM_H5AD,
        help="Path to isoform-level AnnData file.",
    )
    parser.add_argument("--train-n", type=int, default=1000, help="Number of samples for training split.")
    parser.add_argument("--val-n", type=int, default=700, help="Number of samples for validation split.")
    parser.add_argument("--test-n", type=int, default=300, help="Number of samples for test split.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("isoform_model"),
        help="Directory to store model + summaries.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--top-k", type=int, default=3, help="How many isoforms to list per gene.")
    return parser.parse_args()


def build_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 42,
) -> GeneIsoformDataLoaders:
    train_pair, val_pair, test_pair = train_val_test_split(
        X,
        Y,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        seed=seed,
    )
    train_loader = make_loader(train_pair, batch_size, shuffle=True)
    train_eval_loader = make_loader(train_pair, batch_size, shuffle=False)
    val_loader = make_loader(val_pair, batch_size, shuffle=False)
    test_loader = make_loader(test_pair, batch_size, shuffle=False)
    return GeneIsoformDataLoaders(train_loader, train_eval_loader, val_loader, test_loader)


def presence_accuracy(
    pred_counts: np.ndarray, true_counts: np.ndarray, threshold: float = 0.5
) -> float:
    """Binary accuracy on isoform presence (count > threshold)."""
    pred_presence = pred_counts > threshold
    true_presence = true_counts > threshold
    return float((pred_presence == true_presence).mean())


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, List[float]]:
    """Standard MSE training loop with best-checkpoint tracking in memory."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss = evaluate_loss(model, val_loader, device, criterion)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_loss(model, loader, device, criterion=None) -> float:
    criterion = criterion or nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def inference(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())
    return np.vstack(preds), np.vstack(targets)


def main():
    args = parse_args()

    print("Loading AnnData files...")
    bulk_genes = sc.read_h5ad(args.gene_h5ad)
    bulk_transcripts = sc.read_h5ad(args.isoform_h5ad)
    print("Data loaded.")
    bulk_genes, bulk_transcripts = align_anndata(bulk_genes, bulk_transcripts)

    gene_to_transcripts = bulk_genes.uns.get("gene_to_transcripts") or {}

    X = normalize_inputs(densify(bulk_genes))
    Y = prepare_targets(densify(bulk_transcripts))

    gene_names = bulk_genes.var_names.to_numpy()
    transcript_names = bulk_transcripts.var_names.to_numpy()
    transcript_gene_idx, _ = build_transcript_gene_index(
        gene_to_transcripts, gene_names, transcript_names
    )

    loaders = build_dataloaders(
        X,
        Y,
        batch_size=args.batch_size,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = IsoformPredictor(
        n_inputs=X.shape[1],
        n_outputs=Y.shape[1],
        hidden_sizes=(args.hidden1, args.hidden2),
        dropout=args.dropout,
    ).to(device)

    history = train_model(
        model,
        loaders.train,
        loaders.val,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()
    split_loaders = {
        "train": loaders.train_eval,
        "val": loaders.val,
        "test": loaders.test,
    }
    split_metrics = {}
    for split_name, loader in split_loaders.items():
        mse = evaluate_loss(model, loader, device, criterion)
        preds_log, targets_log = inference(model, loader, device)
        preds_counts = np.expm1(preds_log)
        targets_counts = np.expm1(targets_log)
        acc = presence_accuracy(preds_counts, targets_counts)
        split_metrics[split_name] = {
            "mse": mse,
            "accuracy": acc,
            "pred_counts": preds_counts,
            "true_counts": targets_counts,
        }
        print(f"{split_name.capitalize()} | log-space MSE: {mse:.4f} | presence accuracy: {acc*100:.2f}%")

    test_preds_counts = split_metrics["test"]["pred_counts"]
    test_true_counts = split_metrics["test"]["true_counts"]
    gene_pred_counts = aggregate_by_gene(test_preds_counts, transcript_gene_idx, len(gene_names))
    gene_true_counts = aggregate_by_gene(test_true_counts, transcript_gene_idx, len(gene_names))
    gene_mse = float(np.mean((gene_pred_counts - gene_true_counts) ** 2))
    print(f"Gene-level MSE (counts): {gene_mse:.4f}")

    isoform_df = summarise_gene_isoforms(
        test_preds_counts,
        test_true_counts,
        transcript_gene_idx,
        gene_names,
        transcript_names,
        top_k=args.top_k,
    )
    print("\nTop genes by predicted isoform abundance:")
    print(isoform_df.head(10).to_string(index=False))

    print("\nSkipping artifact saving to avoid large files.")


if __name__ == "__main__":
    main()

"\"\"\"Train a proportion-predicting MLP with softmax + KL loss.\"\"\""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data import (
    GeneIsoformDataLoaders,
    align_anndata,
    counts_to_proportions,
    densify,
    isoform_correlations,
    make_loader,
    train_val_test_split,
)
from models import IsoformPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP to predict isoform proportions (softmax + KL).")
    parser.add_argument("--gene-h5ad", default=DEFAULT_GENE_H5AD, help="Path to gene-level AnnData file.")
    parser.add_argument("--isoform-h5ad", default=DEFAULT_ISOFORM_H5AD, help="Path to isoform-level AnnData file.")
    parser.add_argument("--train-n", type=int, default=1000, help="Samples for training split.")
    parser.add_argument("--val-n", type=int, default=700, help="Samples for validation split.")
    parser.add_argument("--test-n", type=int, default=300, help="Samples for test split.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--hidden3", type=int, default=256)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
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
        X, Y, train_n=train_n, val_n=val_n, test_n=test_n, seed=seed
    )
    train_loader = make_loader(train_pair, batch_size, shuffle=True)
    train_eval_loader = make_loader(train_pair, batch_size, shuffle=False)
    val_loader = make_loader(val_pair, batch_size, shuffle=False)
    test_loader = make_loader(test_pair, batch_size, shuffle=False)
    return GeneIsoformDataLoaders(train_loader, train_eval_loader, val_loader, test_loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, List[float]]:
    """
    Train with KLDivLoss between log-softmax preds and target proportions.
    """
    criterion = nn.KLDivLoss(reduction="batchmean")
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
            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, yb)
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
    criterion = criterion or nn.KLDivLoss(reduction="batchmean")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, yb)
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def inference(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
            targets.append(yb.numpy())
    return np.vstack(preds), np.vstack(targets)


def main():
    args = parse_args()

    print("Loading AnnData files...")
    bulk_genes = sc.read_h5ad(args.gene_h5ad)
    bulk_transcripts = sc.read_h5ad(args.isoform_h5ad)
    print("Data loaded.")
    bulk_genes, bulk_transcripts = align_anndata(bulk_genes, bulk_transcripts)

    # Convert counts to per-sample proportions (summing to 1).
    X = counts_to_proportions(densify(bulk_genes))
    true_iso_counts = densify(bulk_transcripts)
    Y = counts_to_proportions(true_iso_counts)
    total_iso_counts = true_iso_counts.sum(axis=1, keepdims=True)

    loaders = build_dataloaders(
        X,
        Y,
        batch_size=args.batch_size,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
    )
    count_splits = train_val_test_split(
        true_iso_counts,
        total_iso_counts,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
        seed=42,
    )
    split_counts = {"train": count_splits[0], "val": count_splits[1], "test": count_splits[2]}

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = IsoformPredictor(
        n_inputs=X.shape[1],
        n_outputs=Y.shape[1],
        hidden_sizes=(args.hidden1, args.hidden2, args.hidden3),
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

    criterion = nn.KLDivLoss(reduction="batchmean")
    split_loaders = {"train": loaders.train_eval, "val": loaders.val, "test": loaders.test}
    for split_name, loader in split_loaders.items():
        kl = evaluate_loss(model, loader, device, criterion)
        preds, targets = inference(model, loader, device)
        prop_mse = float(np.mean((preds - targets) ** 2))

        true_counts, total_counts = split_counts[split_name]
        # Recover counts by scaling proportions with observed library sizes.
        pred_counts = preds * total_counts
        count_mse = float(np.mean((pred_counts - true_counts) ** 2))
        corrs = isoform_correlations(pred_counts, true_counts)
        mean_corr = float(np.nanmean(corrs))

        print(
            f"{split_name.capitalize()} | KLDiv: {kl:.4f} | proportion MSE: {prop_mse:.6f} | "
            f"count MSE: {count_mse:.2f} | mean isoform corr: {mean_corr:.3f}"
        )

    print("\nSkipping artifact saving to avoid large files.")


if __name__ == "__main__":
    main()

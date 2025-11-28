"""Training entrypoint for gene-to-isoform prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data import (
    GeneIsoformDataLoaders,
    make_loader,
    train_val_test_split,
    align_anndata,
    build_transcript_gene_index,
    densify,
    filter_silent_genes_isoforms,
    isoform_correlations,
    normalize_inputs,
    prepare_targets,
    summarise_isoforms,
    summarise_gene_isoforms,
)
#from models import IsoformPredictor
#from models import ResidualIsoformPredictor
from models import TransformerIsoformPredictor

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
    parser.add_argument("--train-n", type=int, default=5000, help="Number of samples for training split.")
    parser.add_argument("--val-n", type=int, default=3500, help="Number of samples for validation split.")
    parser.add_argument("--test-n", type=int, default=1500, help="Number of samples for test split.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--hidden3", type=int, default=256)
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
    patience: int = 10,
) -> Dict[str, List[float]]:
    """Masked-MSE training loop with best-checkpoint tracking in memory."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")
    no_improve = 0

    def masked_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ignore targets that are zero so silent isoforms/genes do not drive the loss.
        mask = (targets > 0).float()
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=preds.device)
        diff = (preds - targets) * mask
        return (diff * diff).sum() / denom

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = masked_mse(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss = evaluate_loss(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"No val improvement for {patience} epochs, stopping early at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_loss(model, loader, device, criterion=None) -> float:
    def masked_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = (targets > 0).float()
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=preds.device)
        diff = (preds - targets) * mask
        return (diff * diff).sum() / denom

    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = masked_mse(preds, yb)
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

    X_raw = densify(bulk_genes)
    Y_raw = densify(bulk_transcripts)
    (
        X_raw,
        Y_raw,
        gene_names,
        transcript_names,
        gene_to_transcripts,
    ) = filter_silent_genes_isoforms(
        X_raw,
        Y_raw,
        bulk_genes.var_names.to_numpy(),
        bulk_transcripts.var_names.to_numpy(),
        gene_to_transcripts,
    )
    print(
        f"Filtered zero-expression features: genes kept {len(gene_names)}/{bulk_genes.var_names.size}, "
        f"isoforms kept {len(transcript_names)}/{bulk_transcripts.var_names.size}"
    )

        # =====================================================
    # FIX 1: Reduce 45k genes → 2048 bins to avoid OOM
    # =====================================================
    from sklearn.cluster import MiniBatchKMeans

    print(f"Original number of genes: {X_raw.shape[1]}")
    print("Clustering genes into 2048 super-genes (bins) for memory-efficient Transformer...")

    kmeans = MiniBatchKMeans(
        n_clusters=2048,
        batch_size=2048,
        random_state=42,
        max_iter=500,
        n_init=10,
    )
    # Transpose: treat genes as samples, cells as features
    cluster_labels = kmeans.fit_predict(X_raw.T)

    # Aggregate raw counts per cluster
    X_binned = np.zeros((X_raw.shape[0], 2048), dtype=np.float32)
    for i in range(2048):
        mask = cluster_labels == i
        if mask.sum() > 0:
            X_binned[:, i] = X_raw[:, mask].sum(axis=1)

    # Re-normalize the binned expression (critical!)
    X = normalize_inputs(X_binned)

    # Keep your current target (log-counts for now)
    Y = prepare_targets(Y_raw)

    print(f"Reduced input dimension: {X_raw.shape[1]} → {X.shape[1]} (binned genes)")
    print(f"New X shape: {X.shape}, Y shape: {Y.shape}")


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
    #model = IsoformPredictor(
    #    n_inputs=X.shape[1],
    #    n_outputs=Y.shape[1],
    #    hidden_sizes=(args.hidden1, args.hidden2, args.hidden3),
    #    dropout=args.dropout,
    #).to(device)
    #model = ResidualIsoformPredictor(
    #    n_inputs=X.shape[1],
    #    n_outputs=Y.shape[1],
    #    hidden_sizes=(1024, 512, 512),   # keep last two the same
    #    dropout=args.dropout,
    #).to(device)
    model = TransformerIsoformPredictor(
        n_genes=X.shape[1],       
        n_isoforms=Y.shape[1],
        d_model=512,
        nhead=8,
        num_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
    ).to(device)

    history = train_model(
        model,
        loaders.train,
        loaders.val,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=20,
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
    # Isoform-wise correlation analysis (test split)
    iso_corr = isoform_correlations(test_preds_counts, test_true_counts)
    valid_corr = iso_corr[~np.isnan(iso_corr)]
    if valid_corr.size == 0:
        print("Isoform correlation: no isoforms with variance; skipping boxplot.")
    else:
        corr_mean = float(valid_corr.mean())
        print(f"Isoform correlation: mean Pearson {corr_mean:.4f} over {valid_corr.size}/{iso_corr.size} isoforms")
        args.save_dir.mkdir(parents=True, exist_ok=True)
        corr_plot_path = args.save_dir / "isoform_correlation_boxplot_transformer.png"
        plt.figure(figsize=(6, 4))
        plt.boxplot(valid_corr, vert=True, patch_artist=True)
        plt.ylabel("Pearson correlation")
        plt.title("Isoform prediction correlation (test)")
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=150)
        plt.close()
        print(f"Saved isoform correlation boxplot to {corr_plot_path}")

    # Isoform ranking by predicted abundance (with correlation if available)
    isoform_ranking = summarise_isoforms(
        test_preds_counts,
        test_true_counts,
        transcript_gene_idx,
        gene_names,
        transcript_names,
        correlations=iso_corr,
    )
    print("\nTop isoforms by predicted abundance (test):")
    print(isoform_ranking.head(15).to_string(index=False))

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
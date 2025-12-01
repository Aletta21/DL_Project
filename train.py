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
    isoform_rank_correlation,
    gene_spearman_rank_correlations,
    normalize_inputs,
    prepare_targets,
    summarise_isoforms,
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
        default=Path("isoform_model_fcnn"),
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

    X = normalize_inputs(X_raw)
    Y = prepare_targets(Y_raw)

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
        corr_plot_path = args.save_dir / "isoform_correlation_boxplot_fcnn.png"
        plt.figure(figsize=(6, 4))
        plt.boxplot(valid_corr, vert=True, patch_artist=True)
        plt.ylabel("Pearson correlation")
        plt.title("Isoform prediction correlation (test)")
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=150)
        plt.close()
        print(f"Saved isoform correlation boxplot to {corr_plot_path}")

    # Scatter plot predicted vs. true counts (sampled) to visualise overall correlation
    args.save_dir.mkdir(parents=True, exist_ok=True)
    flat_pred = test_preds_counts.ravel()
    flat_true = test_true_counts.ravel()
    max_points = 50000
    if flat_pred.size > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(flat_pred.size, size=max_points, replace=False)
        flat_pred = flat_pred[idx]
        flat_true = flat_true[idx]
    scatter_path = args.save_dir / "pred_vs_true_scatter.png"
    plt.figure(figsize=(5, 5))
    plt.scatter(flat_pred, flat_true, alpha=0.2, s=5)
    # Bisector y=x
    xy_min = min(flat_pred.min(), flat_true.min())
    xy_max = max(flat_pred.max(), flat_true.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color="gray", linestyle="--", linewidth=1, label="bisector")
    # Regression line
    if np.std(flat_pred) > 0 and np.std(flat_true) > 0:
        m, b = np.polyfit(flat_pred, flat_true, 1)
        plt.plot([xy_min, xy_max], [m * xy_min + b, m * xy_max + b], color="orange", linewidth=1.5, label="regression")
    plt.xlabel("Predicted counts")
    plt.ylabel("True counts")
    plt.title("Predicted vs. true counts (sampled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    if flat_pred.std() == 0 or flat_true.std() == 0:
        print("Scatter correlation: insufficient variance to compute Pearson.")
    else:
        scatter_corr = float(np.corrcoef(flat_pred, flat_true)[0, 1])
        print(f"Scatter correlation (pred vs true, {flat_pred.size} points): {scatter_corr:.4f}")
    print(f"Saved scatter plot to {scatter_path}")

    # Clipped scatter (0..40k) with bisector and regression
    clip_mask = (flat_pred >= 0) & (flat_pred <= 40000) & (flat_true >= 0) & (flat_true <= 40000)
    cp = flat_pred[clip_mask]
    ct = flat_true[clip_mask]
    clipped_path = args.save_dir / "pred_vs_true_scatter_clipped.png"
    if cp.size > 0:
        plt.figure(figsize=(5, 5))
        plt.scatter(cp, ct, alpha=0.2, s=5)
        xy_min_c = min(cp.min(), ct.min())
        xy_max_c = max(cp.max(), ct.max())
        plt.plot([xy_min_c, xy_max_c], [xy_min_c, xy_max_c], color="gray", linestyle="--", linewidth=1, label="bisector")
        if np.std(cp) > 0 and np.std(ct) > 0:
            m, b = np.polyfit(cp, ct, 1)
            plt.plot([xy_min_c, xy_max_c], [m * xy_min_c + b, m * xy_max_c + b], color="orange", linewidth=1.5, label="regression")
        plt.xlabel("Predicted counts (0-40k)")
        plt.ylabel("True counts (0-40k)")
        plt.title("Pred vs true (clipped to 0-40k)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(clipped_path, dpi=150)
        plt.close()
        if np.std(cp) > 0 and np.std(ct) > 0:
            corr_clip = float(np.corrcoef(cp, ct)[0, 1])
            print(f"Scatter correlation clipped (n={cp.size}): {corr_clip:.4f}")
    print(f"Saved clipped scatter to {clipped_path}")

    # Log-scale scatter on log1p values
    lp = np.log1p(flat_pred)
    lt = np.log1p(flat_true)
    log_path = args.save_dir / "pred_vs_true_scatter_log.png"
    plt.figure(figsize=(5, 5))
    plt.scatter(lp, lt, alpha=0.2, s=5)
    xy_min_l = min(lp.min(), lt.min())
    xy_max_l = max(lp.max(), lt.max())
    plt.plot([xy_min_l, xy_max_l], [xy_min_l, xy_max_l], color="gray", linestyle="--", linewidth=1, label="bisector")
    if np.std(lp) > 0 and np.std(lt) > 0:
        m, b = np.polyfit(lp, lt, 1)
        plt.plot([xy_min_l, xy_max_l], [m * xy_min_l + b, m * xy_max_l + b], color="orange", linewidth=1.5, label="regression")
        corr_log = float(np.corrcoef(lp, lt)[0, 1])
        print(f"Scatter correlation log1p (n={lp.size}): {corr_log:.4f}")
    plt.xlabel("log1p Predicted counts")
    plt.ylabel("log1p True counts")
    plt.title("Pred vs true (log1p)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path, dpi=150)
    plt.close()
    print(f"Saved log-scale scatter to {log_path}")

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
    rank_corr = isoform_rank_correlation(isoform_ranking)
    if np.isnan(rank_corr):
        print("Isoform rank correlation: insufficient data.")
    else:
        print(f"Isoform rank correlation (pred vs true ranks): {rank_corr:.4f}")
    # Per-gene Spearman-like rank correlation across isoforms
    gene_rank_corrs = gene_spearman_rank_correlations(
        test_preds_counts,
        test_true_counts,
        transcript_gene_idx,
        gene_names,
    )
    valid_gene = ~np.isnan(gene_rank_corrs)
    if valid_gene.any():
        print(
            f"Gene isoform rank correlation (per-gene Spearman): "
            f"mean {float(np.nanmean(gene_rank_corrs)):.4f} | "
            f"median {float(np.nanmedian(gene_rank_corrs)):.4f} over {valid_gene.sum()} genes"
        )
        gene_corr_path = args.save_dir / "gene_isoform_rank_correlation_boxplot.png"
        plt.figure(figsize=(6, 4))
        plt.boxplot(gene_rank_corrs[valid_gene], vert=True, patch_artist=True)
        plt.ylabel("Spearman-like rank correlation")
        plt.title("Per-gene isoform rank correlation")
        plt.tight_layout()
        plt.savefig(gene_corr_path, dpi=150)
        plt.close()
        print(f"Saved gene isoform rank correlation boxplot to {gene_corr_path}")
    else:
        print("Gene isoform rank correlation: insufficient data.")

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

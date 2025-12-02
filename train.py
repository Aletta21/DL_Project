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
    aggregate_by_gene,
    align_anndata,
    build_transcript_gene_index,
    densify,
    filter_silent_genes_isoforms,
    isoform_correlations,
    gene_spearman_per_sample,
    normalize_inputs,
    prepare_targets,
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
    parser.add_argument(
        "--whole-dataset",
        action="store_true",
        help="Use the entire dataset with default split 50/35/15 instead of fixed counts.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden1", type=int, default=1024)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--hidden3", type=int, default=256)
    parser.add_argument(
        "--skip-gene-rank",
        action="store_true",
        help="Skip per-gene Spearman rank correlations to speed up evaluation.",
    )
    parser.add_argument(
        "--gene-rank-max-cells",
        type=int,
        default=1000,
        help="Subsample this many cells for per-gene rank correlation (speeds up evaluation).",
    )
    parser.add_argument(
        "--gene-rank-top-genes",
        type=int,
        default=1000,
        help="Use only the top-N genes by total test-set expression for per-gene rank correlation (<=0 uses all).",
    )
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
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
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.2e}")

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

    # Optionally use the full dataset with default proportions 50/35/15
    if args.whole_dataset:
        total = X.shape[0]
        train_n = int(total * 0.5)
        val_n = int(total * 0.35)
        test_n = total - train_n - val_n
        if min(train_n, val_n, test_n) <= 0:
            raise SystemExit(
                f"Dataset too small for 50/35/15 split (train {train_n}, val {val_n}, test {test_n})."
            )
        print(f"Using whole dataset: train {train_n}, val {val_n}, test {test_n}")
    else:
        train_n, val_n, test_n = args.train_n, args.val_n, args.test_n

    loaders = build_dataloaders(
        X,
        Y,
        batch_size=args.batch_size,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
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
        split_metrics[split_name] = {
            "mse": mse,
            "pred_counts": preds_counts,
            "true_counts": targets_counts,
        }
        print(f"{split_name.capitalize()} | log-space MSE: {mse:.4f}")

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

    # Scatter plot (log1p) for overall correlation
    args.save_dir.mkdir(parents=True, exist_ok=True)
    flat_pred = test_preds_counts.ravel()
    flat_true = test_true_counts.ravel()
    max_points = 50000
    if flat_pred.size > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(flat_pred.size, size=max_points, replace=False)
        flat_pred = flat_pred[idx]
        flat_true = flat_true[idx]

    lp = np.log1p(flat_pred)
    lt = np.log1p(flat_true)
    log_path = args.save_dir / "pred_vs_true_scatter_log.png"
    plt.figure(figsize=(5, 5))
    plt.scatter(lp, lt, alpha=0.2, s=5)
    xy_min_l = min(lp.min(), lt.min())
    xy_max_l = max(lp.max(), lt.max())
    plt.plot([xy_min_l, xy_max_l], [xy_min_l, xy_max_l], color="gray", linestyle="--", linewidth=1, label="bisector")
    if lp.size > 1 and lt.size > 1 and np.std(lp) > 0 and np.std(lt) > 0:
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

    # Per-gene Spearman correlation of isoform ranks computed per observation (optional)
    if not args.skip_gene_rank:
        # Optional: restrict to top-N genes by total test-set expression to speed up evaluation
        top_gene_idx = None
        if args.gene_rank_top_genes is not None and args.gene_rank_top_genes > 0:
            gene_counts = aggregate_by_gene(test_true_counts, transcript_gene_idx, len(gene_names))
            gene_sums = gene_counts.sum(axis=0)
            n_top = min(args.gene_rank_top_genes, len(gene_names))
            top_gene_idx = np.argsort(gene_sums)[::-1][:n_top]
            iso_mask = np.isin(transcript_gene_idx, top_gene_idx)
            if iso_mask.any():
                test_preds_counts_top = test_preds_counts[:, iso_mask]
                test_true_counts_top = test_true_counts[:, iso_mask]
                tg = transcript_gene_idx[iso_mask]
                remap = {g: i for i, g in enumerate(top_gene_idx)}
                transcript_gene_idx_top = np.array([remap.get(g, -1) for g in tg], dtype=np.int32)
                gene_names_top = gene_names[top_gene_idx]
            else:
                transcript_gene_idx_top = transcript_gene_idx
                gene_names_top = gene_names
                test_preds_counts_top = test_preds_counts
                test_true_counts_top = test_true_counts
        else:
            transcript_gene_idx_top = transcript_gene_idx
            gene_names_top = gene_names
            test_preds_counts_top = test_preds_counts
            test_true_counts_top = test_true_counts

        gene_mean_corrs, gene_median_corrs = gene_spearman_per_sample(
            test_preds_counts_top,
            test_true_counts_top,
            transcript_gene_idx_top,
            gene_names_top,
            max_cells=args.gene_rank_max_cells,
        )
        valid_mean = ~np.isnan(gene_mean_corrs)
        valid_median = ~np.isnan(gene_median_corrs)
        if valid_mean.any():
            print(
                f"Per-gene isoform rank correlation (Spearman over observations): "
                f"mean of means {float(np.nanmean(gene_mean_corrs)):.4f} | "
                f"median of means {float(np.nanmedian(gene_mean_corrs)):.4f} over {valid_mean.sum()} genes"
            )
            mean_box_path = args.save_dir / "gene_rank_corr_mean_boxplot.png"
            plt.figure(figsize=(6, 4))
            plt.boxplot(gene_mean_corrs[valid_mean], vert=True, patch_artist=True)
            plt.ylabel("Spearman correlation (mean across observations)")
            plt.title("Per-gene isoform rank correlation (mean)")
            plt.tight_layout()
            plt.savefig(mean_box_path, dpi=150)
            plt.close()
            print(f"Saved per-gene mean rank correlation boxplot to {mean_box_path}")
        else:
            print("Per-gene mean rank correlation: insufficient data.")

        if valid_median.any():
            print(
                f"Per-gene isoform rank correlation (Spearman over observations): "
                f"mean of medians {float(np.nanmean(gene_median_corrs)):.4f} | "
                f"median of medians {float(np.nanmedian(gene_median_corrs)):.4f} over {valid_median.sum()} genes"
            )
            median_box_path = args.save_dir / "gene_rank_corr_median_boxplot.png"
            plt.figure(figsize=(6, 4))
            plt.boxplot(gene_median_corrs[valid_median], vert=True, patch_artist=True)
            plt.ylabel("Spearman correlation (median across observations)")
            plt.title("Per-gene isoform rank correlation (median)")
            plt.tight_layout()
            plt.savefig(median_box_path, dpi=150)
            plt.close()
            print(f"Saved per-gene median rank correlation boxplot to {median_box_path}")
        else:
            print("Per-gene median rank correlation: insufficient data.")
    else:
        print("Skipping per-gene rank correlations (flag --skip-gene-rank).")

    print("\nSkipping artifact saving to avoid large files.")


if __name__ == "__main__":
    main()

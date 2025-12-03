"""Simple VAE training loop on gene expression only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data import (
    align_anndata,
    densify,
    filter_silent_genes_isoforms,
    isoform_correlations,
    gene_spearman_per_sample,
    aggregate_by_gene,
    build_transcript_gene_index,
    normalize_inputs,
    prepare_targets,
)
from models.residual_model import ResidualIsoformPredictor
from models.Vae import VAE


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on genes, then MLP on latents to predict isoforms.")
    p.add_argument("--gene-h5ad", default=DEFAULT_GENE_H5AD, help="Path to gene-level AnnData file.")
    p.add_argument("--isoform-h5ad", default=DEFAULT_ISOFORM_H5AD, help="Path to isoform-level AnnData file.")
    p.add_argument("--train-n", type=int, default=5000)
    p.add_argument("--val-n", type=int, default=3500)
    p.add_argument("--test-n", type=int, default=1500)
    p.add_argument(
        "--whole-dataset",
        action="store_true",
        help="Use the entire dataset with default 50/35/15 split instead of fixed counts.",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("isoform_model_vae_res"),
        help="Directory to store model + summaries.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE).")
    p.add_argument("--beta-warmup", type=int, default=50, help="Epochs to ramp beta 0->beta.")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=5.0)
    # Predictor (latent -> isoforms)
    p.add_argument("--pred-epochs", type=int, default=200)
    p.add_argument("--pred-batch", type=int, default=128)
    p.add_argument("--pred-lr", type=float, default=5e-4)
    p.add_argument("--pred-weight-decay", type=float, default=1e-4)
    p.add_argument("--pred-dropout", type=float, default=0.25)
    p.add_argument("--hidden1", type=int, default=1536)
    p.add_argument("--hidden2", type=int, default=1024)
    p.add_argument("--hidden3", type=int, default=1024)
    p.add_argument("--hidden4", type=int, default=512)
    p.add_argument("--pred-patience", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--skip-gene-rank",
        action="store_true",
        help="Skip per-gene Spearman rank correlations to speed up evaluation.",
    )
    p.add_argument(
        "--gene-rank-max-cells",
        type=int,
        default=1000,
        help="Subsample this many cells for per-gene rank correlation (speeds up evaluation).",
    )
    p.add_argument(
        "--gene-rank-top-genes",
        type=int,
        default=1000,
        help="Use only the top-N genes by total test-set expression for per-gene rank correlation (<=0 uses all).",
    )
    return p.parse_args()


def make_split_indices(n: int, train_n: int, val_n: int, test_n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    total_needed = train_n + val_n + test_n
    if total_needed > n:
        raise SystemExit(f"Requested {total_needed} samples but only have {n}")
    idx = rng.permutation(n)[:total_needed]
    train_idx = idx[:train_n]
    val_idx = idx[train_n : train_n + val_n]
    test_idx = idx[-test_n:]
    return train_idx, val_idx, test_idx


def make_loader(X: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def make_loader_xy(X: np.ndarray, Y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def vae_loss(recon_x, x, mu, logvar, beta):
    # Clamp logvar to avoid exp overflow
    logvar = torch.clamp(logvar, -20, 20)
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def masked_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = (targets > 0).float()
    denom = mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=preds.device)
    diff = (preds - targets) * mask
    return (diff * diff).sum() / denom


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading AnnData...")
    bulk_genes = sc.read_h5ad(args.gene_h5ad)
    bulk_iso = sc.read_h5ad(args.isoform_h5ad)
    bulk_genes, bulk_iso = align_anndata(bulk_genes, bulk_iso)
    print("Data loaded.")

    gene_to_transcripts = bulk_genes.uns.get("gene_to_transcripts") or {}

    X_raw = densify(bulk_genes)
    Y_raw = densify(bulk_iso)
    X_raw, Y_raw, gene_names, transcript_names, gene_to_transcripts = filter_silent_genes_isoforms(
        X_raw,
        Y_raw,
        bulk_genes.var_names.to_numpy(),
        bulk_iso.var_names.to_numpy(),
        gene_to_transcripts,
    )
    transcript_gene_idx, _ = build_transcript_gene_index(
        gene_to_transcripts, gene_names, transcript_names
    )
    print(
        f"Filtered zero-expression features: genes kept {len(gene_names)}/{bulk_genes.var_names.size}, "
        f"isoforms kept {len(transcript_names)}/{bulk_iso.var_names.size}"
    )

    # Scale genes the same way as the baseline MLP (log1p + z-score) so latents feed a comparable predictor.
    X = normalize_inputs(X_raw)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Data stats after scaling: mean {X.mean():.3f}, std {X.std():.3f}")

    # Targets: log1p isoforms
    Y = prepare_targets(Y_raw)

    # Split (use the same indices for X and Y to keep them aligned)
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

    train_idx, val_idx, test_idx = make_split_indices(X.shape[0], train_n, val_n, test_n, seed=42)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    train_loader = make_loader(X_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, args.batch_size, shuffle=False)
    test_loader = make_loader(X_test, args.batch_size, shuffle=False)

    model = VAE(input_dim=X.shape[1], latent_dim=args.latent, intermediate_dim=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        beta_factor = min(1.0, epoch / max(1, args.beta_warmup))
        beta_cur = args.beta * beta_factor

        model.train()
        tr_tot = tr_rec = tr_kl = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta_cur)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_tot += loss.item() * xb.size(0)
            tr_rec += rec.item() * xb.size(0)
            tr_kl += kl.item() * xb.size(0)
        n_tr = len(train_loader.dataset)
        tr_tot /= n_tr
        tr_rec /= n_tr
        tr_kl /= n_tr

        model.eval()
        va_tot = va_rec = va_kl = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, mu, logvar = model(xb)
                loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta_cur)
                va_tot += loss.item() * xb.size(0)
                va_rec += rec.item() * xb.size(0)
                va_kl += kl.item() * xb.size(0)
        n_va = len(val_loader.dataset)
        va_tot /= n_va
        va_rec /= n_va
        va_kl /= n_va

        print(
            f"Epoch {epoch:03d} | train {tr_tot:.4f} (rec {tr_rec:.4f}, kl {tr_kl:.4f}) | "
            f"val {va_tot:.4f} (rec {va_rec:.4f}, kl {va_kl:.4f}) | beta {beta_cur:.3f}"
        )

        if va_tot < best_val:
            best_val = va_tot
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs without val improvement.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test loss
    model.eval()
    te_tot = te_rec = te_kl = 0.0
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta=args.beta)
            te_tot += loss.item() * xb.size(0)
            te_rec += rec.item() * xb.size(0)
            te_kl += kl.item() * xb.size(0)
    n_te = len(test_loader.dataset)
    te_tot /= n_te
    te_rec /= n_te
    te_kl /= n_te
    print(f"Test loss {te_tot:.4f} (rec {te_rec:.4f}, kl {te_kl:.4f})")

    # Encode latents
    def encode_loader(loader):
        model.eval()
        zs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                mu, _ = model.encode(xb)
                zs.append(mu.cpu().numpy())
        return np.vstack(zs)

    Z_train = encode_loader(make_loader(X_train, args.pred_batch, shuffle=False))
    Z_val = encode_loader(make_loader(X_val, args.pred_batch, shuffle=False))
    Z_test = encode_loader(make_loader(X_test, args.pred_batch, shuffle=False))

    # Train predictor on latents -> isoforms
    predictor = ResidualIsoformPredictor(
        n_inputs=Z_train.shape[1],
        n_outputs=Y.shape[1],
        hidden_sizes=(args.hidden1, args.hidden2, args.hidden3, args.hidden4),
        dropout=args.pred_dropout,
    ).to(device)
    opt_pred = torch.optim.Adam(
        predictor.parameters(), lr=args.pred_lr, weight_decay=args.pred_weight_decay
    )
    sched_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_pred, mode="min", factor=0.5, patience=10, min_lr=max(args.pred_lr * 0.1, 1e-5)
    )

    def eval_loss(model_pred, loader_xy):
        model_pred.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader_xy:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model_pred(xb)
                loss = masked_mse(pred, yb)
                total += loss.item() * xb.size(0)
        return total / len(loader_xy.dataset)

    train_xy = make_loader_xy(Z_train, Y_train, args.pred_batch, shuffle=True)
    val_xy = make_loader_xy(Z_val, Y_val, args.pred_batch, shuffle=False)
    test_xy = make_loader_xy(Z_test, Y_test, args.pred_batch, shuffle=False)

    best_pred = None
    best_val = float("inf")
    no_improve = 0
    for epoch in range(1, args.pred_epochs + 1):
        predictor.train()
        tr_loss = 0.0
        for xb, yb in train_xy:
            xb = xb.to(device)
            yb = yb.to(device)
            opt_pred.zero_grad()
            pred = predictor(xb)
            loss = masked_mse(pred, yb)
            loss.backward()
            opt_pred.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_xy.dataset)

        val_loss = eval_loss(predictor, val_xy)
        sched_pred.step(val_loss)
        cur_lr = opt_pred.param_groups[0]["lr"]
        print(f"[Predictor] Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | lr {cur_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_pred = {k: v.cpu().clone() for k, v in predictor.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.pred_patience:
            print(f"[Predictor] Early stop after {args.pred_patience} epochs without val improvement.")
            break

    if best_pred is not None:
        predictor.load_state_dict(best_pred)

    # Final metrics
    train_mse = eval_loss(predictor, train_xy)
    val_mse = eval_loss(predictor, val_xy)
    test_mse = eval_loss(predictor, test_xy)
    print(f"Predictor MSE | train {train_mse:.4f} | val {val_mse:.4f} | test {test_mse:.4f}")

    # Isoform-wise correlations on test
    predictor.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_xy:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = predictor(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    pred_counts = np.expm1(np.vstack(preds))
    true_counts = np.expm1(np.vstack(trues))
    iso_corr = isoform_correlations(pred_counts, true_counts)
    valid = iso_corr[~np.isnan(iso_corr)]
    if valid.size == 0:
        print("Isoform correlation: no isoforms with variance.")
    else:
        corr_mean = float(valid.mean())
        corr_median = float(np.median(valid))
        print(
            f"Isoform correlation: mean Pearson {corr_mean:.4f}, median {corr_median:.4f} "
            f"over {valid.size}/{iso_corr.size} isoforms"
        )
        args.save_dir.mkdir(parents=True, exist_ok=True)
        corr_plot_path = args.save_dir / "isoform_correlation_boxplot_vae.png"
        plt.figure(figsize=(6, 4))
        plt.boxplot(valid, vert=True, patch_artist=True)
        plt.ylabel("Pearson correlation")
        plt.title("Isoform prediction correlation (test)")
        plt.tight_layout()
        plt.savefig(corr_plot_path, dpi=150)
        plt.close()
        print(f"Saved isoform correlation boxplot to {corr_plot_path}")

    # Scatter plot (log1p) for overall correlation
    args.save_dir.mkdir(parents=True, exist_ok=True)
    flat_pred = pred_counts.ravel()
    flat_true = true_counts.ravel()
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
    lims = [
        min(lp.min(), lt.min()),
        max(lp.max(), lt.max()),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("log1p Pred counts")
    plt.ylabel("log1p True counts")
    plt.title("Pred vs true (log1p)")
    plt.tight_layout()
    plt.savefig(log_path, dpi=150)
    plt.close()
    print(f"Saved log-scale scatter to {log_path}")

    # Per-gene Spearman correlation of isoform ranks computed per observation (optional)
    if not args.skip_gene_rank:
        # Optional: restrict to top-N genes by total test-set expression to speed up evaluation
        top_gene_idx = None
        if args.gene_rank_top_genes is not None and args.gene_rank_top_genes > 0:
            gene_counts = aggregate_by_gene(true_counts, transcript_gene_idx, len(gene_names))
            gene_sums = gene_counts.sum(axis=0)
            n_top = min(args.gene_rank_top_genes, len(gene_names))
            top_gene_idx = np.argsort(gene_sums)[::-1][:n_top]
            iso_mask = np.isin(transcript_gene_idx, top_gene_idx)
            if iso_mask.any():
                preds_top = pred_counts[:, iso_mask]
                trues_top = true_counts[:, iso_mask]
                tg = transcript_gene_idx[iso_mask]
                remap = {g: i for i, g in enumerate(top_gene_idx)}
                transcript_gene_idx_top = np.array([remap.get(g, -1) for g in tg], dtype=np.int32)
                gene_names_top = gene_names[top_gene_idx]
            else:
                transcript_gene_idx_top = transcript_gene_idx
                gene_names_top = gene_names
                preds_top = pred_counts
                trues_top = true_counts
        else:
            transcript_gene_idx_top = transcript_gene_idx
            gene_names_top = gene_names
            preds_top = pred_counts
            trues_top = true_counts

        gene_mean_corrs, gene_median_corrs = gene_spearman_per_sample(
            preds_top,
            trues_top,
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
            mean_box_path = args.save_dir / "gene_rank_corr_mean_boxplot_vae.png"
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
            median_box_path = args.save_dir / "gene_rank_corr_median_boxplot_vae.png"
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


if __name__ == "__main__":
    main()

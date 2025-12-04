"""Train a VAE on gene inputs only (no downstream predictor)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD
from data import densify, normalize_inputs
from models.Vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on genes only (no predictor).")
    parser.add_argument("--gene-h5ad", default=DEFAULT_GENE_H5AD, help="Path to gene-level AnnData file.")
    parser.add_argument("--train-n", type=int, default=5000, help="Train split size.")
    parser.add_argument("--val-n", type=int, default=3500, help="Val split size.")
    parser.add_argument("--test-n", type=int, default=1500, help="Test split size.")
    parser.add_argument(
        "--whole-dataset",
        action="store_true",
        help="Use 50/35/15 split of full dataset instead of fixed counts.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent", type=int, default=128, help="Latent dim for VAE.")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden dim for VAE encoder/decoder.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0, help="Final KL weight (beta-VAE).")
    parser.add_argument("--beta-warmup", type=int, default=10, help="Warmup epochs to linearly ramp KL from 0 to beta.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clip norm for VAE (0 to disable).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("isoform_model_vae_latent"),
        help="Directory to save optional artifacts (currently unused).",
    )
    return parser.parse_args()


def make_loader(X: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    recon = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    beta: float,
    beta_warmup: int,
    grad_clip: float,
    patience: int,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = None
    best_val = float("inf")
    no_imp = 0
    for epoch in range(1, epochs + 1):
        model.train()
        tr_tot = tr_rec = tr_kl = 0.0
        beta_factor = min(1.0, epoch / max(1, beta_warmup))
        beta_cur = beta * beta_factor

        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta_cur)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta)
                va_tot += loss.item() * xb.size(0)
                va_rec += rec.item() * xb.size(0)
                va_kl += kl.item() * xb.size(0)
        n_va = len(val_loader.dataset)
        va_tot /= n_va
        va_rec /= n_va
        va_kl /= n_va

        print(
            f"[VAE] Epoch {epoch:03d} | train {tr_tot:.4f} (rec {tr_rec:.4f}, kl {tr_kl:.4f}) | "
            f"val {va_tot:.4f} (rec {va_rec:.4f}, kl {va_kl:.4f})"
        )

        if va_tot < best_val:
            best_val = va_tot
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"[VAE] Early stop after {patience} epochs without improvement.")
            break

    if best is not None:
        model.load_state_dict(best)


def make_split_indices(n: int, train_n: int, val_n: int, test_n: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    total_needed = train_n + val_n + test_n
    if total_needed > n:
        raise SystemExit(f"Requested {total_needed} samples but only have {n}")
    idx = rng.permutation(n)[:total_needed]
    train_idx = idx[:train_n]
    val_idx = idx[train_n : train_n + val_n]
    test_idx = idx[-test_n:]
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Loading gene AnnData...")
    bulk_genes = sc.read_h5ad(args.gene_h5ad)
    print("Data loaded.")

    X_raw = densify(bulk_genes)
    gene_mask = X_raw.sum(axis=0) > 0
    X_raw = X_raw[:, gene_mask]
    gene_names = bulk_genes.var_names.to_numpy()[gene_mask]
    print(f"Filtered zero-expression genes: kept {len(gene_names)}/{bulk_genes.var_names.size}")

    X = normalize_inputs(X_raw)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if args.whole_dataset:
        total = X.shape[0]
        train_n = int(total * 0.5)
        val_n = int(total * 0.35)
        test_n = total - train_n - val_n
        if min(train_n, val_n, test_n) <= 0:
            raise SystemExit("Dataset too small for 50/35/15 split.")
        print(f"Using whole dataset: train {train_n}, val {val_n}, test {test_n}")
    else:
        train_n, val_n, test_n = args.train_n, args.val_n, args.test_n

    train_idx, val_idx, test_idx = make_split_indices(X.shape[0], train_n, val_n, test_n)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    vae = VAE(input_dim=X.shape[1], latent_dim=args.latent, intermediate_dim=args.hidden).to(device)
    vae_train_loader = make_loader(X_train, batch=args.batch_size, shuffle=True)
    vae_val_loader = make_loader(X_val, batch=args.batch_size, shuffle=False)
    train_vae(
        vae,
        vae_train_loader,
        vae_val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        beta_warmup=args.beta_warmup,
        grad_clip=args.grad_clip,
        patience=args.patience,
    )

    # Final test reconstruction metrics
    vae_test_loader = make_loader(X_test, batch=args.batch_size, shuffle=False)
    vae.eval()
    te_tot = te_rec = te_kl = 0.0
    with torch.no_grad():
        for (xb,) in vae_test_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta=args.beta)
            te_tot += loss.item() * xb.size(0)
            te_rec += rec.item() * xb.size(0)
            te_kl += kl.item() * xb.size(0)
    n_te = len(vae_test_loader.dataset)
    te_tot /= n_te
    te_rec /= n_te
    te_kl /= n_te
    print(f"[VAE] Test loss {te_tot:.4f} (rec {te_rec:.4f}, kl {te_kl:.4f})")


if __name__ == "__main__":
    main()

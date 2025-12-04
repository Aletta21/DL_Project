"""Train simple linear regressors on PCA and VAE latents derived from X."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via ⁠ pip install scanpy ⁠.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data import (
    align_anndata,
    apply_pca,
    densify,
    filter_silent_genes_isoforms,
    normalize_inputs,
    prepare_targets,
    train_val_test_split,
)
from models.Vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare linear regression on PCA features vs VAE latents of gene inputs."
    )
    parser.add_argument("--gene-h5ad", default=DEFAULT_GENE_H5AD, help="Path to gene-level AnnData file.")
    parser.add_argument("--isoform-h5ad", default=DEFAULT_ISOFORM_H5AD, help="Path to isoform-level AnnData file.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent/PCA dimension.")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size for VAE encoder/decoder.")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0, help="Final KL weight (beta-VAE).")
    parser.add_argument("--beta-warmup", type=int, default=10, help="Warmup epochs for KL weight.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clip norm (0 disables).")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on VAE val loss.")
    parser.add_argument("--train-n", type=int, default=1000, help="Training samples for split.")
    parser.add_argument("--val-n", type=int, default=700, help="Validation samples for split.")
    parser.add_argument("--test-n", type=int, default=300, help="Test samples for split.")
    parser.add_argument(
        "--whole-dataset",
        action="store_true",
        help="Use 50/35/15 split on full dataset instead of fixed counts.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    return parser.parse_args()

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    # Match train_vae.py: recon mean, KL mean over all elements
    recon = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def make_loader(X: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def compute_splits(
    X: np.ndarray,
    Y: np.ndarray,
    whole_dataset: bool,
    train_n: int,
    val_n: int,
    test_n: int,
):
    if whole_dataset:
        total = X.shape[0]
        train_n = int(total * 0.5)
        val_n = int(total * 0.35)
        test_n = total - train_n - val_n
        if min(train_n, val_n, test_n) <= 0:
            raise SystemExit("Dataset too small for 50/35/15 split.")
        print(f"Using whole dataset: train {train_n}, val {val_n}, test {test_n}")
    else:
        print(f"Using fixed split: train {train_n}, val {val_n}, test {test_n}")
    return train_val_test_split(X, Y, train_n=train_n, val_n=val_n, test_n=test_n, seed=None)


def train_vae_model(
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
            loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta=beta_cur)
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
                loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
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


def encode_latent(model: VAE, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = make_loader(X, batch=batch_size, shuffle=False)
    latents = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            latents.append(mu.cpu().numpy())
    return np.vstack(latents)


def train_and_score_linear(X_train, Y_train, X_val, Y_val):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    val_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, val_pred)
    r2 = r2_score(Y_val, val_pred)
    return mse, r2


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("Loading AnnData files...")
    genes = sc.read_h5ad(args.gene_h5ad)
    transcripts = sc.read_h5ad(args.isoform_h5ad)
    genes, transcripts = align_anndata(genes, transcripts)
    print("Data loaded.")

    gene_to_transcripts = genes.uns.get("gene_to_transcripts") or {}

    X_raw = densify(genes)
    Y_raw = densify(transcripts)
    X_raw, Y_raw, gene_names, transcript_names, gene_to_transcripts = filter_silent_genes_isoforms(
        X_raw,
        Y_raw,
        genes.var_names.to_numpy(),
        transcripts.var_names.to_numpy(),
        gene_to_transcripts,
    )
    print(
        f"Filtered zero-expression features: genes kept {len(gene_names)}/{genes.var_names.size}, "
        f"isoforms kept {len(transcript_names)}/{transcripts.var_names.size}"
    )

    X = normalize_inputs(X_raw)
    Y = prepare_targets(Y_raw)

    (train_pair, val_pair, test_pair) = compute_splits(
        X,
        Y,
        whole_dataset=args.whole_dataset,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
    )
    X_train, Y_train = train_pair
    X_val, Y_val = val_pair
    X_test, Y_test = test_pair  # noqa: F841 (kept for possible future use)

    # PCA features with latent_dim components (fit on train only)
    X_train_pca, pca_model, pca_var = apply_pca(X_train, args.latent_dim)
    X_val_pca = pca_model.transform(X_val)

    # VAE on original preprocessed X
    vae = VAE(input_dim=X.shape[1], latent_dim=args.latent_dim, intermediate_dim=args.hidden).to(device)
    vae_train_loader = make_loader(X_train, batch=args.batch_size, shuffle=True)
    vae_val_loader = make_loader(X_val, batch=args.batch_size, shuffle=False)
    train_vae_model(
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

    # Encode train/val into latent space and z-score using train stats.
    z_train = encode_latent(vae, X_train, device=device, batch_size=args.batch_size)
    z_val = encode_latent(vae, X_val, device=device, batch_size=args.batch_size)
    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std = z_train.std(axis=0, keepdims=True) + 1e-6
    z_train_norm = (z_train - z_mean) / z_std
    z_val_norm = (z_val - z_mean) / z_std

    # Linear regression on PCA features
    pca_mse, pca_r2 = train_and_score_linear(X_train_pca, Y_train, X_val_pca, Y_val)
    # Linear regression on VAE latents
    vae_mse, vae_r2 = train_and_score_linear(z_train_norm, Y_train, z_val_norm, Y_val)

    print("\nValidation scores")
    print(f"PCA features (n={args.latent_dim}): MSE={pca_mse:.4f}, R2={pca_r2:.4f}, explained_var={pca_var:.4f}")
    print(f"VAE latents (n={args.latent_dim}): MSE={vae_mse:.4f}, R2={vae_r2:.4f}")


if __name__ == "__main__":
    main()

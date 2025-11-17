"""Training entrypoint for gene-to-isoform prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scanpy is required. Install it via `pip install scanpy`.") from exc

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD
from data_utils import (
    aggregate_by_gene,
    align_anndata,
    build_transcript_gene_index,
    densify,
    normalize_inputs,
    prepare_targets,
    summarise_gene_isoforms,
)
from dataloaders import GeneIsoformDataLoaders, make_loader, train_val_test_split
from metrics import presence_accuracy
from models import IsoformPredictor
from trainer import evaluate_loss, inference, train_model


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

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.save_dir / "isoform_predictor.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
    with (args.save_dir / "training_history.json").open("w") as f:
        json.dump(
            {
                "loss_history": history,
                "metrics": {
                    split: {"mse": m["mse"], "accuracy": m["accuracy"]}
                    for split, m in split_metrics.items()
                },
                "gene_mse": gene_mse,
            },
            f,
            indent=2,
        )
    isoform_df.to_csv(args.save_dir / "gene_isoform_summary.csv", index=False)
    print(f"\nArtifacts saved to: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()

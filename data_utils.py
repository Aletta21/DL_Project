"""Utility functions for preparing AnnData inputs and gene/isoform mappings."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def densify(adata) -> np.ndarray:
    """Return a dense float32 numpy array for AnnData.X regardless of storage."""
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def align_anndata(genes, transcripts):
    """Ensure gene/isoform AnnData objects share the same obs order."""
    if np.array_equal(genes.obs_names, transcripts.obs_names):
        return genes, transcripts
    transcripts = transcripts[genes.obs_names, :]
    return genes, transcripts


def normalize_inputs(X: np.ndarray) -> np.ndarray:
    """Apply log1p transform followed by z-scoring per gene."""
    X = np.log1p(X)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std


def prepare_targets(Y: np.ndarray) -> np.ndarray:
    """Log-transform isoform counts so the network predicts log counts."""
    return np.log1p(Y)


def build_transcript_gene_index(
    gene_to_transcripts: Dict[str, Sequence[str]],
    gene_names: Sequence[str],
    transcript_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Return an array mapping transcript column indices to gene indices."""
    gene_index = {gene: idx for idx, gene in enumerate(gene_names)}
    transcript_gene_idx = np.full(len(transcript_names), -1, dtype=np.int32)
    unmapped: List[str] = []
    transcript_to_gene = {}
    for gene, transcripts in gene_to_transcripts.items():
        for transcript in transcripts:
            transcript_to_gene[transcript] = gene
    for transcript_id, transcript in enumerate(transcript_names):
        parent_gene = transcript_to_gene.get(transcript)
        if parent_gene is None or parent_gene not in gene_index:
            unmapped.append(transcript)
            continue
        transcript_gene_idx[transcript_id] = gene_index[parent_gene]
    return transcript_gene_idx, unmapped


def aggregate_by_gene(
    matrix: np.ndarray,
    transcript_gene_idx: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Sum isoform counts per gene for each sample."""
    agg = np.zeros((matrix.shape[0], n_genes), dtype=matrix.dtype)
    for gene_idx in range(n_genes):
        mask = transcript_gene_idx == gene_idx
        if not np.any(mask):
            continue
        agg[:, gene_idx] = matrix[:, mask].sum(axis=1)
    return agg


def summarise_gene_isoforms(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    transcript_gene_idx: np.ndarray,
    gene_names: Sequence[str],
    transcript_names: Sequence[str],
    top_k: int = 3,
):
    """Return a dataframe summarising the top predicted isoforms per gene."""
    import pandas as pd

    rows = []
    avg_preds = pred_counts.mean(axis=0)
    avg_true = true_counts.mean(axis=0)
    for gene_idx, gene in enumerate(gene_names):
        mask = transcript_gene_idx == gene_idx
        if mask.sum() == 0:
            continue
        tr_names = np.array(transcript_names)[mask]
        gene_pred = avg_preds[mask]
        gene_true = avg_true[mask]
        order = np.argsort(gene_pred)[::-1]
        top_tr = tr_names[order][:top_k]
        rows.append(
            {
                "gene": gene,
                "num_isoforms": mask.sum(),
                "top_predicted_isoforms": ", ".join(top_tr.tolist()),
                "avg_pred_total": float(gene_pred.sum()),
                "avg_true_total": float(gene_true.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_pred_total", ascending=False)

"\"\"\"Data preparation utilities for gene/isoform AnnData objects.\"\"\""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def densify(adata) -> np.ndarray:
    """Return a dense float32 numpy array for AnnData.X regardless of storage."""
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def counts_to_proportions(matrix: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """
    Convert counts to per-row proportions (summing to 1 along the chosen axis).

    If a row/column sum is zero, it is replaced with 1 to avoid division by zero,
    yielding all-zero proportions for that slice.
    """
    matrix = matrix.astype(np.float32, copy=False)
    sums = matrix.sum(axis=axis, keepdims=True)
    safe_sums = np.where(sums > 0, sums, 1.0)
    return matrix / safe_sums


def align_anndata(genes, transcripts):
    """Ensure gene/isoform AnnData objects share the same obs order."""
    if np.array_equal(genes.obs_names, transcripts.obs_names):
        return genes, transcripts
    print("Reindexing isoform AnnData to match gene observations order.")
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


def filter_silent_genes_isoforms(
    X_genes: np.ndarray,
    Y_isoforms: np.ndarray,
    gene_names: Sequence[str],
    transcript_names: Sequence[str],
    gene_to_transcripts: Dict[str, Sequence[str]] | None = None,
):
    """
    Remove genes with zero expression and isoforms with zero observations or missing genes.

    Returns filtered X, Y, gene names, transcript names, and an updated gene->transcripts mapping
    containing only kept entries.
    """
    gene_to_transcripts = gene_to_transcripts or {}
    gene_names = np.asarray(gene_names)
    transcript_names = np.asarray(transcript_names)

    gene_mask = X_genes.sum(axis=0) > 0
    kept_genes = gene_names[gene_mask]
    kept_gene_set = set(kept_genes)

    transcript_to_gene: Dict[str, str] = {}
    for gene, transcripts in gene_to_transcripts.items():
        for tr in transcripts:
            transcript_to_gene[tr] = gene

    isoform_mask = np.zeros_like(transcript_names, dtype=bool)
    for idx, tr in enumerate(transcript_names):
        has_counts = Y_isoforms[:, idx].sum() > 0
        parent_gene = transcript_to_gene.get(tr)
        isoform_mask[idx] = bool(has_counts and parent_gene in kept_gene_set)

    filtered_transcripts = transcript_names[isoform_mask]
    filtered_gene_to_transcripts: Dict[str, List[str]] = {}
    filtered_transcript_set = set(filtered_transcripts)
    for gene in kept_genes:
        gene_transcripts = gene_to_transcripts.get(gene, [])
        kept_trs = [tr for tr in gene_transcripts if tr in filtered_transcript_set]
        if kept_trs:
            filtered_gene_to_transcripts[gene] = kept_trs

    return (
        X_genes[:, gene_mask],
        Y_isoforms[:, isoform_mask],
        kept_genes,
        filtered_transcripts,
        filtered_gene_to_transcripts,
    )


def build_transcript_gene_index(
    gene_to_transcripts: Dict[str, Sequence[str]],
    gene_names: Sequence[str],
    transcript_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Return an array mapping transcript column indices to gene indices."""
    gene_index = {gene: idx for idx, gene in enumerate(gene_names)}
    transcript_gene_idx = np.full(len(transcript_names), -1, dtype=np.int32)
    unmapped: List[str] = []
    transcript_to_gene: Dict[str, str] = {}
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


def isoform_correlations(pred_counts: np.ndarray, true_counts: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation per isoform across samples.

    Returns an array of shape (n_isoforms,) with NaN where variance is zero
    (pred or true) so that downstream stats can ignore those entries.
    """
    n_isoforms = pred_counts.shape[1]
    corrs = np.full(n_isoforms, np.nan, dtype=np.float32)
    for i in range(n_isoforms):
        x = pred_counts[:, i]
        y = true_counts[:, i]
        if x.std() == 0 or y.std() == 0:
            continue
        corrs[i] = float(np.corrcoef(x, y)[0, 1])
    return corrs


def summarise_isoforms(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    transcript_gene_idx: np.ndarray,
    gene_names: Sequence[str],
    transcript_names: Sequence[str],
    correlations: np.ndarray | None = None,
):
    """Return a dataframe with per-isoform averages, ranks, and optional correlation."""
    import pandas as pd

    gene_names = np.asarray(gene_names)
    avg_pred = pred_counts.mean(axis=0)
    avg_true = true_counts.mean(axis=0)
    # Compute descending ranks (1 = highest)
    rank_pred = np.empty_like(avg_pred, dtype=np.int32)
    rank_true = np.empty_like(avg_true, dtype=np.int32)
    rank_pred[np.argsort(-avg_pred)] = np.arange(1, len(avg_pred) + 1, dtype=np.int32)
    rank_true[np.argsort(-avg_true)] = np.arange(1, len(avg_true) + 1, dtype=np.int32)

    rows = []
    for iso_idx, iso_name in enumerate(transcript_names):
        gene_idx = transcript_gene_idx[iso_idx]
        gene_name = gene_names[gene_idx] if 0 <= gene_idx < len(gene_names) else "unknown"
        row = {
            "isoform": iso_name,
            "gene": gene_name,
            "avg_pred": float(avg_pred[iso_idx]),
            "avg_true": float(avg_true[iso_idx]),
            "rank_pred": int(rank_pred[iso_idx]),
            "rank_true": int(rank_true[iso_idx]),
        }
        if correlations is not None and iso_idx < len(correlations):
            row["correlation"] = float(correlations[iso_idx]) if not np.isnan(correlations[iso_idx]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_pred", ascending=False)


def isoform_rank_correlation(isoform_df) -> float:
    """
    Compute Spearman-like correlation between predicted and true isoform ranks.

    Expects columns 'rank_pred' and 'rank_true' in the dataframe returned by
    `summarise_isoforms`. Returns NaN if insufficient data.
    """
    ranks_pred = isoform_df["rank_pred"].to_numpy()
    ranks_true = isoform_df["rank_true"].to_numpy()
    mask = ~np.isnan(ranks_pred) & ~np.isnan(ranks_true)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(ranks_pred[mask], ranks_true[mask])[0, 1])


def gene_spearman_rank_correlations(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    transcript_gene_idx: np.ndarray,
    gene_names: Sequence[str],
) -> np.ndarray:
    """
    Compute Spearman-like rank correlation per gene between predicted and true isoform abundances.

    For each gene, takes the isoforms belonging to that gene, ranks their average predicted
    and average true counts (descending), and computes the correlation between rank vectors.
    Returns an array of shape (n_genes,) with NaN for genes with <2 isoforms or zero variance.
    """
    gene_names = np.asarray(gene_names)
    n_genes = len(gene_names)
    avg_pred = pred_counts.mean(axis=0)
    avg_true = true_counts.mean(axis=0)
    corrs = np.full(n_genes, np.nan, dtype=np.float32)

    for g in range(n_genes):
        mask = transcript_gene_idx == g
        if mask.sum() < 2:
            continue
        p = avg_pred[mask]
        t = avg_true[mask]
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        rank_p = np.argsort(np.argsort(-p))
        rank_t = np.argsort(np.argsort(-t))
        corrs[g] = float(np.corrcoef(rank_p, rank_t)[0, 1])
    return corrs


def gene_spearman_per_sample(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    transcript_gene_idx: np.ndarray,
    gene_names: Sequence[str],
    max_cells: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each gene, compute Spearman correlation of isoform ranks per observation.

    Returns two arrays of shape (n_genes,):
    - mean correlation per gene across observations
    - median correlation per gene across observations
    NaN for genes with <2 isoforms or no valid correlations.
    """
    gene_names = np.asarray(gene_names)
    if max_cells is not None and pred_counts.shape[0] > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(pred_counts.shape[0], size=max_cells, replace=False)
        pred_counts = pred_counts[idx]
        true_counts = true_counts[idx]

    n_genes = len(gene_names)
    mean_corrs = np.full(n_genes, np.nan, dtype=np.float32)
    median_corrs = np.full(n_genes, np.nan, dtype=np.float32)

    for g in range(n_genes):
        mask = transcript_gene_idx == g
        if mask.sum() < 2:
            continue
        corrs: List[float] = []
        p_sub = pred_counts[:, mask]
        t_sub = true_counts[:, mask]
        for p, t in zip(p_sub, t_sub):
            if np.all(p == p[0]) or np.all(t == t[0]):
                continue
            corr = spearmanr(p, t).correlation
            if np.isnan(corr):
                continue
            corrs.append(float(corr))
        if corrs:
            mean_corrs[g] = float(np.mean(corrs))
            median_corrs[g] = float(np.median(corrs))
    return mean_corrs, median_corrs

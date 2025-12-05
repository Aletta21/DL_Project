"\"\"\"Data preparation utilities for gene/isoform AnnData objects.\"\"\""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


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


def isoform_correlations(pred_counts: np.ndarray, true_counts: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation per isoform across samples.

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


def isoform_rank_correlation(isoform_df) -> float:
    """
    Compute Spearman-like correlation between predicted and true isoform ranks.

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

    """
    gene_names = np.asarray(gene_names)
    if max_cells is not None and pred_counts.shape[0] > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(pred_counts.shape[0], size=max_cells, replace=False)
        pred_counts = pred_counts[idx]
        true_counts = true_counts[idx]

    # Drop isoforms that are always zero in both pred and true across the sampled cells
    iso_mask_nonzero = (pred_counts.sum(axis=0) > 0) | (true_counts.sum(axis=0) > 0)
    if not np.all(iso_mask_nonzero):
        pred_counts = pred_counts[:, iso_mask_nonzero]
        true_counts = true_counts[:, iso_mask_nonzero]
        transcript_gene_idx = transcript_gene_idx[iso_mask_nonzero]

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


def apply_pca(X, n_components):
    """Apply PCA dimensionality reduction."""
    print(f"Applying PCA with {n_components} components...")
    
    max_components = min(X.shape[0], X.shape[1]) - 1
    if n_components > max_components:
        print(f"Reducing n_components from {n_components} to {max_components}")
        n_components = max_components
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA complete: {X_pca.shape}")
    print(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    
    return X_pca, pca, explained_var

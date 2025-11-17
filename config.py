"""Centralised configuration for dataset locations."""

from __future__ import annotations

from pathlib import Path

# Base directory containing the AnnData files.
DATA_ROOT = Path("/work3/s193518/scIsoPred/data").expanduser().resolve()

DEFAULT_GENE_H5AD = str(DATA_ROOT / "bulk_processed_genes.h5ad")
DEFAULT_ISOFORM_H5AD = str(DATA_ROOT / "bulk_processed_transcripts.h5ad")

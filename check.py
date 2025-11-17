from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_GENE_H5AD, DEFAULT_ISOFORM_H5AD

# This code is just to test whether there are isoforms mapping to multiple genes in the bulk data
gene_h5ad = DEFAULT_GENE_H5AD
isoform_h5ad = DEFAULT_ISOFORM_H5AD
print("Loading AnnData files...")
bulk_genes = ad.read_h5ad(gene_h5ad)          #Load gene-level AnnData file    
bulk_transcripts = ad.read_h5ad(isoform_h5ad)          #Load isoform-level AnnData file
print("Data loaded.")
   
g2t = bulk_genes.uns["gene_to_transcripts"]

tx_to_genes = defaultdict(list)
for gene, txs in g2t.items():
    for tx in txs:
        tx_to_genes[tx].append(gene)

multi_mapped = {tx: genes for tx, genes in tx_to_genes.items() if len(genes) > 1}
print(f"Transcripts mappati a più di un gene: {len(multi_mapped)}")
# opzionale: vedere i primi
for tx, genes in list(multi_mapped.items())[:10]:
    print(tx, "→", genes)

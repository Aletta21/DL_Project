"""Shallow Transformer for gene → isoform prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerIsoformPredictor(nn.Module):
    """
    Shallow Transformer — 2–4 layers, 8 heads, dim=512 → state-of-the-art for this task in 2025.
    Input: (batch_size, n_genes) → treat each gene as a token.
    Uses learnable position embeddings + gene embeddings.
    """
    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Learnable gene token embeddings (one per gene)
        self.gene_embedding = nn.Linear(1, d_model)  # each gene value → vector
        # Or use Embedding if you prefer: nn.Embedding(n_genes, d_model)

        # Learnable positional encoding (fixed size = n_genes)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_genes, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN = much more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction head: per-gene → per-isoform contribution
        self.output_head = nn.Linear(d_model, n_isoforms)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_genes) → normalized gene expression
        """
        batch_size = x.shape[0]

        # Expand each gene value to a vector: (B, n_genes, 1) → (B, n_genes, d_model)
        x = x.unsqueeze(-1)                     # (B, n_genes, 1)
        x = self.gene_embedding(x)              # (B, n_genes, d_model)

        # Add positional encoding
        x = x + self.pos_embedding              # broadcasting

        # Transformer expects (B, seq_len, d_model)
        x = self.transformer(x)                 # (B, n_genes, d_model)

        x = self.norm(x)
        x = self.dropout(x)

        # Predict isoform logits: sum over gene dimension? No — we want per-isoform from global context
        # → use the full sequence representation
        # Here: use mean pooling over genes → one vector per sample
        x = x.mean(dim=1)                       # (B, d_model)

        logits = self.output_head(x)            # (B, n_isoforms)
        return logits
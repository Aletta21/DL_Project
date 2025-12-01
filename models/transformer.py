"""Shallow Transformer for gene → isoform prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerIsoformPredictor(nn.Module):
    """
    Shallow Transformer — 2–4 layers, 8 heads, dim=512 → state-of-the-art for this task in 2025.
    Input: (batch_size, n_genes) → learn a projection to a smaller set of tokens
    (n_tokens) to keep attention tractable, then treat each projected token as a
    “pseudo-gene” with its own embedding and positional encoding.
    """
    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        n_tokens: int = 1024,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cls: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_tokens = n_tokens
        self.use_cls = use_cls

        # Learnable projection: map n_genes scalars → n_tokens scalars (trainable, end-to-end)
        self.token_proj = nn.Linear(n_genes, n_tokens, bias=False)

        # Learnable token embeddings (shared linear on scalar token value)
        self.gene_embedding = nn.Linear(1, d_model)  # each projected token value → vector

        # Learnable positional encoding (includes slot for CLS if enabled)
        pos_len = n_tokens + 1 if use_cls else n_tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, pos_len, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls else None

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

        # Project original genes down to a smaller set of trainable tokens
        x = self.token_proj(x)                  # (B, n_tokens)

        # Expand each token value to a vector: (B, n_tokens, 1) → (B, n_tokens, d_model)
        x = x.unsqueeze(-1)                     # (B, n_tokens, 1)
        x = self.gene_embedding(x)              # (B, n_tokens, d_model)

        # Optional CLS token for better pooling than mean
        if self.use_cls and self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)                   # (B, 1 + n_tokens, d_model)

        # Add positional encoding
        pos = self.pos_embedding[:, : x.size(1), :]
        x = x + pos                                         # broadcasting

        # Transformer expects (B, seq_len, d_model)
        x = self.transformer(x)                 # (B, n_genes, d_model)

        x = self.norm(x)
        x = self.dropout(x)

        # Pool sequence → sample representation
        if self.use_cls and self.cls_token is not None:
            pooled = x[:, 0, :]                 # (B, d_model) CLS
        else:
            pooled = x.mean(dim=1)              # fallback to mean pooling

        logits = self.output_head(pooled)       # (B, n_isoforms)

        return logits

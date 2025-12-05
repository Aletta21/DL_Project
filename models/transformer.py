"""Shallow Transformer for gene → isoform prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerIsoformPredictor(nn.Module):
    """
    Shallow Transformer model for gene → isoform prediction.
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

        # Learnable projection: map n_genes scalars → n_tokens scalars 
        self.token_proj = nn.Linear(n_genes, n_tokens, bias=False)

        # Learnable token embeddings 
        self.gene_embedding = nn.Linear(1, d_model)  

        # Learnable positional encoding
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
            norm_first=True,       
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

       
        self.output_head = nn.Linear(d_model, n_isoforms)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.token_proj(x)                 

        x = x.unsqueeze(-1)                    
        x = self.gene_embedding(x)             

       
        if self.use_cls and self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)  
            x = torch.cat([cls, x], dim=1)                  

        pos = self.pos_embedding[:, : x.size(1), :]
        x = x + pos                                     

        x = self.transformer(x)              

        x = self.norm(x)
        x = self.dropout(x)

        if self.use_cls and self.cls_token is not None:
            pooled = x[:, 0, :]                
        else:
            pooled = x.mean(dim=1)              

        logits = self.output_head(pooled)       

        return logits

"""Feed-forward architectures."""

from __future__ import annotations

import torch.nn as nn


class IsoformPredictor(nn.Module):
    """Simple 3-layer feed-forward network for multi-output regression."""

    def __init__(self, n_inputs: int, n_outputs: int, hidden_sizes=(1024, 512, 256), dropout=0.2):
        super().__init__()
        h1, h2, h3 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_inputs, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, n_outputs),
        )

    def forward(self, x):
        return self.net(x)

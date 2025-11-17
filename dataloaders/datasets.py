"""Dataset definitions."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class GeneIsoformDataset(Dataset):
    """Wrap numpy arrays for torch consumption."""

    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]

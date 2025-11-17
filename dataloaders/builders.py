"""Factories for datasets and dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from .datasets import GeneIsoformDataset


@dataclass
class GeneIsoformDataLoaders:
    train: DataLoader
    train_eval: DataLoader
    val: DataLoader
    test: DataLoader


def train_val_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return deterministic subset splits."""
    n = X.shape[0]
    total_needed = train_n + val_n + test_n
    if total_needed > n:
        raise ValueError(
            f"Requested {total_needed} samples (train {train_n}, val {val_n}, test {test_n}) "
            f"but dataset only has {n} observations."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)[:total_needed]
    X = X[perm]
    Y = Y[perm]
    train = (X[:train_n], Y[:train_n])
    val = (X[train_n : train_n + val_n], Y[train_n : train_n + val_n])
    test = (X[-test_n:], Y[-test_n:])
    return train, val, test


def make_loader(pair, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = GeneIsoformDataset(*pair)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

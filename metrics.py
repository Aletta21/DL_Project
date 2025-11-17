"""Metric utilities for isoform prediction."""

from __future__ import annotations

import numpy as np


def presence_accuracy(
    pred_counts: np.ndarray, true_counts: np.ndarray, threshold: float = 0.5
) -> float:
    """Binary accuracy on isoform presence (count > threshold)."""
    pred_presence = pred_counts > threshold
    true_presence = true_counts > threshold
    return float((pred_presence == true_presence).mean())

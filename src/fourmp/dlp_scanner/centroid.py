"""Sub-pixel line-center recovery from a captured (fuzzy) frame.

This is the algorithm intended to run on real captured data as well as
simulated frames — kept to a plain weighted mean per column so behavior is
easy to reason about; Gaussian-fit or parabolic-interpolation variants are a
future extension, not implemented here.
"""
from __future__ import annotations

import numpy as np


def centroid_rows(frame: np.ndarray, intensity_threshold: float = 1e-3) -> np.ndarray:
    """Per-column intensity-weighted centroid row. Shape (cols,), NaN where no signal."""
    rows, cols = frame.shape
    row_idx = np.arange(rows)
    centroids = np.full(cols, np.nan)
    for col in range(cols):
        weights = np.where(frame[:, col] > intensity_threshold, frame[:, col], 0.0)
        total = weights.sum()
        if total > 0:
            centroids[col] = float((row_idx * weights).sum() / total)
    return centroids

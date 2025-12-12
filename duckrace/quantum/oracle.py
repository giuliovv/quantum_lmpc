from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_adaptive_threshold(cost_table: np.ndarray, percentile: float = 25.0) -> float:
    costs = np.asarray(cost_table, dtype=float).reshape(-1)
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, percentile))


def marked_mask(cost_table: np.ndarray, threshold: float) -> np.ndarray:
    costs = np.asarray(cost_table, dtype=float).reshape(-1)
    return np.isfinite(costs) & (costs <= float(threshold))


def phase_oracle_diagonal(cost_table: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns the diagonal entries for a phase oracle:
      diag[i] = -1 if marked else +1
    """
    mask = marked_mask(cost_table, threshold)
    diag = np.where(mask, -1.0 + 0.0j, 1.0 + 0.0j)
    return diag


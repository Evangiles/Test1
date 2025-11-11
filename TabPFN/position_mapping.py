from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from . import metric


def map_to_positions(
    y_pred: np.ndarray,
    *,
    k: float,
    smoothing_alpha: float | None = None,
    delta_cap: float | None = None,
) -> np.ndarray:
    """
    Map prediction array to positions in [0, 2] using z-score scaling and optional smoothing.
    - z = (y - mean) / std
    - w = clip(1 + k*z, 0, 2)
    - EMA smoothing if smoothing_alpha provided
    - Rate limit if delta_cap provided (max |w_t - w_{t-1}| <= delta_cap)
    """
    y = np.asarray(y_pred).astype(float).reshape(-1)
    z = (y - y.mean()) / (y.std(ddof=0) + 1e-8)
    w = np.clip(1.0 + k * z, 0.0, 2.0)
    if smoothing_alpha is not None:
        smoothed = np.empty_like(w)
        smoothed[0] = w[0]
        for t in range(1, len(w)):
            smoothed[t] = smoothing_alpha * w[t] + (1 - smoothing_alpha) * smoothed[t - 1]
        w = smoothed
    if delta_cap is not None:
        limited = np.empty_like(w)
        limited[0] = w[0]
        for t in range(1, len(w)):
            delta = np.clip(w[t] - limited[t - 1], -delta_cap, delta_cap)
            limited[t] = np.clip(limited[t - 1] + delta, 0.0, 2.0)
        w = limited
    return w


def calibrate_k(
    solution_df: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    k_grid: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
    row_id_column_name: str = "date_id",
    smoothing_alpha: float | None = None,
    delta_cap: float | None = None,
) -> Tuple[float, float]:
    """
    Search k over grid to maximize adjusted Sharpe via metric.score, returning (best_k, best_score).
    """
    best_k = None
    best_score = -np.inf
    for k in k_grid:
        positions = map_to_positions(
            y_pred, k=k, smoothing_alpha=smoothing_alpha, delta_cap=delta_cap
        )
        submission = pd.DataFrame({"prediction": positions})
        score = metric.score(solution_df.copy(), submission, row_id_column_name=row_id_column_name)
        if score > best_score:
            best_score = score
            best_k = k
    if best_k is None:
        raise RuntimeError("Failed to find best k")
    return float(best_k), float(best_score)



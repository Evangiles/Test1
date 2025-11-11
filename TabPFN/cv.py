import numpy as np
import pandas as pd
from typing import List, Tuple


def purged_rolling_cv_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo_days: int = 5,
    date_col: str = "date_id",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged rolling cross-validation splits with embargo.

    - Each fold uses an expanding window for training up to (test_start - embargo).
    - Test windows are non-overlapping sequential blocks.
    - An embargo gap is kept between last train index and first test index.
    - Assumes df[date_col] increases with time (monotonic after sorting).
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in dataframe")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    dates = df_sorted[date_col].to_numpy()
    unique_dates = np.unique(dates)

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(unique_dates) < (n_splits + 1) * embargo_days + n_splits:
        # Very conservative minimal length check
        raise ValueError("Not enough unique dates for the requested splits and embargo.")

    # Create n_splits contiguous test blocks
    block_size = int(np.floor(len(unique_dates) / (n_splits + 1)))
    if block_size <= 0:
        block_size = max(1, len(unique_dates) // n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        test_start_ix = (i + 1) * block_size
        test_end_ix = min(test_start_ix + block_size, len(unique_dates))

        test_start_date = unique_dates[test_start_ix]
        test_end_date = unique_dates[test_end_ix - 1]

        # Train up to (test_start_ix - embargo_days)
        train_end_ix = max(0, test_start_ix - embargo_days)
        if train_end_ix == 0:
            # ensure at least some training samples exist
            continue

        train_mask = df_sorted[date_col] <= unique_dates[train_end_ix - 1]
        test_mask = (df_sorted[date_col] >= test_start_date) & (df_sorted[date_col] <= test_end_date)

        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)

        if len(train_idx) == 0 or len(test_idx) == 0:
            # skip degenerate folds
            continue

        splits.append((train_idx, test_idx))

    if len(splits) == 0:
        raise ValueError("Failed to create any valid CV folds. Check data length and parameters.")

    return splits



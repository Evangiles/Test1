from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Use a list (stable order) to avoid set-indexing issues in pandas
TARGET_COLUMNS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]


def _mad(series: pd.Series) -> float:
    med = series.median()
    return (series - med).abs().median()


@dataclass
class FoldSafePreprocessor:
    """
    Simple, fold-safe preprocessing:
    - Select numeric feature columns
    - Median imputation
    - MAD-based winsorization (criterion=4) on train stats
    - Standardization to z-score using train mean/std
    """
    feature_cols_: List[str] | None = None
    medians_: Dict[str, float] | None = None
    mads_: Dict[str, float] | None = None
    means_: Dict[str, float] | None = None
    stds_: Dict[str, float] | None = None

    def fit(self, df: pd.DataFrame) -> "FoldSafePreprocessor":
        # Identify numeric columns excluding targets and date_id
        candidate_cols = [
            c for c in df.columns
            if c not in TARGET_COLUMNS and c != "date_id" and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Drop columns that are entirely NaN in train
        candidate_cols = [c for c in candidate_cols if df[c].notna().any()]
        self.feature_cols_ = candidate_cols

        self.medians_ = {c: float(df[c].median()) for c in candidate_cols}
        self.mads_ = {c: float(_mad(df[c])) for c in candidate_cols}

        # Winsorize train copy to compute mean/std on clipped & imputed data
        df_proc = df[candidate_cols].copy()
        for c in candidate_cols:
            med = self.medians_[c]
            mad = self.mads_[c]
            crit = 4.0
            if mad == 0 or np.isnan(mad):
                # fallback to no clipping if MAD == 0
                low, high = -np.inf, np.inf
            else:
                low = med - crit * mad
                high = med + crit * mad
            s = df_proc[c].copy()
            s = s.fillna(med).clip(lower=low, upper=high)
            df_proc[c] = s

        self.means_ = {c: float(df_proc[c].mean()) for c in candidate_cols}
        self.stds_ = {c: float(df_proc[c].std(ddof=0)) for c in candidate_cols}
        # Filter out zero-variance or NaN-variance features
        kept_cols = [c for c in candidate_cols if np.isfinite(self.stds_[c]) and self.stds_[c] > 0]
        # Update internals to kept columns only
        self.feature_cols_ = kept_cols
        self.medians_ = {c: self.medians_[c] for c in kept_cols}
        self.mads_ = {c: self.mads_[c] for c in kept_cols}
        self.means_ = {c: self.means_[c] for c in kept_cols}
        # add small epsilon after filtering to avoid div by zero later
        self.stds_ = {c: float(self.stds_[c] + 1e-8) for c in kept_cols}
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if self.feature_cols_ is None:
            raise RuntimeError("Preprocessor not fitted")
        df_proc = df.copy()
        for c in self.feature_cols_:
            med = self.medians_[c]  # type: ignore[index]
            mad = self.mads_[c]  # type: ignore[index]
            mean = self.means_[c]  # type: ignore[index]
            std = self.stds_[c]  # type: ignore[index]
            crit = 4.0
            if mad == 0 or np.isnan(mad):
                low, high = -np.inf, np.inf
            else:
                low = med - crit * mad
                high = med + crit * mad
            s = df_proc[c].copy()
            s = s.fillna(med).clip(lower=low, upper=high)
            s = (s - mean) / std
            df_proc[c] = s
        return df_proc[self.feature_cols_].copy(), self.feature_cols_



from __future__ import annotations

from typing import Optional

import numpy as np

try:
    # expects TabPFN to be importable; runner will adjust sys.path if needed
    from tabpfn import TabPFNRegressor  # type: ignore
except ImportError as _e:  # pragma: no cover
    TabPFNRegressor = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e
    _IMPORT_DETAILS = str(_e)
except Exception as _e:
    TabPFNRegressor = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e
    _IMPORT_DETAILS = f"Unexpected error: {type(_e).__name__}: {_e}"
else:
    _IMPORT_ERROR = None
    _IMPORT_DETAILS = None


def make_tabpfn_regressor(
    n_estimators: int = 8,
    random_state: int = 0,
) -> Optional[object]:
    if TabPFNRegressor is None:
        return None
    # Create with minimal settings to avoid potential issues
    return TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        device="auto",  # Let it auto-detect
        ignore_pretraining_limits=True,  # Be more permissive
    )


def fit_predict_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    n_estimators: int = 8,
    random_state: int = 0,
) -> np.ndarray:
    model = make_tabpfn_regressor(n_estimators=n_estimators, random_state=random_state)
    if model is None:
        raise ImportError(f"Failed to import TabPFNRegressor: {_IMPORT_ERROR}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return np.asarray(y_pred).reshape(-1)



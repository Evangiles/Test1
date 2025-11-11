from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LinearRegression, Ridge


def get_sklearn_regressors(random_state: int = 42) -> Dict[str, object]:
    """
    Return a dict of baseline regressors for market_forward_excess_returns.
    Optional models (LightGBM, CatBoost) are included if available.
    """
    models: Dict[str, object] = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=random_state),
    }

    # Optional: LightGBM
    try:
        import lightgbm as lgb  # type: ignore

        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    # Optional: CatBoost
    try:
        from catboost import CatBoostRegressor  # type: ignore

        models["catboost"] = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=1000,
            loss_function="RMSE",
            verbose=False,
            random_seed=random_state,
        )
    except Exception:
        pass

    return models



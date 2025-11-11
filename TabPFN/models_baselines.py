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
            n_estimators=20,  # Reduced from 500
            learning_rate=0.3,  # Higher learning rate for shallow trees
            max_depth=3,  # Limit tree depth to avoid overfitting
            min_child_samples=20,  # Require more samples per leaf
            min_split_gain=0.001,  # Require min gain to split
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,  # Reduce warnings
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



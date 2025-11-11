from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cv import purged_rolling_cv_splits
from preprocessing_utils import FoldSafePreprocessor, TARGET_COLUMNS
from position_mapping import calibrate_k, K_GRID_DEFAULT, K_GRID_TABPFN
import metric
from models_baselines import get_sklearn_regressors
from models_tabpfn import fit_predict_tabpfn, make_tabpfn_regressor
from models_ensemble import fit_predict_weighted_ensemble


def _get_xy(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy().astype(float).reshape(-1)
    return X, y


def evaluate_models_with_cv(
    df: pd.DataFrame,
    *,
    date_col: str = "date_id",
    target_col: str = "market_forward_excess_returns",
    n_splits: int = 5,
    embargo_days: int = 5,
    include_tabpfn: bool = True,
    random_state: int = 42,
) -> Dict[str, dict]:
    """
    Run purged rolling CV for baselines (+ optional TabPFN) and compute adjusted Sharpe via metric.score.
    Returns a dict model_name -> {fold_scores: List[float], mean_score: float}
    """
    # ensure sorting and drop rows with missing target
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df = df[pd.notna(df[target_col]) & pd.notna(df["forward_returns"]) & pd.notna(df["risk_free_rate"])].copy()

    splits = purged_rolling_cv_splits(df, n_splits=n_splits, embargo_days=embargo_days, date_col=date_col)
    results: Dict[str, dict] = {}

    # Constant benchmark (always 1.0 position)
    bench_scores: List[float] = []
    for _, test_idx in splits:
        valid = df.iloc[test_idx]
        sol = valid[["forward_returns", "risk_free_rate"]].dropna().reset_index(drop=True)
        if len(sol) == 0:
            bench_scores.append(float("nan"))
            continue
        submission = pd.DataFrame({"prediction": np.clip(np.ones(len(sol)), 0, 2)})
        score_val = metric.score(sol, submission, row_id_column_name=date_col)
        bench_scores.append(float(score_val))
    results["benchmark_1.0"] = {"fold_scores": bench_scores, "mean_score": float(np.mean(bench_scores))}

    # Baseline regressors
    regressors = get_sklearn_regressors(random_state=random_state)
    for name, reg in regressors.items():
        fold_scores: List[float] = []
        for train_idx, test_idx in splits:
            train = df.iloc[train_idx].copy()
            valid = df.iloc[test_idx].copy()

            # Preprocess
            pre = FoldSafePreprocessor().fit(train)
            X_train, feat_cols = pre.transform(train)
            X_valid, _ = pre.transform(valid)

            Xtr, ytr = _get_xy(pd.concat([X_train, train[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)
            Xva, yva = _get_xy(pd.concat([X_valid, valid[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)

            # Fit/predict
            try:
                reg.fit(Xtr, ytr)
                y_pred = reg.predict(Xva).reshape(-1)
            except Exception:
                # Skip fold if model fails
                continue

            # Calibrate k and score (use default k_grid for baselines)
            solution = valid[["forward_returns", "risk_free_rate"]].copy()
            best_k, best_score = calibrate_k(solution, y_pred, k_grid=K_GRID_DEFAULT, row_id_column_name=date_col)
            fold_scores.append(best_score)
        if len(fold_scores) > 0:
            results[name] = {"fold_scores": fold_scores, "mean_score": float(np.mean(fold_scores))}

    # TabPFN regressor (with wider k_grid)
    if include_tabpfn and make_tabpfn_regressor() is not None:
        name = "tabpfn_v2_5"
        fold_scores = []
        for train_idx, test_idx in splits:
            train = df.iloc[train_idx].copy()
            valid = df.iloc[test_idx].copy()
            pre = FoldSafePreprocessor().fit(train)
            X_train, feat_cols = pre.transform(train)
            X_valid, _ = pre.transform(valid)
            Xtr, ytr = _get_xy(pd.concat([X_train, train[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)
            Xva, yva = _get_xy(pd.concat([X_valid, valid[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)
            try:
                y_pred = fit_predict_tabpfn(Xtr, ytr, Xva)
            except Exception:
                continue
            solution = valid[["forward_returns", "risk_free_rate"]].copy()
            # Use wider k_grid for TabPFN
            best_k, best_score = calibrate_k(solution, y_pred, k_grid=K_GRID_TABPFN, row_id_column_name=date_col)
            fold_scores.append(best_score)
        if len(fold_scores) > 0:
            results[name] = {"fold_scores": fold_scores, "mean_score": float(np.mean(fold_scores))}
    elif include_tabpfn:
        # Provide explicit reason when TabPFN is not available
        results["tabpfn_v2_5"] = {"fold_scores": [], "mean_score": None, "skip_reason": "TabPFN not importable (check sys.path to TabPFN/src and dependencies)"}

    # Ensemble: LightGBM + CatBoost + TabPFN (if all available)
    if "lightgbm" in results and "catboost" in results and "tabpfn_v2_5" in results:
        if (results["lightgbm"].get("mean_score") is not None and 
            results["catboost"].get("mean_score") is not None and 
            results["tabpfn_v2_5"].get("mean_score") is not None):
            
            name = "ensemble_lgb_cat_tabpfn"
            fold_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                train = df.iloc[train_idx].copy()
                valid = df.iloc[test_idx].copy()
                pre = FoldSafePreprocessor().fit(train)
                X_train, feat_cols = pre.transform(train)
                X_valid, _ = pre.transform(valid)
                Xtr, ytr = _get_xy(pd.concat([X_train, train[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)
                Xva, yva = _get_xy(pd.concat([X_valid, valid[list(TARGET_COLUMNS)]], axis=1), feat_cols, target_col)
                
                # Get predictions from all three models
                preds = {}
                try:
                    # LightGBM
                    lgb_models = get_sklearn_regressors(random_state)
                    if "lightgbm" in lgb_models:
                        lgb_models["lightgbm"].fit(Xtr, ytr)
                        preds["lightgbm"] = lgb_models["lightgbm"].predict(Xva).reshape(-1)
                    
                    # CatBoost
                    cat_models = get_sklearn_regressors(random_state)
                    if "catboost" in cat_models:
                        cat_models["catboost"].fit(Xtr, ytr)
                        preds["catboost"] = cat_models["catboost"].predict(Xva).reshape(-1)
                    
                    # TabPFN
                    preds["tabpfn"] = fit_predict_tabpfn(Xtr, ytr, Xva)
                    
                    if len(preds) == 3:
                        # Optimize ensemble weights on this fold
                        solution = valid[["forward_returns", "risk_free_rate"]].copy()
                        ensemble_pred, optimal_weights = fit_predict_weighted_ensemble(
                            preds, solution, weights=None, row_id_column_name=date_col
                        )
                        
                        # Score ensemble
                        best_k, best_score = calibrate_k(solution, ensemble_pred, k_grid=K_GRID_DEFAULT, row_id_column_name=date_col)
                        fold_scores.append(best_score)
                except Exception:
                    continue
            
            if len(fold_scores) > 0:
                results[name] = {"fold_scores": fold_scores, "mean_score": float(np.mean(fold_scores))}

    return results



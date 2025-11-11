"""Ensemble methods combining multiple models."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import metric
from position_mapping import calibrate_k, K_GRID_DEFAULT


def fit_predict_weighted_ensemble(
    predictions_dict: Dict[str, np.ndarray],
    solution_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    *,
    row_id_column_name: str = "date_id",
) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Combine multiple model predictions with optimal weights.
    
    Args:
        predictions_dict: {model_name: predictions_array}
        solution_df: DataFrame with forward_returns, risk_free_rate
        weights: Optional fixed weights. If None, optimizes weights on validation set.
        row_id_column_name: Column name for row IDs
        
    Returns:
        ensemble_predictions, optimal_weights
    """
    model_names = list(predictions_dict.keys())
    preds_array = np.column_stack([predictions_dict[name] for name in model_names])
    
    if weights is None:
        # Optimize weights using grid search
        weights = optimize_ensemble_weights(
            preds_array, 
            solution_df, 
            model_names,
            row_id_column_name=row_id_column_name
        )
    else:
        # Normalize provided weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    # Compute weighted average
    weight_array = np.array([weights.get(name, 0.0) for name in model_names])
    ensemble_pred = preds_array @ weight_array
    
    return ensemble_pred, weights


def optimize_ensemble_weights(
    preds_array: np.ndarray,
    solution_df: pd.DataFrame,
    model_names: List[str],
    *,
    row_id_column_name: str = "date_id",
    method: str = "grid",
) -> Dict[str, float]:
    """
    Find optimal ensemble weights by grid search or optimization.
    
    Args:
        preds_array: (n_samples, n_models) array of predictions
        solution_df: Validation data with forward_returns, risk_free_rate
        model_names: List of model names
        row_id_column_name: Column name for row IDs
        method: 'grid' for grid search, 'optimize' for continuous optimization
        
    Returns:
        Dict of optimal weights {model_name: weight}
    """
    n_models = preds_array.shape[1]
    
    if method == "grid":
        # Grid search over weight combinations
        if n_models == 2:
            weight_grid = [(w, 1-w) for w in np.arange(0, 1.1, 0.1)]
        elif n_models == 3:
            # For 3 models: coarse grid
            weight_grid = []
            for w1 in np.arange(0, 1.1, 0.2):
                for w2 in np.arange(0, 1.1-w1, 0.2):
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0:
                        weight_grid.append((w1, w2, w3))
        else:
            # Fallback: equal weights
            return {name: 1.0/n_models for name in model_names}
        
        best_score = -np.inf
        best_weights = None
        
        for weights_tuple in weight_grid:
            if len(weights_tuple) != n_models:
                continue
            
            # Compute ensemble prediction
            ensemble_pred = preds_array @ np.array(weights_tuple)
            
            # Calibrate k and score
            try:
                _, score = calibrate_k(
                    solution_df, 
                    ensemble_pred, 
                    k_grid=K_GRID_DEFAULT,
                    row_id_column_name=row_id_column_name
                )
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_weights = weights_tuple
            except Exception:
                continue
        
        if best_weights is None:
            # Fallback: equal weights
            best_weights = tuple([1.0/n_models] * n_models)
        
        return {name: weight for name, weight in zip(model_names, best_weights)}
    
    else:
        # scipy.optimize for continuous optimization
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = np.abs(weights)  # Ensure non-negative
            weights = weights / (weights.sum() + 1e-8)  # Normalize
            ensemble_pred = preds_array @ weights
            try:
                _, score = calibrate_k(
                    solution_df, 
                    ensemble_pred,
                    k_grid=K_GRID_DEFAULT,
                    row_id_column_name=row_id_column_name
                )
                return -score if np.isfinite(score) else 1e9
            except Exception:
                return 1e9
        
        # Initialize with equal weights
        x0 = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': False}
        )
        
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return {name: weight for name, weight in zip(model_names, optimal_weights)}


def meta_learner_ensemble(
    preds_dict: Dict[str, np.ndarray],
    y_train: np.ndarray,
    X_valid_preds: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Meta-learner approach: train a Ridge model on predictions as features.
    
    Args:
        preds_dict: Training predictions {model_name: train_predictions}
        y_train: Training targets
        X_valid_preds: Validation predictions {model_name: valid_predictions}
        
    Returns:
        Meta-model predictions on validation set
    """
    # Stack training predictions as features
    train_meta_features = np.column_stack([preds_dict[name] for name in preds_dict.keys()])
    
    # Train meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(train_meta_features, y_train)
    
    # Predict on validation
    valid_meta_features = np.column_stack([X_valid_preds[name] for name in preds_dict.keys()])
    ensemble_pred = meta_model.predict(valid_meta_features)
    
    return ensemble_pred


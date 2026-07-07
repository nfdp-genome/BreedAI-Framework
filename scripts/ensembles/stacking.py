"""Stacking utilities using non-negative ridge regression."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import r2_score

try:
    from scipy.optimize import lsq_linear

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def _solve_nonneg_ridge_pgd(
    p_mat: np.ndarray,
    y_vec: np.ndarray,
    alpha: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Projected gradient descent fallback for non-negative ridge."""
    n_models = p_mat.shape[1]
    w = np.zeros(n_models, dtype=float)
    gram = p_mat.T @ p_mat
    l_const = 2.0 * (np.linalg.norm(gram, ord=2) + alpha + 1e-12)
    lr = 1.0 / l_const
    prev_obj = np.inf

    for _ in range(max_iter):
        residual = p_mat @ w - y_vec
        grad = 2.0 * (p_mat.T @ residual + alpha * w)
        w = np.maximum(0.0, w - lr * grad)
        obj = float(np.dot(residual, residual) + alpha * np.dot(w, w))
        if prev_obj != np.inf:
            rel_improve = (prev_obj - obj) / max(prev_obj, 1e-12)
            if rel_improve < tol:
                break
        prev_obj = obj
    return w


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_nonneg_ridge(
    P_oof: np.ndarray,
    y: np.ndarray,
    alpha: float,
    fit_intercept: bool = True,
    standardize_cols: bool = True,
    normalize_weights: bool = False,
) -> Dict:
    """Fit non-negative ridge stacker using OOF base-model predictions."""
    p_mat = np.asarray(P_oof, dtype=float)
    y_vec = np.asarray(y, dtype=float).reshape(-1)
    if p_mat.ndim != 2:
        raise ValueError("P_oof must be a 2D array of shape (n_samples, n_models).")
    if y_vec.ndim != 1 or y_vec.shape[0] != p_mat.shape[0]:
        raise ValueError("y must be 1D with same number of samples as P_oof.")

    valid = np.isfinite(y_vec) & np.all(np.isfinite(p_mat), axis=1)
    p_mat = p_mat[valid]
    y_vec = y_vec[valid]
    if p_mat.shape[0] == 0:
        raise ValueError("No valid rows remain after filtering NaN/Inf values.")

    p_mean = np.mean(p_mat, axis=0) if fit_intercept else np.zeros(p_mat.shape[1], dtype=float)
    y_mean = float(np.mean(y_vec)) if fit_intercept else 0.0
    p_center = p_mat - p_mean
    y_center = y_vec - y_mean

    if standardize_cols:
        col_scale = np.std(p_center, axis=0)
        col_scale = np.where(col_scale > 1e-12, col_scale, 1.0)
    else:
        col_scale = np.ones(p_mat.shape[1], dtype=float)

    p_proc = p_center / col_scale
    alpha_used = float(alpha)

    if SCIPY_AVAILABLE:
        reg = np.sqrt(alpha_used) * np.eye(p_proc.shape[1])
        A = np.vstack([p_proc, reg])
        b = np.concatenate([y_center, np.zeros(p_proc.shape[1], dtype=float)])
        sol = lsq_linear(A, b, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto")
        w_scaled = sol.x
    else:
        w_scaled = _solve_nonneg_ridge_pgd(p_proc, y_center, alpha_used)

    weights = w_scaled / col_scale
    weights = np.maximum(0.0, weights)
    if normalize_weights:
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            weights = weights / w_sum

    intercept = y_mean - float(np.dot(p_mean, weights)) if fit_intercept else 0.0
    y_hat = predict_stacker(p_mat, weights, intercept)
    diagnostics = {
        "oof_rmse": _rmse(y_vec, y_hat),
        "oof_r2": float(r2_score(y_vec, y_hat)),
        "oof_corr": _corr(y_vec, y_hat),
    }
    return {
        "weights": weights,
        "intercept": float(intercept),
        "alpha_used": alpha_used,
        "diagnostics": diagnostics,
    }


def predict_stacker(P: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Predict stacked outputs from base-model prediction matrix."""
    p_mat = np.asarray(P, dtype=float)
    w_vec = np.asarray(weights, dtype=float).reshape(-1)
    return p_mat @ w_vec + float(intercept)


def summarize_weights(
    weights_by_fold: List[np.ndarray],
    model_names: List[str],
    model_families: Dict[str, str],
) -> Dict:
    """Summarize stacking weights across folds by model and family."""
    if not weights_by_fold:
        return {"model_summary": [], "family_summary": []}

    w_mat = np.vstack([np.asarray(w, dtype=float).reshape(1, -1) for w in weights_by_fold])
    if w_mat.shape[1] != len(model_names):
        raise ValueError("weights_by_fold width must match model_names length.")

    model_summary = []
    for i, model_name in enumerate(model_names):
        model_summary.append(
            {
                "model": model_name,
                "family": model_families.get(model_name, "other"),
                "mean_weight": float(np.mean(w_mat[:, i])),
                "sd_weight": float(np.std(w_mat[:, i], ddof=0)),
            }
        )

    families = sorted({model_families.get(m, "other") for m in model_names})
    family_summary = []
    for family in families:
        idx = [i for i, m in enumerate(model_names) if model_families.get(m, "other") == family]
        if not idx:
            continue
        family_fold_totals = np.sum(w_mat[:, idx], axis=1)
        family_summary.append(
            {
                "family": family,
                "mean_weight": float(np.mean(family_fold_totals)),
                "sd_weight": float(np.std(family_fold_totals, ddof=0)),
            }
        )

    return {"model_summary": model_summary, "family_summary": family_summary}

"""
Canonical GBLUP implementation for BreedAI.

Used by both the default track (06_default_track.py / s10_default_gblup.py)
and the R&D benchmarking track (02a_phase1_train_validate_array.py).

GBLUP = Ridge regression on the VanRaden G-matrix kernel.

Two entry points:
  - train_and_evaluate(): single split, returns metrics + predictions
  - make_ridge_model():   returns a configured sklearn Ridge/RidgeCV for
                          use inside the R&D algorithm registry
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {
        "r2": round(r2, 6),
        "pearson_r": round(r, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "bias": round(bias, 6),
    }


def train_and_evaluate(
    G: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray | None = None,
    alphas: list[float] | None = None,
    use_cv: bool = True,
) -> dict:
    """
    Train GBLUP (Ridge on G-matrix kernel) and return evaluation metrics.

    Parameters
    ----------
    G         : n x n genomic relationship matrix
    y         : n-length phenotype vector
    train_idx : training sample indices
    val_idx   : validation sample indices
    test_idx  : optional test sample indices
    alphas    : regularization candidates (used by RidgeCV)
    use_cv    : if True use RidgeCV, else use Ridge(alpha=1.0)
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    G_train = G[np.ix_(train_idx, train_idx)]
    y_train = y[train_idx]
    G_val = G[np.ix_(val_idx, train_idx)]
    y_val = y[val_idx]

    if use_cv:
        model = RidgeCV(alphas=alphas, store_cv_results=True)
    else:
        model = Ridge(alpha=1.0)

    model.fit(G_train, y_train)
    alpha_used = float(model.alpha_) if hasattr(model, "alpha_") else 1.0

    val_pred = model.predict(G_val)
    result = {
        "alpha": alpha_used,
        "val_metrics": metrics(y_val, val_pred),
        "val_predictions": val_pred.tolist(),
        "model": model,
    }

    if test_idx is not None and len(test_idx) > 0:
        G_test = G[np.ix_(test_idx, train_idx)]
        y_test = y[test_idx]
        test_pred = model.predict(G_test)
        result["test_metrics"] = metrics(y_test, test_pred)
        result["test_predictions"] = test_pred.tolist()

    return result


def make_ridge_model(alpha: float = 1.0, random_state: int = 42) -> Ridge:
    """Return a configured Ridge for use in the R&D algorithm registry."""
    return Ridge(alpha=alpha, random_state=random_state)


def make_ridgecv_model(alphas: list[float] | None = None) -> RidgeCV:
    """Return a configured RidgeCV for use in the R&D algorithm registry."""
    if alphas is None:
        alphas = np.logspace(-4, 4, 20).tolist()
    return RidgeCV(alphas=alphas, cv=5)

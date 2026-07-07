import importlib.util
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from ensembles.stacking import fit_nonneg_ridge
except ImportError:
    from breedai.ensemble.stacking import fit_nonneg_ridge


def _load_phase1_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "02a_phase1_train_validate_array.py"
    spec = importlib.util.spec_from_file_location("phase1_train_validate_array", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_nonneg_ridge_weights_non_negative():
    rng = np.random.default_rng(7)
    y = rng.normal(size=120)
    P = np.column_stack(
        [
            y + 0.02 * rng.normal(size=120),
            rng.normal(size=120),
            rng.normal(size=120),
        ]
    )
    out = fit_nonneg_ridge(P, y, alpha=1e-2)
    assert np.all(out["weights"] >= -1e-12)


def test_fit_nonneg_ridge_prefers_best_predictor():
    rng = np.random.default_rng(17)
    y = rng.normal(size=200)
    P = np.column_stack(
        [
            y,  # perfect base predictor
            0.5 * y + 0.5 * rng.normal(size=200),
            rng.normal(size=200),
        ]
    )
    out = fit_nonneg_ridge(P, y, alpha=1e-2)
    top_idx = int(np.argmax(out["weights"]))
    assert top_idx == 0


def test_fit_nonneg_ridge_deterministic():
    rng = np.random.default_rng(23)
    y = rng.normal(size=160)
    P = np.column_stack([rng.normal(size=160), y + 0.1 * rng.normal(size=160)])
    out_a = fit_nonneg_ridge(P, y, alpha=0.05)
    out_b = fit_nonneg_ridge(P, y, alpha=0.05)
    np.testing.assert_allclose(out_a["weights"], out_b["weights"], atol=1e-10, rtol=1e-10)
    assert abs(out_a["intercept"] - out_b["intercept"]) < 1e-12


def test_stacking_integration_creates_weight_artifacts(tmp_path):
    phase1_module = _load_phase1_module()
    GenomicBenchmarkingArray = phase1_module.GenomicBenchmarkingArray

    rng = np.random.default_rng(101)
    n = 90
    p = 8
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 0.1 * rng.normal(size=n)

    n_train = 60
    n_val = 15
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    benchmarker = GenomicBenchmarkingArray(
        output_dir=str(tmp_path),
        ensemble_options=["stacking_nonneg_ridge"],
        random_state=42,
        stack_n_splits=3,
    )
    algorithms = {
        "GBLUP_Ridge": Ridge(alpha=1.0, random_state=42),
        "LASSO": Lasso(alpha=0.01, max_iter=2000, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=42),
    }

    result = benchmarker._run_stacking_ensemble(
        trait_name="Trait_Toy",
        algorithms=algorithms,
        successful_algorithms=list(algorithms.keys()),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    assert result is not None
    assert result["algorithm"] == "Ensemble_Stacking_NonNeg_Ridge"

    out_dir = tmp_path / "stacking_Trait_Toy"
    assert (out_dir / "stacking_weights_by_fold.csv").exists()
    assert (out_dir / "stacking_weights_summary.csv").exists()
    assert (out_dir / "stacking_family_weights_summary.csv").exists()
    assert (out_dir / "oof_predictions_stacking.npz").exists()

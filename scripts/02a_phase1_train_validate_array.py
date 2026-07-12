#!/usr/bin/env python3
"""
File: 02a_phase1_train_validate_array.py
Purpose: Array job genomic prediction train-validate (algorithm selection) - ONE JOB PER TRAIT
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import pickle
import argparse
import logging
import subprocess
import tempfile
import shutil
import resource
import importlib.util
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ensembles.stacking import fit_nonneg_ridge, predict_stacker, summarize_weights
except ImportError:
    from breedai.ensemble.stacking import fit_nonneg_ridge, predict_stacker, summarize_weights

# Core genomic prediction methods
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LassoCV, ElasticNetCV, RidgeCV, ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Advanced methods
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    import gpflow
    GPFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    GPFLOW_AVAILABLE = False

def _get_rscript_bin():
    """Return Rscript path from env or PATH."""
    env_path = os.environ.get("BGLR_RSCRIPT", "").strip()
    if env_path:
        return env_path
    return shutil.which("Rscript") or ""

def _check_bglr_available(rscript_bin):
    """Check whether Rscript can load the BGLR package.

    BGLR is only needed for the Bayesian methods (BayesA/B/Cpi). The probe is
    bounded by a timeout so a slow or hanging R install can never block the
    pipeline (e.g. a cold compute node loading BGLR over a network home dir).
    - Set BREEDAI_SKIP_BGLR_CHECK=1 to skip the probe entirely (default track).
    - Set BREEDAI_BGLR_CHECK_TIMEOUT to change the timeout in seconds (default 60).
    On timeout/failure it returns False -> the Bayesian methods are skipped.
    """
    if not rscript_bin:
        return False
    if os.environ.get("BREEDAI_SKIP_BGLR_CHECK", "0").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        timeout_s = float(os.environ.get("BREEDAI_BGLR_CHECK_TIMEOUT", "60"))
    except ValueError:
        timeout_s = 60.0
    try:
        result = subprocess.run(
            [rscript_bin, "-e", "suppressMessages(library(BGLR))"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=timeout_s,
        )
        return result.returncode == 0
    except Exception:
        return False

RSCRIPT_BIN = _get_rscript_bin()
BGLR_AVAILABLE = _check_bglr_available(RSCRIPT_BIN)

# #region agent log
def _debug_log(hypothesis_id, location, message, data=None):
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open("${BREEDAI_DEBUG_LOG:-/dev/null}", "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion agent log


def _get_rss_mb():
    """Return current process RSS in MB (for memory debugging). Linux: ru_maxrss is KB."""
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_kb / 1024.0
    except Exception:
        return float("nan")


MODEL_FAMILY_MAP = {
    "GBLUP_Ridge": "linear",
    "GBLUP_RidgeCV": "linear",
    "LASSO": "linear",
    "LASSO_CV": "linear",
    "ElasticNet": "linear",
    "ElasticNet_CV": "linear",
    "BayesianRidge": "linear",
    "RandomForest": "tree_boosting",
    "XGBoost": "tree_boosting",
    "LightGBM": "tree_boosting",
    "SVR_RBF": "kernel_prob",
    "SVR_Linear": "kernel_prob",
    "GP_RBF": "kernel_prob",
    "GP_Matern52": "kernel_prob",
    "NeuralNet_MLP": "neural",
    "BayesA": "bayesian_alphabet",
    "BayesB": "bayesian_alphabet",
    "BayesCpi": "bayesian_alphabet",
}


# #region agent log
_debug_log(
    "H1",
    "02a_phase1_train_validate_array.py:88",
    "BGLR env and availability",
    {
        "BGLR_RSCRIPT": os.environ.get("BGLR_RSCRIPT", ""),
        "BREEDAI_REQUIRE_ALL_ALGOS": os.environ.get("BREEDAI_REQUIRE_ALL_ALGOS", ""),
        "which_Rscript": shutil.which("Rscript") or "",
        "RSCRIPT_BIN": RSCRIPT_BIN or "",
        "BGLR_AVAILABLE": BGLR_AVAILABLE,
    },
)
# #endregion agent log

class BGLRRegressor(BaseEstimator, RegressorMixin):
    """Bayesian Genomic Regressor via R/BGLR backend (BayesA/B/Cpi)."""
    
    def __init__(self, method="BayesA", n_iter=12000, burn_in=2000, seed=42, r_script_path=None):
        self.method = method
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.seed = seed
        self.r_script_path = r_script_path or str(Path(__file__).parent / "02g_backend_bglr.R")
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        if not BGLR_AVAILABLE:
            raise ImportError("Rscript/BGLR not available")
        
        r_script = Path(self.r_script_path)
        if not r_script.exists():
            raise FileNotFoundError(f"BGLR backend script not found: {r_script}")
        
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            X_path = tmpdir / "X.csv"
            y_path = tmpdir / "y.csv"
            b_path = tmpdir / "b.csv"
            mu_path = tmpdir / "mu.csv"
            
            np.savetxt(X_path, X, delimiter=",")
            np.savetxt(y_path, y, delimiter=",")
            
            cmd = [
                RSCRIPT_BIN or "Rscript",
                str(r_script),
                str(X_path),
                str(y_path),
                self.method,
                str(b_path),
                str(mu_path),
                str(self.n_iter),
                str(self.burn_in),
                str(self.seed),
            ]
            
            # #region agent log
            _debug_log(
                "H8",
                "02a_phase1_train_validate_array.py:122",
                "BGLR subprocess start",
                {"cmd": cmd, "r_script": str(r_script)},
            )
            # #endregion agent log
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # #region agent log
            _debug_log(
                "H8",
                "02a_phase1_train_validate_array.py:127",
                "BGLR subprocess result",
                {"returncode": result.returncode, "stderr": result.stderr.strip()},
            )
            # #endregion agent log
            if result.returncode != 0:
                raise RuntimeError(
                    f"BGLR backend failed (method={self.method}): {result.stderr.strip()}"
                )
            
            # #region agent log
            try:
                b_first = open(b_path, "r").readline().strip()
            except Exception:
                b_first = ""
            try:
                mu_first = open(mu_path, "r").readline().strip()
            except Exception:
                mu_first = ""
            _debug_log(
                "H9",
                "02a_phase1_train_validate_array.py:138",
                "BGLR output file heads",
                {"b_first_line": b_first, "mu_first_line": mu_first},
            )
            # #endregion agent log
            self.coef_ = np.loadtxt(b_path, delimiter=",")
            mu_val = np.loadtxt(mu_path, delimiter=",")
            self.intercept_ = float(mu_val) if np.ndim(mu_val) == 0 else float(mu_val[0])
        
        return self
    
    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

from scipy.stats import pearsonr, spearmanr

def _load_module_from_file(module_filename, module_name):
    module_path = Path(__file__).parent / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import Gmatrix calculation function
try:
    _gmatrix_module = _load_module_from_file("01a_utils_calculate_gmatrix.py", "utils_calculate_gmatrix")
    calculate_gmatrix = _gmatrix_module.calculate_gmatrix
    GMATRIX_AVAILABLE = True
except Exception:
    GMATRIX_AVAILABLE = False

class GPflowRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible GPflow wrapper for train-validate"""
    
    def __init__(self, kernel_type='rbf', max_iter=300, lengthscale=1.0, 
                 variance=1.0, noise_variance=0.1, use_sparse=True, n_inducing=50):
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_variance = noise_variance
        self.use_sparse = use_sparse
        self.n_inducing = n_inducing
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def _create_kernel(self, input_dim):
        if self.kernel_type == 'rbf':
            return gpflow.kernels.RBF(lengthscales=self.lengthscale, variance=self.variance)
        elif self.kernel_type == 'matern52':
            return gpflow.kernels.Matern52(lengthscales=self.lengthscale, variance=self.variance)
        elif self.kernel_type == 'linear':
            return gpflow.kernels.Linear(variance=self.variance)
        else:
            return gpflow.kernels.RBF(lengthscales=self.lengthscale, variance=self.variance)
    
    def fit(self, X, y):
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow not available")
            
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_tensor = tf.constant(X_scaled, dtype=tf.float64)
        y_tensor = tf.constant(y_scaled.reshape(-1, 1), dtype=tf.float64)
        
        kernel = self._create_kernel(X.shape[1])
        
        if self.use_sparse and len(X) > 500:
            n_inducing = min(self.n_inducing, len(X) // 2)
            indices = np.random.choice(len(X), n_inducing, replace=False)
            inducing_points = X_scaled[indices]
            
            self.model = gpflow.models.SGPR(
                data=(X_tensor, y_tensor),
                kernel=kernel,
                inducing_variable=inducing_points
            )
        else:
            self.model = gpflow.models.GPR(
                data=(X_tensor, y_tensor),
                kernel=kernel,
                noise_variance=self.noise_variance
            )
        
        optimizer = gpflow.optimizers.Scipy()
        try:
            optimizer.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options=dict(maxiter=self.max_iter)
            )
        except:
            pass
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler_X.transform(X)
        X_tensor = tf.constant(X_scaled, dtype=tf.float64)
        
        mean, _ = self.model.predict_f(X_tensor)
        mean_np = mean.numpy().flatten()
        predictions_unscaled = self.scaler_y.inverse_transform(mean_np.reshape(-1, 1)).flatten()
        
        return predictions_unscaled

class ArrayJobManager:
    """Manages job array execution for genomic train-validate - ONE JOB PER TRAIT"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'train_validate'
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'array_manager.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_trait_jobs(self, trait_names, algorithms):
        """Create one job per trait - each job processes all algorithms for that trait"""
        jobs = []
        
        for job_id, trait_name in enumerate(trait_names):
            jobs.append({
                'job_id': job_id,
                'trait_name': trait_name,
                'algorithms': algorithms  # All algorithms for this trait
            })
        
        self.logger.info(f"Created {len(jobs)} trait jobs (one per trait)")
        return jobs
    
    def save_job_data(self, data, job_id, data_type='job'):
        """Save job data for array processing"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return str(filepath)
    
    def load_job_data(self, job_id, data_type='job'):
        """Load specific job data"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Job data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_job_results(self, results, job_id):
        """Save results for a specific job"""
        filename = f"results_{job_id:04d}.pkl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Saved results for job {job_id}")
        return str(filepath)
    
    def combine_all_results(self, n_jobs):
        """Combine results from all completed jobs"""
        combined_results = []
        successful_jobs = 0
        
        for job_id in range(n_jobs):
            filename = f"results_{job_id:04d}.pkl"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        job_results = pickle.load(f)
                    combined_results.extend(job_results)
                    successful_jobs += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load job {job_id}: {e}")
            else:
                self.logger.warning(f"Missing results for job {job_id}")
        
        self.logger.info(f"Combined {successful_jobs} successful jobs")
        # #region agent log
        algo_counts = {}
        for row in combined_results:
            algo = row.get("algorithm")
            if algo:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
        _debug_log(
            "H7",
            "02a_phase1_train_validate_array.py:375",
            "Combined results algorithm counts",
            {"successful_jobs": successful_jobs, "algo_counts": algo_counts},
        )
        # #endregion agent log
        return combined_results
    
    def generate_slurm_array_script(self, n_jobs, job_name='genomic_train_validate', extra_cli_args=""):
        """Generate SLURM array job script"""
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{n_jobs-1}
#SBATCH --time=14-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=batch
#SBATCH --output=$PROJECT_DIR/logs/train_validate/{job_name}_%A_%a.out
#SBATCH --error=$PROJECT_DIR/logs/train_validate/{job_name}_%A_%a.err

# Create logs directory
mkdir -p $PROJECT_DIR/logs/train_validate

echo "=================================================================="
echo "GENOMIC TRAIN-VALIDATE ARRAY JOB - ONE TRAIT PER JOB"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=================================================================="

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="${{CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${{CONDA_BASE_PATH}}" ] && [ -f "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh" ] && source "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh"
_ep=$(CONDA_BASE_PATH="$CONDA_BASE_PATH" ENVNAME="$USER_CONDA_ENV_PATH" timeout 45 bash -c 'source "$CONDA_BASE_PATH/etc/profile.d/conda.sh" 2>/dev/null; conda activate "$ENVNAME" >/dev/null 2>&1; printf %s "$CONDA_PREFIX"' 2>/dev/null || true)
if [ -n "$_ep" ] && [ -x "$_ep/bin/python" ]; then export CONDA_PREFIX="$_ep"; export PATH="$_ep/bin:$PATH"; export CONDA_DEFAULT_ENV="$USER_CONDA_ENV_PATH"; fi

# Fix for CXXABI error
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

# Verify environment
if [ -z "$CONDA_PREFIX" ] || ( [ "$CONDA_PREFIX" != "$USER_CONDA_ENV_PATH" ] && [ "$(basename "$CONDA_PREFIX")" != "$(basename "$USER_CONDA_ENV_PATH")" ] ); then
    echo "ERROR: Failed to activate Conda environment"
    exit 1
fi

echo "Python Version: $(python --version)"
echo "Environment: $CONDA_PREFIX"

# Set threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Project paths - automatically detect BreedAI-Framework directory
# Function to find project root by looking for environment.yml
find_project_dir() {{
    local search_dir="$1"
    while [[ "$search_dir" != "/" ]]; do
        if [[ -f "$search_dir/environment.yml" ]] && [[ -d "$search_dir/scripts" ]]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    return 1
}}

# Try multiple methods to find project directory
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    PROJECT_DIR="$(find_project_dir "$SLURM_SUBMIT_DIR")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$(find_project_dir "$PWD")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: Could not find BreedAI-Framework directory"
    exit 1
fi

SCRIPTS_DIR="$PROJECT_DIR/scripts"

cd "$SCRIPTS_DIR"

echo "Processing trait job $SLURM_ARRAY_TASK_ID (all algorithms for one trait)"

# Run specific job
python3 02a_phase1_train_validate_array.py \\
    --mode process_job \\
    --job_id $SLURM_ARRAY_TASK_ID \\
    --output_dir {self.output_dir} \\
    --n_jobs $SLURM_CPUS_PER_TASK {extra_cli_args}

JOB_EXIT_CODE=$?

echo "=================================================================="
echo "Trait job $SLURM_ARRAY_TASK_ID completed with exit code: $JOB_EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================================================="

exit $JOB_EXIT_CODE
"""
        
        script_path = self.output_dir / f"{job_name}_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated array script: {script_path}")
        return str(script_path)
    
    def generate_combine_script(self, n_jobs, job_name='genomic_combine'):
        """Generate results combination script"""
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output=$PROJECT_DIR/logs/train_validate/{job_name}_%j.out
#SBATCH --error=$PROJECT_DIR/logs/train_validate/{job_name}_%j.err

echo "=================================================================="
echo "GENOMIC TRAIN-VALIDATE RESULTS COMBINATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=================================================================="

mkdir -p logs

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="${{CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${{CONDA_BASE_PATH}}" ] && [ -f "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh" ] && source "${{CONDA_BASE_PATH}}/etc/profile.d/conda.sh"
_ep=$(CONDA_BASE_PATH="$CONDA_BASE_PATH" ENVNAME="$USER_CONDA_ENV_PATH" timeout 45 bash -c 'source "$CONDA_BASE_PATH/etc/profile.d/conda.sh" 2>/dev/null; conda activate "$ENVNAME" >/dev/null 2>&1; printf %s "$CONDA_PREFIX"' 2>/dev/null || true)
if [ -n "$_ep" ] && [ -x "$_ep/bin/python" ]; then export CONDA_PREFIX="$_ep"; export PATH="$_ep/bin:$PATH"; export CONDA_DEFAULT_ENV="$USER_CONDA_ENV_PATH"; fi
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"

if [ -z "$CONDA_PREFIX" ] || ( [ "$CONDA_PREFIX" != "$USER_CONDA_ENV_PATH" ] && [ "$(basename "$CONDA_PREFIX")" != "$(basename "$USER_CONDA_ENV_PATH")" ] ); then
    echo "ERROR: Failed to activate Conda environment"
    exit 1
fi

# Project paths - automatically detect BreedAI-Framework directory
# Function to find project root by looking for environment.yml
find_project_dir() {{
    local search_dir="$1"
    while [[ "$search_dir" != "/" ]]; do
        if [[ -f "$search_dir/environment.yml" ]] && [[ -d "$search_dir/scripts" ]]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    return 1
}}

# Try multiple methods to find project directory
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    PROJECT_DIR="$(find_project_dir "$SLURM_SUBMIT_DIR")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$(find_project_dir "$PWD")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: Could not find BreedAI-Framework directory"
    exit 1
fi

SCRIPTS_DIR="$PROJECT_DIR/scripts"

cd "$SCRIPTS_DIR"

echo "Combining results from {n_jobs} trait jobs..."

python3 02a_phase1_train_validate_array.py \\
    --mode combine_results \\
    --n_jobs {n_jobs} \\
    --output_dir {self.output_dir}

COMBINE_EXIT_CODE=$?

if [[ $COMBINE_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Generating comprehensive reports..."

    # Create Phase 1 folder for results
    mkdir -p "$PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking"

    # Generate Jupyter notebook report
    python3 02b_phase1_report_benchmarking.py \\
        --results_file {self.output_dir}/combined_train_validate_results.csv \\
        --output_file "$PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking/1.2_Learning_Benchmarking_report.ipynb"

    NOTEBOOK_EXIT_CODE=$?

    if [[ $NOTEBOOK_EXIT_CODE -eq 0 ]]; then
        echo "✅ Jupyter notebook report generated successfully"
        echo "📁 Location: $PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking/1.2_Learning_Benchmarking_report.ipynb"
    else
        echo "❌ Jupyter notebook report generation failed"
    fi

    # Generate Phase 1 EDA/QC report (Geno QC, Pheno QC, G-matrix)
    echo ""
    echo "Generating Phase 1 EDA/QC report (Geno QC, Pheno QC, G-matrix)..."
    
    # Find dataset directory
    if [[ -d "$PROJECT_DIR/input" ]]; then
        DATASET_DIR="$PROJECT_DIR/input"
    else
        DATASET_DIR="$PROJECT_DIR/data"
    fi
    
    # G-matrix directory
    GMATRIX_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking/QC/gmatrix"
    
    # Output file
    EDA_QC_REPORT="$PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking/1.1_Preprocessing_report.ipynb"
    
    python3 02c_phase1_report_preprocessing.py \\
        --dataset_dir "$DATASET_DIR" \\
        --gmatrix_dir "$GMATRIX_DIR" \\
        --output_file "$EDA_QC_REPORT"
    
    EDA_QC_EXIT_CODE=$?
    
    if [[ $EDA_QC_EXIT_CODE -eq 0 ]]; then
        echo "✅ Phase 1 EDA/QC report generated successfully"
        echo "📁 Location: $EDA_QC_REPORT"
        echo "   Report includes: Geno QC, Pheno QC, G-matrix analysis"
    else
        echo "❌ Phase 1 EDA/QC report generation failed"
        echo "   You can generate it manually using:"
        echo "   python3 02c_phase1_report_preprocessing.py --dataset_dir $DATASET_DIR --gmatrix_dir $GMATRIX_DIR --output_file $EDA_QC_REPORT"
    fi

    echo ""
    echo "🎯 Phase 1 results available at:"
    echo "   📊 Reports: $PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking/"
    echo "     ├── 1.2_Learning_Benchmarking_report.ipynb (Model Performance)"
    echo "     └── 1.1_Preprocessing_report.ipynb (EDA/QC: Geno, Pheno, G-matrix)"
    echo "   📋 Raw data: $PROJECT_DIR/Phase1_Learning_Benchmarking/"
    echo "     ├── QC/: Quality control files (G-matrix, preprocessing)"
    echo "     └── training_validation/: Training results and metrics"

else
    echo "❌ Results combination failed - skipping report generation"
fi

echo "=================================================================="
echo "Results combination completed with exit code: $COMBINE_EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================================================="

exit $COMBINE_EXIT_CODE
"""
        
        script_path = self.output_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated combine script: {script_path}")
        return str(script_path)

class GenomicBenchmarkingArray:
    """Genomic train-validate (algorithm selection) with array job support - ONE JOB PER TRAIT"""

    def __init__(
        self,
        output_dir,
        qc_dir=None,
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        ensemble_options=None,
        stack_alpha=0.01,
        stack_fit_intercept=True,
        stack_standardize_cols=True,
        stack_normalize_weights=False,
        stack_n_splits=5,
        stack_outer_splits=5,
        stack_inner_splits=3,
    ):
        self.output_dir = Path(output_dir)
        self.qc_dir = Path(qc_dir) if qc_dir else self.output_dir.parent / 'QC'
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size  # Validation set size (of remaining data after test split)
        self.ensemble_options = set(ensemble_options or [])
        self.stack_alpha = stack_alpha
        self.stack_fit_intercept = stack_fit_intercept
        self.stack_standardize_cols = stack_standardize_cols
        self.stack_normalize_weights = stack_normalize_weights
        self.stack_n_splits = stack_n_splits
        if stack_outer_splits == 5 and stack_n_splits != 5:
            self.stack_outer_splits = stack_n_splits
        else:
            self.stack_outer_splits = stack_outer_splits
        self.stack_inner_splits = stack_inner_splits
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'train_validate'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_validate_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_algorithms(self):
        """Get all available algorithms"""
        require_all = os.environ.get("BREEDAI_REQUIRE_ALL_ALGOS", "1") != "0"
        # #region agent log
        _debug_log(
            "H2",
            "02a_phase1_train_validate_array.py:652",
            "Algorithm availability before selection",
            {
                "require_all": require_all,
                "BGLR_AVAILABLE": BGLR_AVAILABLE,
                "GPFLOW_AVAILABLE": GPFLOW_AVAILABLE,
                "XGBOOST_AVAILABLE": XGBOOST_AVAILABLE,
                "LIGHTGBM_AVAILABLE": LIGHTGBM_AVAILABLE,
            },
        )
        # #endregion agent log
        if require_all:
            missing = []
            if not BGLR_AVAILABLE:
                missing.append("BayesA/BayesB/BayesCpi (set BGLR_RSCRIPT or add Rscript+BGLR)")
            if not GPFLOW_AVAILABLE:
                missing.append("GP_RBF/GP_Matern52 (install tensorflow+gpflow)")
            if not XGBOOST_AVAILABLE:
                missing.append("XGBoost (install xgboost)")
            if not LIGHTGBM_AVAILABLE:
                missing.append("LightGBM (install lightgbm)")
            if missing:
                # #region agent log
                _debug_log(
                    "H3",
                    "02a_phase1_train_validate_array.py:669",
                    "Missing required algorithms",
                    {"missing": missing},
                )
                # #endregion agent log
                raise RuntimeError(
                    "Required algorithms are unavailable: "
                    + "; ".join(missing)
                    + ". Set BREEDAI_REQUIRE_ALL_ALGOS=0 to allow skipping."
                )

        algorithms = {
            'GBLUP_Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'GBLUP_RidgeCV': RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5),
            'LASSO': Lasso(alpha=0.01, max_iter=2000, random_state=self.random_state),
            'LASSO_CV': LassoCV(cv=5, max_iter=2000, n_alphas=50, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=self.random_state),
            'ElasticNet_CV': ElasticNetCV(cv=5, max_iter=2000, n_alphas=50, random_state=self.random_state),
            'BayesianRidge': BayesianRidge(compute_score=False),  # compute_score=True is slow/heavy on large p
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=4
            ),
            'SVR_RBF': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR_Linear': SVR(kernel='linear', C=1.0),
        }
        
        if BGLR_AVAILABLE:
            algorithms.update({
                'BayesA': BGLRRegressor(method='BayesA', seed=self.random_state),
                'BayesB': BGLRRegressor(method='BayesB', seed=self.random_state),
                'BayesCpi': BGLRRegressor(method='BayesCpi', seed=self.random_state),
            })
            self.logger.info("Added BayesA/B/Cpi (R/BGLR)")
            # #region agent log
            _debug_log(
                "H4",
                "02a_phase1_train_validate_array.py:690",
                "Bayes algorithms added",
                {"algorithms_added": ["BayesA", "BayesB", "BayesCpi"]},
            )
            # #endregion agent log
        
        algorithms['NeuralNet_MLP'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=self.random_state
        )
        self.logger.info("Added Neural Network (MLPRegressor)")
        
        if GPFLOW_AVAILABLE:
            algorithms.update({
                'GP_RBF': GPflowRegressor(
                    kernel_type='rbf', max_iter=300, use_sparse=True, n_inducing=50
                ),
                'GP_Matern52': GPflowRegressor(
                    kernel_type='matern52', max_iter=300, use_sparse=True, n_inducing=50
                )
            })
            self.logger.info("Added Gaussian Process algorithms")
        
        if XGBOOST_AVAILABLE:
            algorithms['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=4, verbosity=0
            )
            self.logger.info("Added XGBoost algorithm")
        
        if LIGHTGBM_AVAILABLE:
            algorithms['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=4, verbose=-1
            )
            self.logger.info("Added LightGBM algorithm")
        
        return algorithms

    def _build_pipeline(self, alg_name, algorithm):
        """Construct per-algorithm pipeline consistently."""
        if alg_name.startswith('GP_'):
            return clone(algorithm)
        if alg_name.startswith('SVR') or alg_name == 'NeuralNet_MLP':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', clone(algorithm))
            ])
        return Pipeline([('regressor', clone(algorithm))])

    def _stack_param_grid(self, alg_name):
        """Small inner-CV tuning grids for nested CV."""
        grids = {
            'GBLUP_Ridge': [{'regressor__alpha': 0.1}, {'regressor__alpha': 1.0}, {'regressor__alpha': 10.0}],
            'LASSO': [{'regressor__alpha': 0.001}, {'regressor__alpha': 0.01}, {'regressor__alpha': 0.1}],
            'ElasticNet': [
                {'regressor__alpha': 0.01, 'regressor__l1_ratio': 0.3},
                {'regressor__alpha': 0.01, 'regressor__l1_ratio': 0.5},
                {'regressor__alpha': 0.05, 'regressor__l1_ratio': 0.7},
            ],
            'RandomForest': [
                {'regressor__n_estimators': 150, 'regressor__max_depth': 8},
                {'regressor__n_estimators': 200, 'regressor__max_depth': 10},
            ],
            'SVR_RBF': [{'regressor__C': 0.5}, {'regressor__C': 1.0}],
            'SVR_Linear': [{'regressor__C': 0.5}, {'regressor__C': 1.0}],
            'NeuralNet_MLP': [
                {'regressor__hidden_layer_sizes': (64, 32)},
                {'regressor__hidden_layer_sizes': (128, 64, 32)},
            ],
            'XGBoost': [{'regressor__max_depth': 4}, {'regressor__max_depth': 6}],
            'LightGBM': [{'regressor__max_depth': 4}, {'regressor__max_depth': 6}],
        }
        # Leave expensive/CV/self-tuning models on default params.
        return grids.get(alg_name, [{}])

    def _tune_model_and_get_oof(self, alg_name, algorithm, X, y, n_splits):
        """Tune model via inner CV and return best params + OOF predictions."""
        n_splits = max(2, min(n_splits, len(y)))
        inner_kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        candidates = self._stack_param_grid(alg_name)
        best = {"score": -np.inf, "params": {}, "oof": None}

        for params in candidates:
            oof = np.full(len(y), np.nan, dtype=float)
            fold_scores = []
            for tr_idx, va_idx in inner_kf.split(X):
                pipe = self._build_pipeline(alg_name, algorithm)
                if params:
                    pipe.set_params(**params)
                pipe.fit(X[tr_idx], y[tr_idx])
                pred = pipe.predict(X[va_idx])
                oof[va_idx] = pred
                fold_scores.append(r2_score(y[va_idx], pred))

            if np.any(~np.isfinite(oof)):
                continue

            score = float(np.mean(fold_scores))
            if score > best["score"]:
                best = {"score": score, "params": params.copy(), "oof": oof}

        if best["oof"] is None:
            raise RuntimeError(f"Inner CV failed for {alg_name}")
        return best["params"], best["oof"], best["score"]

    def _run_stacking_ensemble(
        self,
        trait_name,
        algorithms,
        successful_algorithms,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ):
        """Nested-CV stacking: outer eval + inner tuning/stacking."""
        if len(successful_algorithms) < 2:
            self.logger.info("    Skipping stacking: need >=2 successful base models.")
            return None

        n_train = X_train.shape[0]
        outer_splits = max(2, min(self.stack_outer_splits, n_train))
        inner_splits = max(2, min(self.stack_inner_splits, n_train - 1))
        if outer_splits < 2 or inner_splits < 2:
            self.logger.info("    Skipping stacking: insufficient samples for nested CV.")
            return None

        self.logger.info(
            "    Nested stacking config: outer=%d, inner=%d, alpha=%.4f, fit_intercept=%s, standardize_cols=%s, normalize_weights=%s",
            outer_splits,
            inner_splits,
            self.stack_alpha,
            self.stack_fit_intercept,
            self.stack_standardize_cols,
            self.stack_normalize_weights,
        )

        model_names = sorted(successful_algorithms)
        outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=self.random_state)
        outer_cache = {}
        excluded = {}

        for fold_id, (outer_fit_idx, outer_test_idx) in enumerate(outer_kf.split(X_train)):
            X_outer_train = X_train[outer_fit_idx]
            y_outer_train = y_train[outer_fit_idx]
            X_outer_test = X_train[outer_test_idx]

            fold_inner_oof = {}
            fold_outer_test_pred = {}
            fold_best_params = {}
            for model_name in model_names:
                try:
                    best_params, inner_oof, _ = self._tune_model_and_get_oof(
                        model_name, algorithms[model_name], X_outer_train, y_outer_train, inner_splits
                    )
                    tuned_model = self._build_pipeline(model_name, algorithms[model_name])
                    if best_params:
                        tuned_model.set_params(**best_params)
                    tuned_model.fit(X_outer_train, y_outer_train)
                    fold_outer_test_pred[model_name] = tuned_model.predict(X_outer_test)
                    fold_inner_oof[model_name] = inner_oof
                    fold_best_params[model_name] = best_params
                except Exception as exc:
                    excluded[model_name] = f"outer_fold_{fold_id}: {exc}"

            outer_cache[fold_id] = {
                "outer_fit_idx": outer_fit_idx,
                "outer_test_idx": outer_test_idx,
                "inner_oof": fold_inner_oof,
                "outer_test_pred": fold_outer_test_pred,
                "best_params": fold_best_params,
            }

        for model_name, reason in excluded.items():
            self.logger.warning("    Excluding %s from stacking: %s", model_name, reason)

        complete_models = [
            m for m in model_names
            if m not in excluded
            and all((m in outer_cache[f]["inner_oof"] and m in outer_cache[f]["outer_test_pred"]) for f in outer_cache)
        ]

        if len(complete_models) < 2:
            self.logger.info("    Skipping stacking: <2 complete models after nested CV.")
            return None

        weights_by_fold = []
        weights_rows = []
        nested_oof_stack = np.full(n_train, np.nan, dtype=float)
        for fold_id, fold_data in outer_cache.items():
            outer_fit_idx = fold_data["outer_fit_idx"]
            outer_test_idx = fold_data["outer_test_idx"]
            p_inner = np.column_stack([fold_data["inner_oof"][m] for m in complete_models])
            p_outer_test = np.column_stack([fold_data["outer_test_pred"][m] for m in complete_models])
            fold_fit = fit_nonneg_ridge(
                p_inner,
                y_train[outer_fit_idx],
                alpha=self.stack_alpha,
                fit_intercept=self.stack_fit_intercept,
                standardize_cols=self.stack_standardize_cols,
                normalize_weights=self.stack_normalize_weights,
            )
            fold_weights = fold_fit["weights"]
            weights_by_fold.append(fold_weights)
            nested_oof_stack[outer_test_idx] = predict_stacker(p_outer_test, fold_weights, fold_fit["intercept"])
            for model_name, weight in zip(complete_models, fold_weights):
                weights_rows.append({"fold": fold_id, "model": model_name, "weight": float(weight)})

        # Final model tuning/training on full train (inner CV), then predict val/test.
        train_oof_cols = []
        train_cols, val_cols, test_cols = [], [], []
        full_best_params = {}
        for model_name in complete_models:
            best_params, train_oof, _ = self._tune_model_and_get_oof(
                model_name, algorithms[model_name], X_train, y_train, inner_splits
            )
            full_best_params[model_name] = best_params
            train_oof_cols.append(train_oof)
            final_model = self._build_pipeline(model_name, algorithms[model_name])
            if best_params:
                final_model.set_params(**best_params)
            final_model.fit(X_train, y_train)
            train_cols.append(final_model.predict(X_train))
            val_cols.append(final_model.predict(X_val))
            test_cols.append(final_model.predict(X_test))

        p_train_oof = np.column_stack(train_oof_cols)
        p_train_full = np.column_stack(train_cols)
        p_val_full = np.column_stack(val_cols)
        p_test_full = np.column_stack(test_cols)

        fit_result = fit_nonneg_ridge(
            p_train_oof,
            y_train,
            alpha=self.stack_alpha,
            fit_intercept=self.stack_fit_intercept,
            standardize_cols=self.stack_standardize_cols,
            normalize_weights=self.stack_normalize_weights,
        )
        final_weights = fit_result["weights"]
        intercept = fit_result["intercept"]

        yhat_train = predict_stacker(p_train_full, final_weights, intercept)
        yhat_val = predict_stacker(p_val_full, final_weights, intercept)
        yhat_test = predict_stacker(p_test_full, final_weights, intercept)

        stack_train_metrics = self.calculate_metrics(y_train, yhat_train, 'Ensemble_Stacking_NonNeg_Ridge')
        stack_val_metrics = self.calculate_metrics(y_val, yhat_val, 'Ensemble_Stacking_NonNeg_Ridge')
        stack_test_metrics = self.calculate_metrics(y_test, yhat_test, 'Ensemble_Stacking_NonNeg_Ridge')

        summary = summarize_weights(weights_by_fold, complete_models, MODEL_FAMILY_MAP)
        trait_dir = self.output_dir / f"stacking_{trait_name}"
        trait_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(weights_rows).to_csv(trait_dir / "stacking_weights_by_fold.csv", index=False)
        pd.DataFrame(summary["model_summary"]).to_csv(trait_dir / "stacking_weights_summary.csv", index=False)
        pd.DataFrame(summary["family_summary"]).to_csv(trait_dir / "stacking_family_weights_summary.csv", index=False)
        np.savez_compressed(
            trait_dir / "oof_predictions_stacking.npz",
            P_oof=p_train_oof,
            model_names=np.array(complete_models, dtype=object),
            y=np.asarray(y_train, dtype=float),
            yhat_stack=predict_stacker(p_train_oof, final_weights, intercept),
            nested_outer_oof=nested_oof_stack,
        )
        with open(trait_dir / "stacking_model_params.json", "w") as f:
            json.dump(full_best_params, f, indent=2)

        valid_outer = np.isfinite(nested_oof_stack)
        if np.sum(valid_outer) >= 3:
            nested_outer_rmse = float(np.sqrt(np.mean((y_train[valid_outer] - nested_oof_stack[valid_outer]) ** 2)))
            nested_outer_r2 = float(r2_score(y_train[valid_outer], nested_oof_stack[valid_outer]))
            nested_outer_corr = float(np.corrcoef(y_train[valid_outer], nested_oof_stack[valid_outer])[0, 1])
        else:
            nested_outer_rmse = float("nan")
            nested_outer_r2 = float("nan")
            nested_outer_corr = float("nan")

        self.logger.info(
            "    ✅ Stacking diagnostics: inner OOF RMSE=%.4f, R²=%.4f, corr=%.4f, outer OOF RMSE=%.4f, R²=%.4f, corr=%.4f, models=%d",
            fit_result["diagnostics"]["oof_rmse"],
            fit_result["diagnostics"]["oof_r2"],
            fit_result["diagnostics"]["oof_corr"],
            nested_outer_rmse,
            nested_outer_r2,
            nested_outer_corr,
            len(complete_models),
        )

        return {
            'trait': trait_name,
            'algorithm': 'Ensemble_Stacking_NonNeg_Ridge',
            'train_r2': stack_train_metrics['r2'],
            'train_pearson_r': stack_train_metrics['pearson_r'],
            'train_rmse': stack_train_metrics['rmse'],
            'train_mae': stack_train_metrics['mae'],
            'train_bias': stack_train_metrics['bias'],
            'val_r2': stack_val_metrics['r2'],
            'val_pearson_r': stack_val_metrics['pearson_r'],
            'val_rmse': stack_val_metrics['rmse'],
            'val_mae': stack_val_metrics['mae'],
            'val_bias': stack_val_metrics['bias'],
            'test_r2': stack_test_metrics['r2'],
            'test_pearson_r': stack_test_metrics['pearson_r'],
            'test_pearson_p': stack_test_metrics['pearson_p'],
            'test_spearman_r': stack_test_metrics['spearman_r'],
            'test_rmse': stack_test_metrics['rmse'],
            'test_mae': stack_test_metrics['mae'],
            'test_bias': stack_test_metrics['bias'],
            'cv_r2_mean': stack_val_metrics['r2'],
            'cv_r2_std': 0.0,
            'cv_pearson_mean': stack_val_metrics['pearson_r'],
            'cv_pearson_std': 0.0,
            'fit_time': 0.0,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'overfitting_indicator': stack_train_metrics['r2'] - stack_test_metrics['r2'],
        }
    
    def preprocess_data(self, X, y, skip_variance_filter=False):
        """Preprocess genomic and phenotypic data
        
        Args:
            X: Genotype matrix (already filtered for low variance if skip_variance_filter=True)
            y: Phenotype vector
            skip_variance_filter: If True, skip variance filtering (already done globally)
        """
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Only filter variance if not already done globally
        # (This allows per-trait filtering if needed, but typically we filter globally first)
        if not skip_variance_filter and X_clean.shape[1] > 1000:
            snp_var = np.var(X_clean, axis=0)
            var_threshold = 0.01
            high_var_mask = snp_var > var_threshold
            X_filtered = X_clean[:, high_var_mask]
            
            removed_snps = np.sum(~high_var_mask)
            if removed_snps > 0:
                self.logger.info(f"Removed {removed_snps} low variance SNPs (per-trait filtering)")
        else:
            X_filtered = X_clean
        
        return X_filtered, y_clean
    
    def calculate_metrics(self, y_true, y_pred, algorithm_name):
        """Calculate prediction metrics"""
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) < 3:
            return {'algorithm': algorithm_name, 'n_samples': len(y_true_clean),
                   'r2': np.nan, 'pearson_r': np.nan, 'pearson_p': np.nan,
                   'spearman_r': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan}
        
        r2 = r2_score(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        bias = np.mean(y_pred_clean - y_true_clean)
        
        pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
        spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
        
        return {
            'algorithm': algorithm_name,
            'n_samples': len(y_true_clean),
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'rmse': rmse,
            'mae': mae,
            'bias': bias
        }
    
    def process_trait_job(self, job_data, shared_data):
        """Process all algorithms for a single trait - NEW APPROACH"""
        
        trait_name = job_data['trait_name']
        algorithms = job_data['algorithms']
        # #region agent log
        _debug_log(
            "H5",
            "02a_phase1_train_validate_array.py:846",
            "Process trait job start",
            {"trait": trait_name, "algorithms": sorted(list(algorithms.keys()))},
        )
        # #endregion agent log
        
        X_train = shared_data['X_train']
        X_val = shared_data['X_val']
        X_test = shared_data['X_test']
        y_train_all = shared_data['y_train_all']
        y_val_all = shared_data['y_val_all']
        y_test_all = shared_data['y_test_all']
        trait_index = shared_data['trait_names'].index(trait_name)
        
        y_train = y_train_all[:, trait_index]
        y_val = y_val_all[:, trait_index]
        y_test = y_test_all[:, trait_index]
        
        self.logger.info(f"Processing all algorithms for trait: {trait_name}")
        
        trait_results = []
        test_predictions = {}  # Store predictions for ensemble creation
        failed_algorithms = []
        
        # Process each algorithm for this trait
        for alg_name, algorithm in algorithms.items():
            rss_before_mb = _get_rss_mb()
            self.logger.info(f"  Running {alg_name}... (RSS {rss_before_mb/1024:.2f} GB)")
            
            try:
                start_time = datetime.now()
                
                # Create pipeline
                pipeline = self._build_pipeline(alg_name, algorithm)
                
                # Train and predict
                pipeline.fit(X_train, y_train)
                y_pred_train = pipeline.predict(X_train)
                y_pred_val = pipeline.predict(X_val)
                y_pred_test = pipeline.predict(X_test)
                
                fit_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate metrics for all three sets
                train_metrics = self.calculate_metrics(y_train, y_pred_train, alg_name)
                val_metrics = self.calculate_metrics(y_val, y_pred_val, alg_name)
                test_metrics = self.calculate_metrics(y_test, y_pred_test, alg_name)
                
                result = {
                    'trait': trait_name,
                    'algorithm': alg_name,
                    # Train metrics
                    'train_r2': train_metrics['r2'],
                    'train_pearson_r': train_metrics['pearson_r'],
                    'train_rmse': train_metrics['rmse'],
                    'train_mae': train_metrics['mae'],
                    'train_bias': train_metrics['bias'],
                    # Validation metrics
                    'val_r2': val_metrics['r2'],
                    'val_pearson_r': val_metrics['pearson_r'],
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_bias': val_metrics['bias'],
                    # Test metrics
                    'test_r2': test_metrics['r2'],
                    'test_pearson_r': test_metrics['pearson_r'],
                    'test_pearson_p': test_metrics['pearson_p'],
                    'test_spearman_r': test_metrics['spearman_r'],
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'test_bias': test_metrics['bias'],
                    # Legacy fields for compatibility
                    'cv_r2_mean': val_metrics['r2'],
                    'cv_r2_std': 0.0,
                    'cv_pearson_mean': val_metrics['pearson_r'],
                    'cv_pearson_std': 0.0,
                    'fit_time': fit_time,
                    'n_train': len(X_train),
                    'n_val': len(X_val),
                    'n_test': len(X_test),
                    'overfitting_indicator': train_metrics['r2'] - test_metrics['r2']
                }
                
                trait_results.append(result)
                
                rss_after_mb = _get_rss_mb()
                self.logger.info(f"    ✅ {alg_name}: Train R²={train_metrics['r2']:.4f}, "
                               f"Val R²={val_metrics['r2']:.4f}, Test R²={test_metrics['r2']:.4f}, "
                               f"Test r={test_metrics['pearson_r']:.4f}, Time={fit_time:.2f}s, RSS={rss_after_mb/1024:.2f} GB")
                
                # Store predictions for ensemble creation
                test_predictions[alg_name] = y_pred_test
                
            except Exception as e:
                self.logger.error(f"    ❌ {alg_name} failed: {str(e)}")
                failed_algorithms.append({"algorithm": alg_name, "error": str(e)})
                # #region agent log
                _debug_log(
                    "H6",
                    "02a_phase1_train_validate_array.py:944",
                    "Algorithm failed",
                    {"trait": trait_name, "algorithm": alg_name, "error": str(e)},
                )
                # #endregion agent log
                continue

        # #region agent log
        _debug_log(
            "H5",
            "02a_phase1_train_validate_array.py:948",
            "Process trait job summary",
            {
                "trait": trait_name,
                "successful_algorithms": [r.get("algorithm") for r in trait_results],
                "failed_algorithms": failed_algorithms,
            },
        )
        # #endregion agent log
        
        # Create ensemble predictions only if ensembles were requested via --ensemble
        train_predictions = {}
        val_predictions = {}
        if self.ensemble_options and len(trait_results) > 1 and len(test_predictions) > 1:
            self.logger.info("  Creating ensemble predictions...")
            try:
                # Re-run predictions for train and validation sets for ensemble
                for alg_name, algorithm in algorithms.items():
                    if alg_name in test_predictions:  # Only for successful algorithms
                        try:
                            pipeline = self._build_pipeline(alg_name, algorithm)
                            
                            pipeline.fit(X_train, y_train)
                            train_predictions[alg_name] = pipeline.predict(X_train)
                            val_predictions[alg_name] = pipeline.predict(X_val)
                        except:
                            pass  # Skip if fails
                
                # Test set ensembles
                test_pred_matrix = np.column_stack(list(test_predictions.values()))
                simple_avg_test = np.mean(test_pred_matrix, axis=1)
                median_test = np.median(test_pred_matrix, axis=1)
                
                weights = []
                for alg_name in test_predictions.keys():
                    if 'Ridge' in alg_name or 'LASSO' in alg_name or 'Bayes' in alg_name:
                        weights.append(2.0)
                    elif 'RandomForest' in alg_name or 'XGBoost' in alg_name:
                        weights.append(1.5)
                    else:
                        weights.append(1.0)
                weights = np.array(weights) / np.sum(weights)
                weighted_avg_test = np.average(test_pred_matrix, axis=1, weights=weights)
                
                # Train set ensembles
                train_pred_matrix = np.column_stack(list(train_predictions.values()))
                simple_avg_train = np.mean(train_pred_matrix, axis=1)
                median_train = np.median(train_pred_matrix, axis=1)
                weighted_avg_train = np.average(train_pred_matrix, axis=1, weights=weights)
                
                # Validation set ensembles
                val_pred_matrix = np.column_stack(list(val_predictions.values()))
                simple_avg_val = np.mean(val_pred_matrix, axis=1)
                median_val = np.median(val_pred_matrix, axis=1)
                weighted_avg_val = np.average(val_pred_matrix, axis=1, weights=weights)
                
                # Calculate metrics for all sets
                simple_train_metrics = self.calculate_metrics(y_train, simple_avg_train, 'Ensemble_Simple_Average')
                simple_val_metrics = self.calculate_metrics(y_val, simple_avg_val, 'Ensemble_Simple_Average')
                simple_test_metrics = self.calculate_metrics(y_test, simple_avg_test, 'Ensemble_Simple_Average')
                
                median_train_metrics = self.calculate_metrics(y_train, median_train, 'Ensemble_Median')
                median_val_metrics = self.calculate_metrics(y_val, median_val, 'Ensemble_Median')
                median_test_metrics = self.calculate_metrics(y_test, median_test, 'Ensemble_Median')
                
                weighted_train_metrics = self.calculate_metrics(y_train, weighted_avg_train, 'Ensemble_Weighted_Average')
                weighted_val_metrics = self.calculate_metrics(y_val, weighted_avg_val, 'Ensemble_Weighted_Average')
                weighted_test_metrics = self.calculate_metrics(y_test, weighted_avg_test, 'Ensemble_Weighted_Average')
                
                # Add ensemble results with full train/val/test metrics
                ensemble_results = [
                    {
                        'trait': trait_name,
                        'algorithm': 'Ensemble_Simple_Average',
                        'train_r2': simple_train_metrics['r2'],
                        'train_pearson_r': simple_train_metrics['pearson_r'],
                        'train_rmse': simple_train_metrics['rmse'],
                        'train_mae': simple_train_metrics['mae'],
                        'train_bias': simple_train_metrics['bias'],
                        'val_r2': simple_val_metrics['r2'],
                        'val_pearson_r': simple_val_metrics['pearson_r'],
                        'val_rmse': simple_val_metrics['rmse'],
                        'val_mae': simple_val_metrics['mae'],
                        'val_bias': simple_val_metrics['bias'],
                        'test_r2': simple_test_metrics['r2'],
                        'test_pearson_r': simple_test_metrics['pearson_r'],
                        'test_pearson_p': simple_test_metrics['pearson_p'],
                        'test_spearman_r': simple_test_metrics['spearman_r'],
                        'test_rmse': simple_test_metrics['rmse'],
                        'test_mae': simple_test_metrics['mae'],
                        'test_bias': simple_test_metrics['bias'],
                        'cv_r2_mean': simple_val_metrics['r2'],
                        'cv_r2_std': 0.0,
                        'cv_pearson_mean': simple_val_metrics['pearson_r'],
                        'cv_pearson_std': 0.0,
                        'fit_time': 0.0,
                        'n_train': len(X_train),
                        'n_val': len(X_val),
                        'n_test': len(X_test),
                        'overfitting_indicator': simple_train_metrics['r2'] - simple_test_metrics['r2']
                    },
                    {
                        'trait': trait_name,
                        'algorithm': 'Ensemble_Median',
                        'train_r2': median_train_metrics['r2'],
                        'train_pearson_r': median_train_metrics['pearson_r'],
                        'train_rmse': median_train_metrics['rmse'],
                        'train_mae': median_train_metrics['mae'],
                        'train_bias': median_train_metrics['bias'],
                        'val_r2': median_val_metrics['r2'],
                        'val_pearson_r': median_val_metrics['pearson_r'],
                        'val_rmse': median_val_metrics['rmse'],
                        'val_mae': median_val_metrics['mae'],
                        'val_bias': median_val_metrics['bias'],
                        'test_r2': median_test_metrics['r2'],
                        'test_pearson_r': median_test_metrics['pearson_r'],
                        'test_pearson_p': median_test_metrics['pearson_p'],
                        'test_spearman_r': median_test_metrics['spearman_r'],
                        'test_rmse': median_test_metrics['rmse'],
                        'test_mae': median_test_metrics['mae'],
                        'test_bias': median_test_metrics['bias'],
                        'cv_r2_mean': median_val_metrics['r2'],
                        'cv_r2_std': 0.0,
                        'cv_pearson_mean': median_val_metrics['pearson_r'],
                        'cv_pearson_std': 0.0,
                        'fit_time': 0.0,
                        'n_train': len(X_train),
                        'n_val': len(X_val),
                        'n_test': len(X_test),
                        'overfitting_indicator': median_train_metrics['r2'] - median_test_metrics['r2']
                    },
                    {
                        'trait': trait_name,
                        'algorithm': 'Ensemble_Weighted_Average',
                        'train_r2': weighted_train_metrics['r2'],
                        'train_pearson_r': weighted_train_metrics['pearson_r'],
                        'train_rmse': weighted_train_metrics['rmse'],
                        'train_mae': weighted_train_metrics['mae'],
                        'train_bias': weighted_train_metrics['bias'],
                        'val_r2': weighted_val_metrics['r2'],
                        'val_pearson_r': weighted_val_metrics['pearson_r'],
                        'val_rmse': weighted_val_metrics['rmse'],
                        'val_mae': weighted_val_metrics['mae'],
                        'val_bias': weighted_val_metrics['bias'],
                        'test_r2': weighted_test_metrics['r2'],
                        'test_pearson_r': weighted_test_metrics['pearson_r'],
                        'test_pearson_p': weighted_test_metrics['pearson_p'],
                        'test_spearman_r': weighted_test_metrics['spearman_r'],
                        'test_rmse': weighted_test_metrics['rmse'],
                        'test_mae': weighted_test_metrics['mae'],
                        'test_bias': weighted_test_metrics['bias'],
                        'cv_r2_mean': weighted_val_metrics['r2'],
                        'cv_r2_std': 0.0,
                        'cv_pearson_mean': weighted_val_metrics['pearson_r'],
                        'cv_pearson_std': 0.0,
                        'fit_time': 0.0,
                        'n_train': len(X_train),
                        'n_val': len(X_val),
                        'n_test': len(X_test),
                        'overfitting_indicator': weighted_train_metrics['r2'] - weighted_test_metrics['r2']
                    }
                ]
                
                trait_results.extend(ensemble_results)
                
                self.logger.info(f"    ✅ Ensembles: Simple R²={simple_test_metrics['r2']:.4f}, "
                               f"Median R²={median_test_metrics['r2']:.4f}, "
                               f"Weighted R²={weighted_test_metrics['r2']:.4f}")

                if 'stacking_nonneg_ridge' in self.ensemble_options:
                    stacking_result = self._run_stacking_ensemble(
                        trait_name=trait_name,
                        algorithms=algorithms,
                        successful_algorithms=list(test_predictions.keys()),
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                    )
                    if stacking_result is not None:
                        trait_results.append(stacking_result)
                        self.logger.info(
                            "    ✅ Stacking ensemble: Test R²=%.4f, Test r=%.4f",
                            stacking_result['test_r2'],
                            stacking_result['test_pearson_r'],
                        )
                
            except Exception as e:
                self.logger.error(f"    ❌ Ensemble creation failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.logger.info(f"Completed {len(trait_results)} algorithms (including {len([r for r in trait_results if 'Ensemble' in r['algorithm']])} ensembles) for {trait_name}")
        return trait_results
    
    def calculate_gmatrix_if_needed(self, X, output_dir=None, method='vanRaden'):
        """Calculate Gmatrix if the module is available"""
        if not GMATRIX_AVAILABLE:
            self.logger.warning("Gmatrix calculation not available - skipping")
            return None, None
        
        try:
            self.logger.info("Calculating Gmatrix...")
            G, info = calculate_gmatrix(
                X, 
                method=method, 
                standardize=True,
                save_intermediate=True,
                output_dir=output_dir
            )
            self.logger.info(f"✅ Gmatrix calculated: shape {G.shape}")
            self.logger.info(f"   Diagonal range: [{np.min(np.diag(G)):.4f}, {np.max(np.diag(G)):.4f}]")
            
            # Print summary report
            print("\n" + "="*60)
            print("G-MATRIX CALCULATION SUMMARY (Train-Validate Phase)")
            print("="*60)
            print(f"Method: {info['method']}")
            print(f"Animals: {info['n_animals']}")
            print(f"Markers: {info['n_markers']}")
            print(f"Computation time: {info['computation_time']:.2f} seconds")
            print(f"Standardized: {info['standardized']}")
            if info.get('scaling_factor'):
                print(f"Scaling factor: {info['scaling_factor']:.6f}")
            print(f"G-matrix shape: {G.shape}")
            print(f"Diagonal range: [{np.min(np.diag(G)):.4f}, {np.max(np.diag(G)):.4f}]")
            print(f"G-matrix saved to: {output_dir}/Gmatrix.csv")
            print("="*60 + "\n")
            
            return G, info
        except Exception as e:
            self.logger.error(f"Failed to calculate Gmatrix: {str(e)}")
            return None, None
    
    def prepare_array_jobs(self, X_file, y_file, traits=None, calculate_gmatrix=False):
        """Prepare data and job configurations for array processing"""
        
        self.logger.info("Preparing array jobs (one job per trait)...")
        
        # Load data
        X_df = pd.read_csv(X_file, index_col=0)
        X = X_df.values
        y_df = pd.read_csv(y_file, index_col=0)
        
        if traits:
            trait_names = traits
            y = y_df[trait_names].values
        else:
            trait_names = list(y_df.columns)
            y = y_df.values
        
        self.logger.info(f"Data: X={X.shape}, y={y.shape}, traits={len(trait_names)}")
        
        # IMPORTANT: Filter low variance SNPs FIRST, before calculating G-matrix
        # This ensures G-matrix uses the same SNPs as prediction models
        self.logger.info("Filtering low variance SNPs before G-matrix calculation...")
        if X.shape[1] > 1000:
            # Calculate variance across all samples (before train/test split)
            # Use a representative sample - take first trait's non-NaN samples
            first_trait_y = y[:, 0] if y.ndim > 1 else y
            mask = ~np.isnan(first_trait_y)
            X_sample = X[mask]
            
            snp_var = np.var(X_sample, axis=0)
            var_threshold = 0.01
            high_var_mask = snp_var > var_threshold
            X_filtered = X[:, high_var_mask]
            
            # Update X_df to match filtered SNPs
            X_df_filtered = X_df.iloc[:, high_var_mask]
            
            removed_snps = np.sum(~high_var_mask)
            if removed_snps > 0:
                self.logger.info(f"Removed {removed_snps} low variance SNPs (before G-matrix calculation)")
                self.logger.info(f"Filtered data: X={X_filtered.shape} (from {X.shape})")
        else:
            X_filtered = X
            X_df_filtered = X_df
            high_var_mask = np.ones(X.shape[1], dtype=bool)
            self.logger.info("No low variance filtering (SNPs < 1000)")
        
        # VALIDATION: Check if a previous mask exists and verify consistency
        variance_mask_file = self.output_dir / 'variance_filter_mask.npy'
        if variance_mask_file.exists():
            previous_mask = np.load(variance_mask_file)
            if previous_mask.shape[0] != high_var_mask.shape[0] or not np.array_equal(previous_mask, high_var_mask):
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ FEATURE MISMATCH ERROR\n"
                    f"{'='*80}\n"
                    f"A previous variance filter mask exists but does not match the current input data.\n"
                    f"\n"
                    f"Previous mask: {previous_mask.shape[0]} features, kept {np.sum(previous_mask)} features\n"
                    f"Current mask:  {high_var_mask.shape[0]} features, kept {np.sum(high_var_mask)} features\n"
                    f"\n"
                    f"This indicates the input dataset has changed since the last training run.\n"
                    f"All phases must use the SAME input dataset with the SAME features.\n"
                    f"\n"
                    f"Previous mask file: {variance_mask_file}\n"
                    f"\n"
                    f"Please ensure:\n"
                    f"  1. The same Geno.csv file is used for all training runs\n"
                    f"  2. The file has not been modified between runs\n"
                    f"  3. If you want to use a different dataset, delete the previous mask file first\n"
                    f"\n"
                    f"Execution stopped to prevent inconsistent results.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                print(error_msg)
                raise ValueError(f"Feature mismatch: previous mask has {previous_mask.shape[0]} features, but current input has {high_var_mask.shape[0]} features")
            else:
                self.logger.info(f"✅ Feature validation passed: current input matches previous mask ({high_var_mask.shape[0]} features)")
        
        # Save variance filter mask for reuse in prediction
        np.save(variance_mask_file, high_var_mask)
        self.logger.info(f"Saved variance filter mask to: {variance_mask_file}")
        self.logger.info(f"  Mask shape: {high_var_mask.shape}, kept features: {np.sum(high_var_mask)}/{len(high_var_mask)}")
        
        # Calculate Gmatrix on FILTERED SNPs (if requested)
        G = None
        G_info = None
        if calculate_gmatrix:
            gmatrix_dir = self.qc_dir / 'gmatrix'
            self.logger.info(f"Calculating G-matrix on filtered SNPs: {X_df_filtered.shape[1]} markers")
            G, G_info = self.calculate_gmatrix_if_needed(X_df_filtered, output_dir=gmatrix_dir)
        
        # Preprocess and create train/validate/test split for each trait
        # Split: 60% train, 20% validate, 20% test
        # Note: Skip variance filtering in preprocess_data since we already did it globally
        processed_data = {}
        for i, trait_name in enumerate(trait_names):
            # Use filtered X for preprocessing (skip variance filter since already done)
            X_proc, y_proc = self.preprocess_data(X_filtered, y[:, i], skip_variance_filter=True)
            
            # First split: separate test set (20%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_proc, y_proc, test_size=self.test_size, random_state=self.random_state
            )
            
            # Second split: separate train (60%) and validate (20%) from remaining 80%
            # val_size is relative to the remaining data after test split
            # To get 20% of total: val_size = 0.2 / (1 - test_size) = 0.2 / 0.8 = 0.25
            val_size_relative = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_relative, random_state=self.random_state
            )
            
            processed_data[trait_name] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            }
        
        # Create shared data structure
        first_trait = trait_names[0]
        X_train_shared = processed_data[first_trait]['X_train']
        X_val_shared = processed_data[first_trait]['X_val']
        X_test_shared = processed_data[first_trait]['X_test']
        
        y_train_all = np.column_stack([processed_data[trait]['y_train'] for trait in trait_names])
        y_val_all = np.column_stack([processed_data[trait]['y_val'] for trait in trait_names])
        y_test_all = np.column_stack([processed_data[trait]['y_test'] for trait in trait_names])
        
        shared_data = {
            'X_train': X_train_shared,
            'X_val': X_val_shared,
            'X_test': X_test_shared,
            'y_train_all': y_train_all,
            'y_val_all': y_val_all,
            'y_test_all': y_test_all,
            'trait_names': trait_names
        }
        
        # Add Gmatrix if calculated
        if G is not None:
            shared_data['Gmatrix'] = G
            shared_data['Gmatrix_info'] = G_info
            self.logger.info("Gmatrix included in shared data")
        
        # Create job combinations - ONE JOB PER TRAIT
        algorithms = self.get_algorithms()
        if hasattr(self, '_algorithm_filter') and self._algorithm_filter:
            keep = {a for a in self._algorithm_filter if a in algorithms}
            if keep:
                algorithms = {k: v for k, v in algorithms.items() if k in keep}
                self.logger.info("Filtered to %d algorithm(s): %s", len(algorithms), sorted(algorithms))
        array_manager = ArrayJobManager(self.output_dir)
        jobs = array_manager.create_trait_jobs(trait_names, algorithms)
        
        # Save shared data
        shared_data_file = array_manager.save_job_data(shared_data, 'shared', 'shared_data')
        
        # Save individual job data
        for job in jobs:
            array_manager.save_job_data(job, job['job_id'], 'job')
        
        self.logger.info(f"Prepared {len(jobs)} jobs for array processing (one per trait)")
        return len(jobs), array_manager

def main():
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        v = str(v).strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description='Array Job Genomic Prediction Train-Validate (Algorithm Selection)')
    parser.add_argument('--mode', required=True, choices=['prepare', 'process_job', 'combine_results'])
    parser.add_argument('--X_file', help='Path to genotype data (CSV)')
    parser.add_argument('--y_file', help='Path to phenotype data (CSV)')
    parser.add_argument('--output_dir', default='./Phase1_Learning_Benchmarking/training_validation')
    parser.add_argument('--qc_dir', default='./Phase1_Learning_Benchmarking/QC',
                       help='Directory for QC-related files (G-matrix, etc.)')
    parser.add_argument('--job_id', type=int, help='Job ID for array processing')
    parser.add_argument('--n_jobs', type=int, help='Total number of jobs')
    parser.add_argument('--traits', nargs='*', help='Specific traits to analyze')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size relative to remaining data after test split (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--calculate_gmatrix', action='store_true',
                       help='Calculate Gmatrix (genomic relationship matrix) during preparation')
    parser.add_argument(
        '--algorithms',
        nargs='*',
        default=None,
        help='Restrict to specific algorithms (e.g. GBLUP_Ridge GBLUP_RidgeCV). If omitted, all available algorithms run.',
    )
    parser.add_argument(
        '--ensemble',
        nargs='*',
        default=[],
        help='Optional ensembles to add. Use stacking_nonneg_ridge to enable non-negative ridge stacking.',
    )
    parser.add_argument('--stack_alpha', type=float, default=0.01, help='Ridge penalty for stacking_nonneg_ridge.')
    parser.add_argument(
        '--stack_fit_intercept',
        type=_str2bool,
        default=True,
        help='Whether stacking_nonneg_ridge fits an intercept (true/false).',
    )
    parser.add_argument(
        '--stack_standardize_cols',
        type=_str2bool,
        default=True,
        help='Whether stacking_nonneg_ridge standardizes base prediction columns (true/false).',
    )
    parser.add_argument(
        '--stack_normalize_weights',
        type=_str2bool,
        default=False,
        help='Whether to normalize stacking weights to sum to 1 (true/false).',
    )
    parser.add_argument(
        '--stack_n_splits',
        type=int,
        default=5,
        help='Deprecated alias for outer folds in nested stacking CV.',
    )
    parser.add_argument(
        '--stack_outer_splits',
        type=int,
        default=5,
        help='Outer K folds for nested stacking evaluation.',
    )
    parser.add_argument(
        '--stack_inner_splits',
        type=int,
        default=3,
        help='Inner J folds for tuning + OOF stack training.',
    )
    
    args = parser.parse_args()
    
    benchmarker = GenomicBenchmarkingArray(
        output_dir=args.output_dir,
        qc_dir=args.qc_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=getattr(args, 'val_size', 0.2),
        ensemble_options=args.ensemble,
        stack_alpha=args.stack_alpha,
        stack_fit_intercept=args.stack_fit_intercept,
        stack_standardize_cols=args.stack_standardize_cols,
        stack_normalize_weights=args.stack_normalize_weights,
        stack_n_splits=args.stack_n_splits,
        stack_outer_splits=args.stack_outer_splits,
        stack_inner_splits=args.stack_inner_splits,
    )
    benchmarker._algorithm_filter = args.algorithms
    
    array_manager = ArrayJobManager(args.output_dir)
    
    if args.mode == 'prepare':
        if not args.X_file or not args.y_file:
            print("Error: X_file and y_file required for prepare mode")
            return 1
        
        n_jobs, array_manager = benchmarker.prepare_array_jobs(
            args.X_file, args.y_file, args.traits,
            calculate_gmatrix=args.calculate_gmatrix
        )
        
        extra_cli_args = ""
        if args.ensemble:
            ensemble_flags = " ".join(args.ensemble)
            extra_cli_args += f" --ensemble {ensemble_flags}"
        extra_cli_args += (
            f" --stack_alpha {args.stack_alpha}"
            f" --stack_fit_intercept {str(args.stack_fit_intercept).lower()}"
            f" --stack_standardize_cols {str(args.stack_standardize_cols).lower()}"
            f" --stack_normalize_weights {str(args.stack_normalize_weights).lower()}"
            f" --stack_n_splits {args.stack_n_splits}"
            f" --stack_outer_splits {args.stack_outer_splits}"
            f" --stack_inner_splits {args.stack_inner_splits}"
        )

        array_script = array_manager.generate_slurm_array_script(n_jobs, extra_cli_args=extra_cli_args)
        combine_script = array_manager.generate_combine_script(n_jobs)
        
        print(f"Array jobs prepared: {n_jobs} jobs (one per trait)")
        print(f"SLURM array script: {array_script}")
        print(f"Combine script: {combine_script}")
        
    elif args.mode == 'process_job':
        if args.job_id is None:
            print("Error: job_id required for process_job mode")
            return 1
        
        try:
            job_data = array_manager.load_job_data(args.job_id, 'job')
            shared_data = array_manager.load_job_data('shared', 'shared_data')
            
            results = benchmarker.process_trait_job(job_data, shared_data)  # Changed function name
            array_manager.save_job_results(results, args.job_id)
            
            print(f"Job {args.job_id} completed successfully")
            
        except Exception as e:
            print(f"Job {args.job_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.mode == 'combine_results':
        if not args.n_jobs:
            print("Error: n_jobs required for combine_results mode")
            return 1
        
        combined_results = array_manager.combine_all_results(args.n_jobs)
        
        if combined_results:
            results_df = pd.DataFrame(combined_results)
            results_file = Path(args.output_dir) / "combined_train_validate_results.csv"
            results_df.to_csv(results_file, index=False)
            
            # Separate individual algorithms and ensembles
            individual_df = results_df[~results_df['algorithm'].str.contains('Ensemble', na=False)]
            ensemble_df = results_df[results_df['algorithm'].str.contains('Ensemble', na=False)]
            
            # Generate summary for individual algorithms
            trait_summary = individual_df.groupby('trait').agg({
                'test_pearson_r': ['mean', 'max'],
                'test_r2': ['mean', 'max'],
                'algorithm': 'count'
            }).round(4)
            
            # Generate ensemble summary
            if not ensemble_df.empty:
                ensemble_summary = ensemble_df.groupby(['trait', 'algorithm']).agg({
                    'test_r2': 'mean',
                    'test_pearson_r': 'mean',
                    'test_rmse': 'mean'
                }).round(4)
                
                print("\n" + "="*60)
                print("ENSEMBLE RESULTS SUMMARY")
                print("="*60)
                for trait in ensemble_df['trait'].unique():
                    trait_ensembles = ensemble_df[ensemble_df['trait'] == trait]
                    print(f"\n{trait}:")
                    for _, row in trait_ensembles.iterrows():
                        print(f"  {row['algorithm']:<30} R²={row['test_r2']:.4f}, "
                              f"Pearson r={row['test_pearson_r']:.4f}, RMSE={row['test_rmse']:.4f}")
                print("="*60 + "\n")
            
            summary_file = Path(args.output_dir) / "trait_summary.csv"
            trait_summary.to_csv(summary_file)
            
            print(f"Combined results saved: {results_file}")
            print(f"Trait summary saved: {summary_file}")
            print(f"Total results: {len(results_df)} (including {len(ensemble_df) if not ensemble_df.empty else 0} ensembles)")
            print(f"Mean Pearson r: {results_df['test_pearson_r'].mean():.4f}")
            print(f"Best result: {results_df['test_pearson_r'].max():.4f}")
            
            # Generate comprehensive report notebook
            try:
                # Add scripts directory to path for import
                scripts_dir = Path(__file__).parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                report_module = _load_module_from_file("02b_phase1_report_benchmarking.py", "phase1_report_benchmarking")
                create_report_notebook = report_module.create_report_notebook
                
                # Generate report in notebooks directory (same location as bash script)
                project_root = Path(args.output_dir).parent.parent
                notebooks_dir = project_root / 'notebooks' / 'Phase_1_Learning_Benchmarking'
                notebooks_dir.mkdir(parents=True, exist_ok=True)
                
                # Use same filename as bash script for consistency
                report_file = notebooks_dir / '1.2_Learning_Benchmarking_report.ipynb'
                create_report_notebook(str(results_file), str(report_file))
                print(f"✅ Comprehensive report notebook created: {report_file}")
            except Exception as e:
                print(f"⚠️  Could not generate report notebook: {e}")
                import traceback
                traceback.print_exc()
                print("   You can generate it manually using:")
                project_root = Path(args.output_dir).parent.parent
                notebooks_dir = project_root / 'notebooks' / 'Phase_1_Learning_Benchmarking'
                print(f"   python {scripts_dir}/02b_phase1_report_benchmarking.py --results_file {results_file} --output_file {notebooks_dir / '1.2_Learning_Benchmarking_report.ipynb'}")
            
            # Generate Phase 1 EDA/QC report (Geno QC, Pheno QC, G-matrix)
            try:
                print("\n" + "="*70)
                print("GENERATING PHASE 1 EDA/QC REPORT")
                print("="*70)
                
                # Determine paths
                project_root = Path(args.output_dir).parent.parent
                notebooks_dir = project_root / 'notebooks' / 'Phase_1_Learning_Benchmarking'
                notebooks_dir.mkdir(parents=True, exist_ok=True)
                
                # Find dataset directory
                dataset_dir = project_root / 'dataset' / 'input'
                if not dataset_dir.exists():
                    dataset_dir = project_root / 'data'
                
                # G-matrix directory
                gmatrix_dir = Path(args.qc_dir) / 'gmatrix'
                
                # Output file
                eda_qc_report_file = notebooks_dir / '1.1_Preprocessing_report.ipynb'
                
                print(f"📁 Dataset directory: {dataset_dir}")
                print(f"📁 G-matrix directory: {gmatrix_dir}")
                print(f"📁 Output file: {eda_qc_report_file}")
                
                # Import and call QC report generator
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                qc_module = _load_module_from_file("02c_phase1_report_preprocessing.py", "phase1_report_preprocessing")
                create_qc_report_notebook = qc_module.create_qc_report_notebook
                
                create_qc_report_notebook(
                    str(dataset_dir),
                    str(gmatrix_dir),
                    str(eda_qc_report_file)
                )
                print(f"✅ Phase 1 EDA/QC report generated: {eda_qc_report_file}")
                print("   Report includes: Geno QC, Pheno QC, G-matrix analysis")
            except Exception as e:
                print(f"⚠️  Could not generate Phase 1 EDA/QC report: {e}")
                import traceback
                traceback.print_exc()
                print("   You can generate it manually using:")
                print(f"   python {scripts_dir}/02c_phase1_report_preprocessing.py --dataset_dir {dataset_dir} --gmatrix_dir {gmatrix_dir} --output_file {eda_qc_report_file}")
            
            # Detailed ensemble reporting (similar to prediction phase)
            if not ensemble_df.empty:
                print("\n" + "="*70)
                print("ENSEMBLE PERFORMANCE SUMMARY (Train-Validate Phase)")
                print("="*70)
                for trait in sorted(ensemble_df['trait'].unique()):
                    trait_ensembles = ensemble_df[ensemble_df['trait'] == trait].sort_values('test_r2', ascending=False)
                    print(f"\n{trait}:")
                    print(f"  {'Ensemble Method':<30} {'Test R²':<12} {'Pearson r':<12} {'RMSE':<12}")
                    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
                    for _, row in trait_ensembles.iterrows():
                        print(f"  {row['algorithm']:<30} {row['test_r2']:>11.4f} {row['test_pearson_r']:>11.4f} {row['test_rmse']:>11.4f}")
                
                # Find best ensemble per trait
                print(f"\n{'='*70}")
                print("BEST ENSEMBLE PER TRAIT:")
                print(f"{'='*70}")
                for trait in sorted(ensemble_df['trait'].unique()):
                    trait_ensembles = ensemble_df[ensemble_df['trait'] == trait]
                    best = trait_ensembles.loc[trait_ensembles['test_r2'].idxmax()]
                    print(f"  {trait}:")
                    print(f"    Best: {best['algorithm']}")
                    print(f"    Test R²: {best['test_r2']:.4f}")
                    print(f"    Pearson r: {best['test_pearson_r']:.4f}")
                    print(f"    RMSE: {best['test_rmse']:.4f}")
                print("="*70 + "\n")
        else:
            print("No results found to combine")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
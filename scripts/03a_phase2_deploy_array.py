#!/usr/bin/env python3
"""
File: 03a_phase2_deploy_array.py
Purpose: Array job deployment system - ONE JOB PER TRAIT
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import argparse
import logging
import subprocess
import tempfile
import shutil
import importlib.util
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

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
    logging.warning("Gmatrix calculation module not available")

def _get_rscript_bin():
    """Return Rscript path from env or PATH."""
    env_path = os.environ.get("BGLR_RSCRIPT", "").strip()
    if env_path:
        return env_path
    return shutil.which("Rscript") or ""

def _check_bglr_available(rscript_bin):
    """Check whether Rscript can load the BGLR package.

    Bounded by a timeout so a slow or hanging R install can never block the
    pipeline (BGLR is only needed for the Bayesian methods). Set
    BREEDAI_SKIP_BGLR_CHECK=1 to skip the probe, or BREEDAI_BGLR_CHECK_TIMEOUT
    to change the timeout in seconds (default 60). Returns False on timeout.
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

class BGLRRegressor:
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
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"BGLR backend failed (method={self.method}): {result.stderr.strip()}"
                )
            
            self.coef_ = np.loadtxt(b_path, delimiter=",")
            mu_val = np.loadtxt(mu_path, delimiter=",")
            self.intercept_ = float(mu_val) if np.ndim(mu_val) == 0 else float(mu_val[0])
        
        return self
    
    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

try:
    import tensorflow as tf
    import gpflow
    GPFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    GPFLOW_AVAILABLE = False

class GPflowRegressor:
    """GPflow wrapper for deployment training."""
    
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
    
    def _create_kernel(self):
        if self.kernel_type == 'rbf':
            return gpflow.kernels.RBF(lengthscales=self.lengthscale, variance=self.variance)
        if self.kernel_type == 'matern52':
            return gpflow.kernels.Matern52(lengthscales=self.lengthscale, variance=self.variance)
        if self.kernel_type == 'linear':
            return gpflow.kernels.Linear(variance=self.variance)
        return gpflow.kernels.RBF(lengthscales=self.lengthscale, variance=self.variance)
    
    def fit(self, X, y):
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow not available")
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_tensor = tf.constant(X_scaled, dtype=tf.float64)
        y_tensor = tf.constant(y_scaled.reshape(-1, 1), dtype=tf.float64)
        
        kernel = self._create_kernel()
        
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
        except Exception:
            pass
        
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler_X.transform(X)
        X_tensor = tf.constant(X_scaled, dtype=tf.float64)
        mean, _ = self.model.predict_f(X_tensor)
        mean_np = mean.numpy().flatten()
        return self.scaler_y.inverse_transform(mean_np.reshape(-1, 1)).flatten()

# Import algorithms
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, LassoCV, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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

class TrainingArrayManager:
    """Manages training array jobs - ONE JOB PER TRAIT"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'deployment'
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'deployment_array.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_trait_training_jobs(self, trait_names, algorithms):
        """Create training jobs - ONE JOB PER TRAIT (all algorithms per trait)"""
        jobs = []
        
        for job_id, trait_name in enumerate(trait_names):
            jobs.append({
                'job_id': job_id,
                'trait_name': trait_name,
                'algorithms': algorithms  # All algorithms for this trait
            })
        
        self.logger.info(f"Created {len(jobs)} training jobs (one per trait)")
        return jobs
    
    def save_job_data(self, data, job_id, data_type='job'):
        """Save job data - FIXED"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return str(filepath)
    
    def load_job_data(self, job_id, data_type='job'):
        """Load job data - FIXED"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Job data not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _save_bglr_custom(self, pipeline, trait_dir, alg_name):
        """Save BGLR pipeline as coef_ + intercept_ (joblib often fails for R-backed models)."""
        regressor = pipeline.named_steps.get('regressor') or pipeline
        if getattr(regressor, 'coef_', None) is None or getattr(regressor, 'intercept_', None) is None:
            return False
        npz_file = trait_dir / f"{alg_name}.npz"
        meta_file = trait_dir / f"{alg_name}_custom_meta.json"
        np.savez(npz_file, coef=regressor.coef_, intercept=np.array(regressor.intercept_))
        with open(meta_file, 'w') as f:
            json.dump({"type": "bglr", "method": getattr(regressor, 'method', 'BayesA')}, f)
        return True

    def _save_gpflow_checkpoint(self, pipeline, trait_dir, alg_name):
        """Save GPflow model using TensorFlow checkpoint (see GPflow saving_and_loading docs)."""
        if not GPFLOW_AVAILABLE:
            return False
        regressor = pipeline.named_steps.get('regressor') if hasattr(pipeline, 'named_steps') else pipeline
        if getattr(regressor, 'model', None) is None:
            return False
        try:
            import tensorflow as tf
            ckpt_dir = trait_dir / f"{alg_name}_ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = tf.train.Checkpoint(model=regressor.model)
            manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=1)
            manager.save()
            joblib.dump(
                {'scaler_X': regressor.scaler_X, 'scaler_y': regressor.scaler_y},
                trait_dir / f"{alg_name}_scalers.joblib"
            )
            n_features = getattr(regressor.scaler_X, 'n_features_in_', None) or (regressor.scaler_X.mean_.shape[0] if hasattr(regressor.scaler_X, 'mean_') and regressor.scaler_X.mean_ is not None else None)
            meta = {
                'type': 'gpflow_ckpt',
                'kernel_type': getattr(regressor, 'kernel_type', 'rbf'),
                'use_sparse': getattr(regressor, 'use_sparse', True),
                'n_inducing': getattr(regressor, 'n_inducing', 50),
                'noise_variance': getattr(regressor, 'noise_variance', 0.1),
                'n_features': n_features,
            }
            with open(trait_dir / f"{alg_name}_meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
            return True
        except Exception as e:
            self.logger.warning(f"GPflow checkpoint save failed for {alg_name}: {e}")
            return False

    def save_trained_models_for_trait(self, trait_results, job_id, trait_name):
        """Save all trained models and preprocessing info for a trait (across all algorithms)"""
        saved_files = []
        
        # Create trait directory (use absolute path so SLURM jobs write to shared filesystem)
        trait_dir = (self.output_dir.resolve() / 'models' / trait_name.replace(' ', '_').replace('/', '_'))
        trait_dir.mkdir(parents=True, exist_ok=True)
        
        for result in trait_results:
            if not result['success']:
                continue
                
            alg_name = result['algorithm']
            model = result['trained_model']
            performance = result['performance']
            
            # Save performance first (always)
            perf_file = trait_dir / f"{alg_name}_performance.json"
            performance_serializable = performance.copy()
            with open(perf_file, 'w') as f:
                json.dump(performance_serializable, f, indent=2)
            
            # Save model: try joblib first; for BGLR always also save .npz so prediction can load (joblib.load often fails for R-backed objects on prediction node)
            model_file = trait_dir / f"{alg_name}.joblib"
            saved = False
            try:
                joblib.dump(model, model_file)
                saved = True
            except Exception as e:
                self.logger.warning(f"joblib.dump failed for {alg_name} ({trait_name}): {e}")
                if alg_name in ('BayesA', 'BayesB', 'BayesCpi'):
                    if self._save_bglr_custom(model, trait_dir, alg_name):
                        saved = True
                        self.logger.info(f"Saved model {alg_name} for {trait_name} (custom BGLR format)")
                if not saved:
                    self.logger.error(f"Could not save model {alg_name} for {trait_name}; prediction will skip it")
            # For BGLR, always save .npz as well so prediction can load without joblib (avoids R deserialization failure)
            if saved and alg_name in ('BayesA', 'BayesB', 'BayesCpi') and self._save_bglr_custom(model, trait_dir, alg_name):
                self.logger.info(f"Saved {alg_name} .npz for {trait_name} (for prediction load)")
            # For GPflow, save using TensorFlow checkpoint (joblib often fails at prediction; see GPflow saving_and_loading docs)
            if alg_name in ('GP_RBF', 'GP_Matern52') and self._save_gpflow_checkpoint(model, trait_dir, alg_name):
                saved = True
                self.logger.info(f"Saved {alg_name} for {trait_name} (GPflow checkpoint)")
            
            if saved:
                saved_files.append({
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'model_file': str(model_file),
                    'performance_file': str(perf_file)
                })
                self.logger.info(f"Saved model {alg_name} for {trait_name}")
        
        return saved_files
    
    def generate_slurm_training_array_script(self, n_jobs, job_name='genomic_deployment', project_dir=None):
        """Generate SLURM training array script"""
        
        # Determine project directory from output_dir if not provided
        if project_dir is None:
            # output_dir is typically .../Phase2_Deployment_Prediction/deployment
            # Go up to find project root
            output_path = Path(self.output_dir)
            project_dir = output_path.parent.parent  # deployment -> Phase2_Deployment_Prediction -> repo root
        
        project_dir = Path(project_dir).resolve()
        logs_dir = project_dir / 'logs' / 'deployment'
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{n_jobs-1}
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --output={logs_dir}/{job_name}_%A_%a.out
#SBATCH --error={logs_dir}/{job_name}_%A_%a.err

echo "=================================================================="
echo "GENOMIC DEPLOYMENT ARRAY JOB - ONE TRAIT PER JOB"
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

echo "Processing deployment job $SLURM_ARRAY_TASK_ID (one trait for all algorithms)"

python3 03a_phase2_deploy_array.py \\
    --mode process_training_job \\
    --job_id $SLURM_ARRAY_TASK_ID \\
    --output_dir {self.output_dir.resolve()} \\
    --n_jobs $SLURM_CPUS_PER_TASK

TRAIN_EXIT_CODE=$?

echo "=================================================================="
echo "Training job $SLURM_ARRAY_TASK_ID completed with exit code: $TRAIN_EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================================================="

exit $TRAIN_EXIT_CODE
"""
        
        script_path = self.output_dir / f"{job_name}_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated training array script: {script_path}")
        return str(script_path)
    
    def generate_combine_models_script(self, n_jobs, job_name='combine_models'):
        """Generate script to combine trained models"""
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
# Logs relative to job cwd (scripts/ when submitted from 05)
#SBATCH --output=../logs/deployment/{job_name}_%j.out
#SBATCH --error=../logs/deployment/{job_name}_%j.err

echo "=================================================================="
echo "COMBINING TRAINED MODELS"
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

echo "Organizing trained models from {n_jobs} deployment jobs..."

python3 03c_utils_organize_models.py \\
    --models_dir {str(self.output_dir)}/models \\
    --output_summary {str(self.output_dir)}/training_summary.json

COMBINE_EXIT_CODE=$?

# Always generate the deployment report notebook (so it exists even if organize failed)
echo ""
echo "=================================================================="
echo "Generating deployment report notebook..."
echo "=================================================================="

mkdir -p "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction"

python3 03b_phase2_report_deployment.py \\
    --models_dir {str(self.output_dir)}/models \\
    --deployment_dir {str(self.output_dir)} \\
    --output_file "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb"

REPORT_EXIT_CODE=$?

if [[ $REPORT_EXIT_CODE -eq 0 ]]; then
    echo "✅ Deployment summary generated successfully"
    echo "📁 Location: $PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb"
    echo "   Report includes: Deployment overview, deployed models, variance mask, training summary"
else
    echo "⚠️  Deployment report generation failed (exit code: $REPORT_EXIT_CODE)"
fi

if [[ $COMBINE_EXIT_CODE -ne 0 ]]; then
    echo "⚠️  Organize models step failed (exit code: $COMBINE_EXIT_CODE); report notebook was still generated."
fi

echo "=================================================================="
echo "Deployment completed with exit code: $COMBINE_EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================================================="

exit $COMBINE_EXIT_CODE
"""
        
        script_path = self.output_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated combine models script: {script_path}")
        return str(script_path)

class GenomicTrainingArray:
    """Genomic training with consistent preprocessing tracking"""
    
    def __init__(self, output_dir, random_state=42):
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()


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
    '''
    def preprocess_data(self, X, y, trait_name=None, save_preprocessing_info=True, skip_variance_filter=False):
        """Preprocess data and save preprocessing info for consistent prediction
        
        Args:
            X: Genotype matrix (already filtered for low variance if skip_variance_filter=True)
            y: Phenotype vector
            trait_name: Name of the trait (for logging)
            save_preprocessing_info: Whether to save preprocessing metadata
            skip_variance_filter: If True, skip variance filtering (already done globally)
        """
        
        # Handle missing values
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        preprocessing_info = {
            'trait_name': trait_name,
            'original_features': X.shape[1],
            'original_samples': X.shape[0],
            'valid_samples': len(X_clean),
            'removed_samples': X.shape[0] - len(X_clean)
        }
        
        # Apply variance filtering only if not already done globally
        if not skip_variance_filter and X_clean.shape[1] > 1000:
            snp_var = np.var(X_clean, axis=0)
            var_threshold = 0.01
            high_var_mask = snp_var > var_threshold
            X_filtered = X_clean[:, high_var_mask]
            
            preprocessing_info.update({
                'variance_threshold': var_threshold,
                'feature_mask': high_var_mask,
                'kept_features': np.sum(high_var_mask),
                'removed_features': np.sum(~high_var_mask),
                'filtered_features': X_filtered.shape[1]
            })
            
            removed_snps = np.sum(~high_var_mask)
            if removed_snps > 0:
                self.logger.info(f"Removed {removed_snps} low variance SNPs for {trait_name} (per-trait filtering)")
        else:
            X_filtered = X_clean
            if skip_variance_filter:
                # Variance filtering was done globally, create mask of all True
                high_var_mask = np.ones(X_clean.shape[1], dtype=bool)
                preprocessing_info.update({
                    'variance_threshold': 0.01,
                    'feature_mask': high_var_mask.tolist(),
                    'kept_features': X_clean.shape[1],
                    'removed_features': 0,
                    'filtered_features': X_clean.shape[1],
                    'note': 'Variance filtering done globally in prepare_training_jobs'
                })
            else:
            preprocessing_info.update({
                'variance_threshold': None,
                    'feature_mask': np.ones(X_clean.shape[1], dtype=bool).tolist(),
                'kept_features': X_clean.shape[1],
                'removed_features': 0,
                'filtered_features': X_clean.shape[1]
            })
            if not skip_variance_filter:
            self.logger.info("No feature filtering applied (< 1000 features)")
        
        # Save preprocessing info if requested
        if save_preprocessing_info and trait_name:
            trait_safe_name = trait_name.replace(' ', '_').replace('/', '_')
            trait_dir = self.output_dir / 'models' / trait_safe_name
            trait_dir.mkdir(parents=True, exist_ok=True)
            
            preprocessing_file = trait_dir / 'preprocessing_info.json'
            
            # Convert numpy arrays to lists for JSON serialization
            preprocessing_info_serializable = preprocessing_info.copy()
            if 'feature_mask' in preprocessing_info_serializable:
                preprocessing_info_serializable['feature_mask'] = preprocessing_info_serializable['feature_mask'].tolist()
            
            with open(preprocessing_file, 'w') as f:
                json.dump(preprocessing_info_serializable, f, indent=2)
            
            self.logger.info(f"Saved preprocessing info for {trait_name}: {preprocessing_file}")
        
        # Return format depends on whether preprocessing info was requested
        if save_preprocessing_info:
        return X_filtered, y_clean, preprocessing_info
        else:
            return X_filtered, y_clean
    '''
    def process_trait_training_job(self, job_data, shared_data):
        """Process training for all algorithms on a single trait"""
        
        trait_name = job_data['trait_name']
        algorithms = job_data['algorithms']
        
        X_all = shared_data['X_all']
        y_all = shared_data['y_all']
        trait_index = shared_data['trait_names'].index(trait_name)
        
        y_trait = y_all[:, trait_index]
        
        self.logger.info(f"Training all algorithms for trait: {trait_name}")
        '''
        # Preprocess data once for this trait and save preprocessing info
        X_processed, y_processed, preprocessing_info = self.preprocess_data(
            X_all, y_trait, trait_name, save_preprocessing_info=True
        )
        '''

        
        # Preprocess data once for this trait and save preprocessing info
        # Preprocess data - variance filtering already done globally
        X_processed, y_processed = self.preprocess_data(X_all, y_trait, skip_variance_filter=True)
        
        
        trait_results = []
        
        # Process each algorithm for this trait
        for alg_name, algorithm in algorithms.items():
            self.logger.info(f"  Training {alg_name}...")
            
            try:
                start_time = datetime.now()
                
                # Create pipeline
                if alg_name.startswith('SVR'):
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', algorithm)
                    ])
                else:
                    pipeline = Pipeline([('regressor', algorithm)])
                
                # Train on full dataset
                pipeline.fit(X_processed, y_processed)
                
                fit_time = (datetime.now() - start_time).total_seconds()
                
                # Skip cross-validation for deployment (too slow, not needed for production models)
                # CV is already done during training/validation phase
                cv_mean, cv_std = 0.0, 0.0
                
                # Performance metrics including preprocessing info
                performance = {
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'fit_time': fit_time,
                    'n_training_samples': len(X_processed),
                    'n_features_used': X_processed.shape[1],
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'timestamp': datetime.now().isoformat()
                    #'preprocessing_info': preprocessing_info
                }
                
                # Store result
                trait_result = {
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'trained_model': pipeline,
                    'performance': performance,
                    #'preprocessing_info': preprocessing_info,
                    'success': True
                }
                
                trait_results.append(trait_result)
                
                self.logger.info(f"    ✅ {alg_name}: CV R²={cv_mean:.4f}±{cv_std:.4f}, "
                               f"Features: {X_processed.shape[1]}, Time={fit_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"    ❌ {alg_name} failed: {str(e)}")
                
                # Store failed result
                trait_result = {
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'trained_model': None,
                    'performance': None,
                    #'preprocessing_info': preprocessing_info,
                    'success': False,
                    'error': str(e)
                }
                
                trait_results.append(trait_result)
                continue
        
        self.logger.info(f"Completed training {len([r for r in trait_results if r['success']])} algorithms for {trait_name}")
        return trait_results
    """Genomic training with array job support - ONE JOB PER TRAIT"""
    
    def __init__(self, output_dir, random_state=42):
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'deployment'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"deployment_{timestamp}.log"
        
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
        """Get all algorithms for training"""
        require_all = os.environ.get("BREEDAI_REQUIRE_ALL_ALGOS", "1") != "0"
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
            'BayesianRidge': BayesianRidge(compute_score=True),
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=4
            ),
            'SVR_RBF': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR_Linear': SVR(kernel='linear', C=1.0),
            'NeuralNet_MLP': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=self.random_state
            ),
        }
        
        if BGLR_AVAILABLE:
            algorithms.update({
                'BayesA': BGLRRegressor(method='BayesA', seed=self.random_state),
                'BayesB': BGLRRegressor(method='BayesB', seed=self.random_state),
                'BayesCpi': BGLRRegressor(method='BayesCpi', seed=self.random_state),
            })
            self.logger.info("Added BayesA/B/Cpi (R/BGLR)")
        
        if GPFLOW_AVAILABLE:
            algorithms.update({
                'GP_RBF': GPflowRegressor(kernel_type='rbf', max_iter=300, use_sparse=True, n_inducing=50),
                'GP_Matern52': GPflowRegressor(kernel_type='matern52', max_iter=300, use_sparse=True, n_inducing=50),
            })
            self.logger.info("Added Gaussian Process algorithms")

        if XGBOOST_AVAILABLE:
            algorithms['XGBoost'] = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=4, verbosity=0
            )
            self.logger.info("Added XGBoost algorithm")
        
        if LIGHTGBM_AVAILABLE:
            algorithms['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=self.random_state, n_jobs=4, verbose=-1
            )
            self.logger.info("Added LightGBM algorithm")
        
        return algorithms
    
    
    def process_trait_training_job(self, job_data, shared_data):
        """Process training for one trait across all algorithms"""
        
        trait_name = job_data['trait_name']
        algorithms = job_data['algorithms']
        
        X_all = shared_data['X_all']
        y_all = shared_data['y_all']
        trait_names = shared_data['trait_names']
        
        trait_index = trait_names.index(trait_name)
        y_trait = y_all[:, trait_index]
        
        self.logger.info(f"Training all algorithms for trait: {trait_name}")
        
        trait_results = []
        
        # Process each algorithm for this trait
        for alg_name, algorithm in algorithms.items():
            self.logger.info(f"  Training {alg_name} for trait: {trait_name}")
            
            try:
                start_time = datetime.now()
                
                # Preprocess data for this trait
                # Skip variance filtering since it was already done globally in prepare_training_jobs
                # This matches the training phase preprocessing (same steps, but on full dataset)
                X_processed, y_processed = self.preprocess_data(X_all, y_trait, skip_variance_filter=True)
                
                # Create pipeline with scaler if needed
                if alg_name in ['SVR_RBF', 'SVR_Linear', 'RandomForest', 'XGBoost', 'LightGBM', 'NeuralNet_MLP']:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', algorithm)
                    ])
                else:
                    pipeline = Pipeline([('regressor', algorithm)])
                
                # Train on full dataset
                pipeline.fit(X_processed, y_processed)
                
                fit_time = (datetime.now() - start_time).total_seconds()
                
                # Skip cross-validation for deployment (too slow, not needed for production models)
                # CV is already done during training/validation phase
                cv_mean, cv_std = 0.0, 0.0
                
                # Performance metrics
                performance = {
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'fit_time': fit_time,
                    'n_training_samples': len(X_processed),
                    'n_features_used': X_processed.shape[1],
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store result
                algorithm_result = {
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'trained_model': pipeline,
                    'performance': performance,
                    'success': True
                }
                
                trait_results.append(algorithm_result)
                
                self.logger.info(f"    ✅ {alg_name}: Time={fit_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"    ❌ {alg_name} failed: {str(e)}")
                
                # Store failed result
                algorithm_result = {
                    'algorithm': alg_name,
                    'trait': trait_name,
                    'trained_model': None,
                    'performance': None,
                    'success': False,
                    'error': str(e)
                }
                
                trait_results.append(algorithm_result)
                continue
        
        self.logger.info(f"Completed training {trait_name} for {len([r for r in trait_results if r['success']])}/{len(algorithms)} algorithms")
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
            self.logger.info(f"Gmatrix calculated: shape {G.shape}")
            return G, info
        except Exception as e:
            self.logger.error(f"Failed to calculate Gmatrix: {str(e)}")
            return None, None
    
    def prepare_training_jobs(self, X_file, y_file, traits=None, benchmark_results=None, calculate_gmatrix=False):
        """Prepare training jobs - ONE JOB PER TRAIT"""
        
        self.logger.info("Preparing deployment array jobs (one per trait)...")
        
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
        
        # SNP ALIGNMENT: Align input features to training features (handle different SNP panels)
        training_mask_file = Path(self.output_dir).parent.parent / 'Phase1_Learning_Benchmarking' / 'training_validation' / 'variance_filter_mask.npy'
        
        # Load training genotype columns to know expected SNP names
        training_geno_file = Path(self.output_dir).parent.parent / 'input' / 'Geno.csv'
        training_snp_names = None
        if training_geno_file.exists():
            training_snp_names = list(pd.read_csv(training_geno_file, index_col=0, nrows=0).columns)
            self.logger.info(f"Loaded training SNP names: {len(training_snp_names)} SNPs from {training_geno_file}")
        
        if X.shape[1] == len(training_snp_names or []):
            self.logger.info(f"✅ Feature count matches training ({X.shape[1]} features) — no alignment needed")
        elif training_snp_names is not None:
            input_snps = set(X_df.columns)
            training_snps = set(training_snp_names)
            common_snps = input_snps & training_snps
            extra_snps = input_snps - training_snps
            missing_snps = training_snps - input_snps
            
            n_common = len(common_snps)
            n_training = len(training_snps)
            overlap_pct = 100.0 * n_common / n_training if n_training > 0 else 0.0
            
            self.logger.info(f"SNP alignment:")
            self.logger.info(f"  Input SNPs: {len(input_snps)}")
            self.logger.info(f"  Training SNPs: {n_training}")
            self.logger.info(f"  Common SNPs: {n_common} ({overlap_pct:.1f}%)")
            self.logger.info(f"  Extra SNPs (will drop): {len(extra_snps)}")
            self.logger.info(f"  Missing SNPs (will fill): {len(missing_snps)}")
            
            # Overlap thresholds
            REJECT_THRESHOLD = 50.0
            WARN_THRESHOLD = 80.0
            
            if overlap_pct < REJECT_THRESHOLD:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ REJECTED: SNP overlap {overlap_pct:.1f}% is below minimum threshold {REJECT_THRESHOLD:.0f}%\n"
                    f"{'='*80}\n"
                    f"Input has {len(input_snps)} SNPs, training used {n_training} SNPs, only {n_common} in common.\n"
                    f"Predictions would be unreliable with this level of imputation.\n"
                    f"\n"
                    f"Please provide a genotype file with higher SNP overlap to the training data.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                print(error_msg)
                raise ValueError(f"REJECTED: SNP overlap {overlap_pct:.1f}% below minimum {REJECT_THRESHOLD:.0f}%")
            
            if overlap_pct < WARN_THRESHOLD:
                self.logger.warning(f"⚠️  SNP overlap {overlap_pct:.1f}% is below warning threshold {WARN_THRESHOLD:.0f}%")
                self.logger.warning("Predictions may be degraded due to heavy imputation of missing SNPs")
                print(f"⚠️  WARNING: SNP overlap {overlap_pct:.1f}% — predictions may be degraded")
            
            # Align: keep only training SNPs in training order, fill missing with mean
            X_aligned = pd.DataFrame(index=X_df.index, columns=training_snp_names, dtype=float)
            
            for snp in training_snp_names:
                if snp in X_df.columns:
                    X_aligned[snp] = X_df[snp].values
                else:
                    X_aligned[snp] = np.nan  # Will be filled during preprocessing
            
            # Fill missing SNPs with column mean from available data, or 0
            for col in X_aligned.columns:
                if X_aligned[col].isna().all():
                    X_aligned[col] = 0.0  # No data at all for this SNP
                elif X_aligned[col].isna().any():
                    X_aligned[col] = X_aligned[col].fillna(X_aligned[col].mean())
            
            X_df = X_aligned
            X = X_df.values.astype(float)
            
            self.logger.info(f"✅ SNP alignment complete: {X.shape[1]} features (aligned to training order)")
            self.logger.info(f"  Filled {len(missing_snps)} missing SNPs with mean/zero")
            self.logger.info(f"  Dropped {len(extra_snps)} extra SNPs not in training set")
        else:
            if training_mask_file.exists():
                training_mask = np.load(training_mask_file)
                expected_features = training_mask.shape[0]
                if X.shape[1] != expected_features:
                    self.logger.warning(f"Feature count mismatch ({X.shape[1]} vs {expected_features}) but no SNP names available for alignment")
            self.logger.warning("Could not load training SNP names — skipping alignment")
        
        # IMPORTANT: Filter low variance SNPs FIRST, before calculating G-matrix
        # Prefer using the Phase 1 variance mask for consistency
        self.logger.info("Filtering low variance SNPs before G-matrix calculation...")
        
        phase1_mask_file = Path(self.output_dir).parent.parent / 'Phase1_Learning_Benchmarking' / 'training_validation' / 'variance_filter_mask.npy'
        
        if phase1_mask_file.exists() and X.shape[1] == len(np.load(phase1_mask_file)):
            high_var_mask = np.load(phase1_mask_file)
            X_filtered = X[:, high_var_mask]
            X_df_filtered = X_df.iloc[:, high_var_mask]
            removed_snps = np.sum(~high_var_mask)
            self.logger.info(f"Applied Phase 1 variance mask: kept {np.sum(high_var_mask)}, removed {removed_snps}")
            if removed_snps > 0:
                self.logger.info(f"Filtered data: X={X_filtered.shape} (from {X.shape})")
        elif X.shape[1] > 1000:
            # Fallback: calculate variance directly on input genotypes (no phenotype mask needed)
            snp_var = np.var(X, axis=0)
            var_threshold = 0.01
            high_var_mask = snp_var > var_threshold
            X_filtered = X[:, high_var_mask]
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
        
        # Save variance filter mask for reuse in prediction
        variance_mask_file = self.output_dir / 'variance_filter_mask.npy'
        np.save(variance_mask_file, high_var_mask)
        self.logger.info(f"Saved variance filter mask to: {variance_mask_file}")
        self.logger.info(f"  Mask shape: {high_var_mask.shape}, kept features: {np.sum(high_var_mask)}/{len(high_var_mask)}")
        
        # Calculate Gmatrix on FILTERED SNPs (if requested)
        G = None
        G_info = None
        if calculate_gmatrix:
            gmatrix_dir = self.output_dir / 'gmatrix'
            self.logger.info(f"Calculating G-matrix on filtered SNPs: {X_df_filtered.shape[1]} markers")
            G, G_info = self.calculate_gmatrix_if_needed(X_df_filtered, output_dir=gmatrix_dir)
        
        self.logger.info(f"Training data: X={X_filtered.shape}, y={y.shape}, traits={len(trait_names)}")
        
        # Create shared data (full dataset for training, using FILTERED X)
        shared_data = {
            'X_all': X_filtered,  # Use filtered X
            'y_all': y,
            'trait_names': trait_names,
            'variance_filter_mask': high_var_mask  # Store mask for reference
        }
        
        # Add Gmatrix if calculated
        if G is not None:
            shared_data['Gmatrix'] = G
            shared_data['Gmatrix_info'] = G_info
            self.logger.info("Gmatrix included in shared data")
        
        # Create training jobs - ONE PER TRAIT
        algorithms = self.get_algorithms()
        array_manager = TrainingArrayManager(self.output_dir)
        jobs = array_manager.create_trait_training_jobs(trait_names, algorithms)
        
        # Save shared data
        shared_data_file = array_manager.save_job_data(shared_data, 'shared', 'shared_data')
        
        # Save individual job data
        for job in jobs:
            array_manager.save_job_data(job, job['job_id'], 'job')
        
        self.logger.info(f"Prepared {len(jobs)} deployment jobs (one per trait)")
        return len(jobs), array_manager

def main():
    parser = argparse.ArgumentParser(description='Array Job Genomic Training')
    parser.add_argument('--mode', required=True, choices=['prepare', 'process_training_job'])
    parser.add_argument('--X_file', help='Path to genotype data (CSV)')
    parser.add_argument('--y_file', help='Path to phenotype data (CSV)')
    parser.add_argument('--output_dir', default='./training_array')
    parser.add_argument('--job_id', type=int, help='Job ID for array processing')
    parser.add_argument('--n_jobs', type=int, help='Number of parallel jobs')
    parser.add_argument('--traits', nargs='*', help='Specific traits to train')
    parser.add_argument('--benchmark_results', help='Path to benchmarking results')
    parser.add_argument('--use_full_dataset', action='store_true', help='Use full dataset for training')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_cv_folds', type=int, default=5)
    parser.add_argument('--calculate_gmatrix', action='store_true',
                       help='Calculate Gmatrix (genomic relationship matrix) during preparation')
    
    args = parser.parse_args()
    
    trainer = GenomicTrainingArray(
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    array_manager = TrainingArrayManager(args.output_dir)
    
    if args.mode == 'prepare':
        if not args.X_file or not args.y_file:
            print("Error: X_file and y_file required for prepare mode")
            return 1
        
        n_jobs, array_manager = trainer.prepare_training_jobs(
            args.X_file, args.y_file, args.traits, args.benchmark_results,
            calculate_gmatrix=args.calculate_gmatrix
        )
        
        # Generate SLURM scripts
        # Get project directory from output_dir
        output_path = Path(args.output_dir)
        project_dir = output_path.parent.parent  # deployment -> Phase2_Deployment_Prediction -> repo root
        training_script = array_manager.generate_slurm_training_array_script(n_jobs, project_dir=project_dir)
        combine_script = array_manager.generate_combine_models_script(n_jobs)
        
        print(f"Deployment jobs prepared: {n_jobs} jobs (one per algorithm)")
        print(f"SLURM deployment script: {training_script}")
        print(f"Combine models script: {combine_script}")
        
    elif args.mode == 'process_training_job':
        if args.job_id is None:
            print("Error: job_id required for process_training_job mode")
            return 1
        
        try:
            job_data = array_manager.load_job_data(args.job_id, 'job')
            shared_data = array_manager.load_job_data('shared', 'shared_data')
            
            trait_results = trainer.process_trait_training_job(job_data, shared_data)
            
            # Save trained models for this trait (across all algorithms)
            saved_files = array_manager.save_trained_models_for_trait(
                trait_results, args.job_id, job_data['trait_name']
            )
            
            print(f"Deployment job {args.job_id} completed successfully")
            print(f"Saved {len(saved_files)} models for {job_data['trait_name']} across {len([r for r in trait_results if r['success']])} algorithms")
            
        except Exception as e:
            print(f"Training job {args.job_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    
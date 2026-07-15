#!/usr/bin/env python3
"""
File: 04a_phase2_predict_array.py - EXACT SAME PREPROCESSING AS TRAINING
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
from pathlib import Path
from datetime import datetime
from typing import Optional
import importlib.util

# Expected algorithm list for consistency checks
EXPECTED_ALGORITHMS = [
    "GBLUP_Ridge",
    "GBLUP_RidgeCV",
    "LASSO",
    "LASSO_CV",
    "ElasticNet",
    "ElasticNet_CV",
    "BayesianRidge",
    "RandomForest",
    "SVR_RBF",
    "SVR_Linear",
    "NeuralNet_MLP",
    "BayesA",
    "BayesB",
    "BayesCpi",
    "GP_RBF",
    "GP_Matern52",
    "XGBoost",
    "LightGBM",
]

# Try to import openpyxl for Excel writing
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logging.warning("openpyxl not available - Excel files will not be generated")

# Import Gmatrix calculation function from numbered filename
def _load_module_from_file(module_filename, module_name):
    module_path = Path(__file__).parent / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    _gmatrix_module = _load_module_from_file("01a_utils_calculate_gmatrix.py", "utils_calculate_gmatrix")
    calculate_gmatrix = _gmatrix_module.calculate_gmatrix
    calculate_gmatrix_from_file = _gmatrix_module.calculate_gmatrix_from_file
    GMATRIX_AVAILABLE = True
except Exception:
    GMATRIX_AVAILABLE = False
    logging.warning("Gmatrix calculation module not available")

class PredictionArrayManager:
    """Manages prediction array jobs"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'prediction'
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'prediction.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_prediction_jobs(self, new_X, animal_ids, available_models, chunk_size=50):
        """Create prediction jobs by chunking animals"""
        jobs = []
        n_animals = len(new_X)
        
        if chunk_size > n_animals:
            chunk_size = n_animals
            self.logger.info(f"Adjusted chunk size to {chunk_size} (total animals)")
        
        n_chunks = (n_animals + chunk_size - 1) // chunk_size
        
        job_id = 0
        for chunk_id in range(n_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, n_animals)
            
            chunk_X = new_X[start_idx:end_idx]
            chunk_animal_ids = animal_ids[start_idx:end_idx]
            
            jobs.append({
                'job_id': job_id,
                'chunk_id': chunk_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_animals_in_chunk': len(chunk_animal_ids),
                'animal_ids': chunk_animal_ids,
                'X_chunk': chunk_X,
                'available_models': available_models
            })
            job_id += 1
            
            self.logger.info(f"Job {job_id-1}: chunk {chunk_id}, animals {start_idx}-{end_idx-1} ({len(chunk_animal_ids)} animals)")
        
        self.logger.info(f"Created {len(jobs)} prediction jobs (chunk size: {chunk_size})")
        return jobs
    
    def save_job_data(self, data, job_id, data_type='job'):
        """Save job data"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return str(filepath)
    
    def load_job_data(self, job_id, data_type='job'):
        """Load job data"""
        if isinstance(job_id, str):
            filename = f"{data_type}_{job_id}.pkl"
        else:
            filename = f"{data_type}_{job_id:04d}.pkl"
        
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Job data not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_prediction_results(self, results, job_id):
        """Save prediction results"""
        filename = f"pred_results_{job_id:04d}.pkl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        return str(filepath)
    
    def combine_prediction_results(self, n_jobs):
        """Combine prediction results from all jobs"""
        all_results = {}
        successful_jobs = 0
        
        for job_id in range(n_jobs):
            filename = f"pred_results_{job_id:04d}.pkl"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        job_results = pickle.load(f)
                    
                    for trait_name, trait_results in job_results.items():
                        if trait_name not in all_results:
                            all_results[trait_name] = {
                                'animal_ids': [],
                                'individual_predictions': {},
                                'ensemble_predictions': {}
                            }
                        
                        all_results[trait_name]['animal_ids'].extend(trait_results['animal_ids'])
                        
                        for model_name, preds in trait_results['individual_predictions'].items():
                            if model_name not in all_results[trait_name]['individual_predictions']:
                                all_results[trait_name]['individual_predictions'][model_name] = []
                            all_results[trait_name]['individual_predictions'][model_name].extend(preds)
                        
                        for ensemble_name, preds in trait_results['ensemble_predictions'].items():
                            if ensemble_name not in all_results[trait_name]['ensemble_predictions']:
                                all_results[trait_name]['ensemble_predictions'][ensemble_name] = []
                            all_results[trait_name]['ensemble_predictions'][ensemble_name].extend(preds)
                    
                    successful_jobs += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load prediction job {job_id}: {e}")
            else:
                self.logger.warning(f"Missing prediction results for job {job_id}")
        
        self.logger.info(f"Combined prediction results from {successful_jobs} jobs")
        return all_results
    
    def generate_slurm_prediction_array_script(self, n_jobs, job_name='genomic_prediction'):
        """Generate SLURM prediction array script"""
        # SBATCH directives are parsed before the script runs, so $PROJECT_DIR is NOT
        # expanded there — write an absolute logs path instead.
        logs_dir = Path(self.output_dir).parent.parent / 'logs' / 'prediction'
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{n_jobs-1}
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output={logs_dir}/{job_name}_%A_%a.out
#SBATCH --error={logs_dir}/{job_name}_%A_%a.err

mkdir -p logs

echo "=================================================================="
echo "GENOMIC PREDICTION ARRAY JOB"
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

echo "Processing prediction job $SLURM_ARRAY_TASK_ID"

python3 04a_phase2_predict_array.py \\
    --mode process_prediction_job \\
    --job_id $SLURM_ARRAY_TASK_ID \\
    --output_dir {self.output_dir}

echo "=================================================================="
echo "Prediction job $SLURM_ARRAY_TASK_ID completed"
echo "Finished at: $(date)"
echo "=================================================================="
"""
        
        script_path = self.output_dir / f"{job_name}_array.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated prediction array script: {script_path}")
        return str(script_path)
    
    def generate_combine_predictions_script(self, n_jobs, job_name='combine_predictions'):
        """Generate script to combine predictions"""
        
        # Extract output_dir to avoid f-string nesting issues
        output_dir_str = str(self.output_dir)
        logs_dir_str = str(Path(self.output_dir).parent.parent / 'logs' / 'prediction')
        
        # Use .format() instead of f-string to avoid nesting issues with triple quotes
        script_content = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

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
# Results saved directly in Phase2_Deployment_Prediction/prediction folder
RESULTS_DIR="{output_dir_str}"

cd "$SCRIPTS_DIR"

python3 04a_phase2_predict_array.py \\
    --mode combine_predictions \\
    --n_jobs {n_jobs} \\
    --output_dir {output_dir_str} \\
    --results_dir "$RESULTS_DIR"

COMBINE_EXIT_CODE=$?

if [[ $COMBINE_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Generating Phase 2 prediction reports..."

    # Create Phase 2 folder for results
    mkdir -p "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction"

    # Generate Phase 2 EDA/QC report (preprocessing on new prediction file)
    echo ""
    echo "Generating Phase 2 EDA/QC report (preprocessing documentation)..."
    
    # Find dataset directory
    if [[ -d "$PROJECT_DIR/input" ]]; then
        DATASET_DIR="$PROJECT_DIR/input"
    else
        DATASET_DIR="$PROJECT_DIR/data"
    fi
    
    # Deployment directory
    DEPLOYMENT_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/deployment"
    
    # Output file
    EDA_QC_REPORT="$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.1_Preprocessing_report.ipynb"
    
    if [[ -f "$PROJECT_DIR/scripts/04b_phase2_report_preprocessing.py" ]]; then
        python3 "$PROJECT_DIR/scripts/04b_phase2_report_preprocessing.py" \\
            --dataset_dir "$DATASET_DIR" \\
            --deployment_dir "$DEPLOYMENT_DIR" \\
            --output_file "$EDA_QC_REPORT"
        
        EDA_QC_EXIT_CODE=$?
        
        if [[ $EDA_QC_EXIT_CODE -eq 0 ]]; then
            echo "✅ Phase 2 EDA/QC report generated successfully"
            echo "📁 Location: $EDA_QC_REPORT"
            echo "   Report includes: EDA on new prediction file, missing data handling, feature filtering"
        else
            echo "❌ Phase 2 EDA/QC report generation failed"
            echo "   You can generate it manually using:"
            echo "   python3 $PROJECT_DIR/scripts/04b_phase2_report_preprocessing.py --dataset_dir $DATASET_DIR --deployment_dir $DEPLOYMENT_DIR --output_file $EDA_QC_REPORT"
        fi
    else
        echo "⚠️  Prediction EDA report generator not found: $PROJECT_DIR/scripts/04b_phase2_report_preprocessing.py"
        echo "   Skipping EDA report generation"
    fi

    # Generate Phase 2 Prediction Results Report
    echo ""
    echo "Generating Phase 2 Prediction Results report (all algorithms + ensembles)..."
    
    RESULTS_DIR_FOR_REPORT="$PROJECT_DIR/Phase2_Deployment_Prediction/prediction"
    RESULTS_REPORT="$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.3_prediction_report.ipynb"
    
    if [[ -f "$PROJECT_DIR/scripts/04c_phase2_report_predictions.py" ]]; then
        python3 "$PROJECT_DIR/scripts/04c_phase2_report_predictions.py" \\
            --results_dir "$RESULTS_DIR_FOR_REPORT" \\
            --output_file "$RESULTS_REPORT"
        
        RESULTS_REPORT_EXIT_CODE=$?
        
        if [[ $RESULTS_REPORT_EXIT_CODE -eq 0 ]]; then
            echo "✅ Phase 2 Prediction Results report generated successfully"
            echo "📁 Location: $RESULTS_REPORT"
            echo "   Report includes: All algorithms, ensembles, trait-wise and animal-wise analysis"
        else
            echo "❌ Phase 2 Prediction Results report generation failed"
        fi
    else
        echo "⚠️  Prediction Results report generator not found"
    fi

    echo ""
    echo "🎯 Phase 2 results available at:"
    echo "   📊 Reports: $PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/"
    echo "     ├── 2.1_Preprocessing_report.ipynb (EDA & G-Matrix: Preprocessing Documentation)"
    echo "     └── 2.3_prediction_report.ipynb (All Algorithms + Ensembles)"
    echo "   📋 Raw predictions: $PROJECT_DIR/Phase2_Deployment_Prediction/prediction/"
    echo "     ├── all_predictions_all_algorithms_*.csv (All algorithms for all traits)"
    echo "     └── ensemble_weighted_average_predictions_*.csv (Weighted Average ensemble only)"

else
    echo "❌ Prediction combination failed - skipping report generation"
fi

echo "Prediction combination completed"
""".format(
            job_name=job_name,
            n_jobs=n_jobs,
            output_dir_str=output_dir_str,
            logs_dir=logs_dir_str
        )
        
        script_path = self.output_dir / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated combine predictions script: {script_path}")
        return str(script_path)

class GenomicPredictionArray:
    """Genomic prediction with EXACT SAME preprocessing as training"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = self.output_dir.parent.parent
        log_dir = project_root / 'logs' / 'prediction'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"prediction_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, X, full_X_for_variance=None, variance_mask=None):
        """
        Apply preprocessing logic with variance filtering
        
        Args:
            X: Data to preprocess
            full_X_for_variance: Full dataset for variance calculation (fallback method)
            variance_mask: Saved variance filter mask (preferred method)
        """
        
        self.logger.info(f"Preprocessing {X.shape[0]} animals, {X.shape[1]} features")
        
        # Handle missing values in the data we're processing
        X_clean = X.copy()
        if np.any(np.isnan(X_clean)):
            self.logger.warning("Missing values found - filling with column mode")
            for col in range(X_clean.shape[1]):
                col_data = X_clean[:, col]
                if np.any(np.isnan(col_data)):
                    unique_vals, counts = np.unique(col_data[~np.isnan(col_data)], return_counts=True)
                    if len(unique_vals) > 0:
                        mode_val = unique_vals[np.argmax(counts)]
                        X_clean[np.isnan(col_data), col] = mode_val
                    else:
                        X_clean[np.isnan(col_data), col] = 0
        
        # Apply variance filtering using saved mask (preferred) or calculate from data (fallback)
        # If both are None, data is already filtered and we only handle missing values
        if variance_mask is not None:
            # Use saved variance mask (from training/deployment)
            if variance_mask.shape[0] != X_clean.shape[1]:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ FEATURE MISMATCH ERROR (in preprocessing)\n"
                    f"{'='*80}\n"
                    f"Variance mask dimension mismatch: mask has {variance_mask.shape[0]} features, but data has {X_clean.shape[1]} features.\n"
                    f"\n"
                    f"This indicates inconsistent feature dimensions during preprocessing.\n"
                    f"All phases must use the SAME input dataset with the SAME features.\n"
                    f"\n"
                    f"Execution stopped to prevent inconsistent results.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                print(error_msg)
                return None
            
            X_filtered = X_clean[:, variance_mask]
            kept_snps = np.sum(variance_mask)
            removed_snps = len(variance_mask) - kept_snps
            
            if removed_snps > 0:
                self.logger.info(f"Applied saved variance mask: kept {kept_snps} features, removed {removed_snps} low variance features")
            else:
                self.logger.info(f"Applied saved variance mask: kept all {kept_snps} features")
        elif full_X_for_variance is not None:
            # Fallback: Calculate variance from training dataset
            variance_data = full_X_for_variance
            self.logger.info(f"Calculating variance from training dataset: {variance_data.shape}")
            
            if variance_data.shape[1] > 1000:
                snp_var = np.var(variance_data, axis=0)
                var_threshold = 0.01
                high_var_mask = snp_var > var_threshold
                X_filtered = X_clean[:, high_var_mask]
                
                removed_snps = np.sum(~high_var_mask)
                kept_snps = np.sum(high_var_mask)
                
                if removed_snps > 0:
                    self.logger.info(f"Selected {kept_snps} features from {X_clean.shape[1]} original features "
                                   f"to match training procedure and discarded {removed_snps} low variance features")
            else:
                X_filtered = X_clean
                self.logger.info(f"No variance filtering applied (< 1000 features) - using all {X_filtered.shape[1]} features")
        else:
            # Both variance_mask and full_X_for_variance are None
            # This means data was already filtered during preparation - only handle missing values
            X_filtered = X_clean
            self.logger.info(f"Data already filtered during preparation - using all {X_filtered.shape[1]} features (only missing value handling applied)")
        
        return X_filtered
    
    def find_available_models(self, models_dir):
        """Find all available trained models (.joblib and BGLR custom .npz)"""
        models_dir = Path(models_dir)
        available_models = {}
        
        # Resolve to absolute path so we always use the same path
        models_dir = models_dir.resolve() if models_dir.exists() else models_dir
        
        self.logger.info(f"Searching for models in: {models_dir}")
        
        if not models_dir.exists():
            self.logger.error(f"Models directory not found: {models_dir}")
            return available_models
        
        if not models_dir.is_dir():
            self.logger.error(f"Models path is not a directory: {models_dir}")
            return available_models
        
        trait_dirs_found = 0
        for trait_dir in models_dir.iterdir():
            if trait_dir.is_dir():
                trait_dirs_found += 1
                trait_name = trait_dir.name.replace('_', ' ')
                trait_models = {}
                BGLR_ALGOS = ('BayesA', 'BayesB', 'BayesCpi')
                # Prefer .npz for BGLR (joblib.load often fails for R-backed objects on prediction node)
                for npz_file in trait_dir.glob('*.npz'):
                    model_name = npz_file.stem
                    meta_file = trait_dir / f"{model_name}_custom_meta.json"
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r') as f:
                                meta = json.load(f)
                            if meta.get('type') == 'bglr':
                                trait_models[model_name] = str(npz_file)
                                self.logger.debug(f"Found custom BGLR model: {model_name}")
                        except Exception:
                            pass
                for model_file in trait_dir.glob('*.joblib'):
                    model_name = model_file.stem
                    if 'performance' in model_name or 'scaler' in model_name:
                        continue
                    # Prefer .npz for BGLR (already added above)
                    if model_name in BGLR_ALGOS and model_name in trait_models:
                        continue
                    trait_models[model_name] = str(model_file)
                # GPflow checkpoint dirs (prefer over .joblib for GP_RBF / GP_Matern52)
                for gp_alg in ('GP_RBF', 'GP_Matern52'):
                    ckpt_dir = trait_dir / f"{gp_alg}_ckpt"
                    scalers_file = trait_dir / f"{gp_alg}_scalers.joblib"
                    if ckpt_dir.is_dir() and scalers_file.exists():
                        trait_models[gp_alg] = str(ckpt_dir)
                        self.logger.debug(f"Found GPflow checkpoint model: {gp_alg}")
                
                if trait_models:
                    available_models[trait_name] = trait_models
                    self.logger.debug(f"Found {len(trait_models)} models for trait: {trait_name}")
                else:
                    self.logger.warning(f"No valid models found in trait directory: {trait_dir.name}")
        
        self.logger.info(f"Scanned {trait_dirs_found} trait directories, found models for {len(available_models)} traits")
        return available_models
    
    def _load_gpflow_from_checkpoint(self, ckpt_dir):
        """Load GPflow model from TensorFlow checkpoint (see GPflow saving_and_loading docs)."""
        ckpt_dir = Path(ckpt_dir)
        alg_name = ckpt_dir.name.replace('_ckpt', '')
        trait_dir = ckpt_dir.parent
        meta_file = trait_dir / f"{alg_name}_meta.json"
        scalers_file = trait_dir / f"{alg_name}_scalers.joblib"
        if not meta_file.exists() or not scalers_file.exists():
            self.logger.error(f"GPflow checkpoint missing meta or scalers: {ckpt_dir}")
            return None
        try:
            import tensorflow as tf
            import gpflow
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            scalers = joblib.load(scalers_file)
            scaler_X = scalers['scaler_X']
            scaler_y = scalers['scaler_y']
            pca = scalers.get('pca') if isinstance(scalers, dict) else None
            scaler_pca = scalers.get('scaler_pca') if isinstance(scalers, dict) else None
            n_features = meta.get('n_features') or getattr(scaler_X, 'n_features_in_', None)
            if n_features is None and hasattr(scaler_X, 'mean_') and scaler_X.mean_ is not None:
                n_features = scaler_X.mean_.shape[0]
            if n_features is None:
                self.logger.error(f"GPflow checkpoint: could not determine n_features for {alg_name}")
                return None
            # The GP was trained in PCA-reduced space, so the checkpointed model's
            # inducing points / kernel live in that reduced dimensionality. Rebuild on
            # n_components (fall back to n_features for legacy non-PCA checkpoints).
            model_dim = meta.get('n_components')
            if model_dim is None and pca is not None:
                model_dim = getattr(pca, 'n_components_', None)
            if model_dim is None:
                model_dim = n_features
            kernel_type = meta.get('kernel_type', 'rbf')
            use_sparse = meta.get('use_sparse', True)
            n_inducing = meta.get('n_inducing', 50)
            noise_variance = meta.get('noise_variance', 0.1)
            if kernel_type == 'rbf':
                kernel = gpflow.kernels.RBF(lengthscales=1.0, variance=1.0)
            elif kernel_type == 'matern52':
                kernel = gpflow.kernels.Matern52(lengthscales=1.0, variance=1.0)
            else:
                kernel = gpflow.kernels.RBF(lengthscales=1.0, variance=1.0)
            X_dummy = np.zeros((1, model_dim), dtype=np.float64)
            Y_dummy = np.zeros((1, 1), dtype=np.float64)
            X_tensor = tf.constant(X_dummy, dtype=tf.float64)
            Y_tensor = tf.constant(Y_dummy, dtype=tf.float64)
            if use_sparse:
                n_inducing = min(max(1, n_inducing), 1000)
                inducing = np.zeros((n_inducing, model_dim), dtype=np.float64)
                gp_model = gpflow.models.SGPR(
                    data=(X_tensor, Y_tensor),
                    kernel=kernel,
                    inducing_variable=inducing,
                )
            else:
                gp_model = gpflow.models.GPR(
                    data=(X_tensor, Y_tensor),
                    kernel=kernel,
                    noise_variance=noise_variance,
                )
            ckpt = tf.train.Checkpoint(model=gp_model)
            latest = tf.train.latest_checkpoint(str(ckpt_dir))
            if latest is None:
                self.logger.error(f"No checkpoint found in {ckpt_dir}")
                return None
            ckpt.restore(latest)
            class _GPflowPredictor:
                def __init__(self, model, sx, sy, pca=None, spca=None):
                    self.model = model
                    self.scaler_X = sx
                    self.scaler_y = sy
                    self.pca = pca
                    self.scaler_pca = spca
                def predict(self, X):
                    X_scaled = self.scaler_X.transform(np.asarray(X, dtype=np.float64))
                    # Match the training-time transform: PCA-reduce then re-standardise.
                    if self.pca is not None:
                        X_scaled = self.pca.transform(X_scaled)
                        if self.scaler_pca is not None:
                            X_scaled = self.scaler_pca.transform(X_scaled)
                    X_t = tf.constant(X_scaled, dtype=tf.float64)
                    mean, _ = self.model.predict_f(X_t)
                    mean_np = mean.numpy().flatten()
                    return self.scaler_y.inverse_transform(mean_np.reshape(-1, 1)).flatten()
            return _GPflowPredictor(gp_model, scaler_X, scaler_y, pca, scaler_pca)
        except Exception as e:
            self.logger.error(f"Failed to load GPflow checkpoint {ckpt_dir}: {e}")
            return None

    def load_model(self, model_path):
        """Load a trained model (joblib, BGLR .npz, or GPflow checkpoint)."""
        model_path = Path(model_path)
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
        # GPflow checkpoint directory
        if model_path.is_dir() and model_path.name.endswith('_ckpt'):
            return self._load_gpflow_from_checkpoint(model_path)
        # BGLR custom format
        if model_path.suffix == '.npz':
            try:
                data = np.load(model_path, allow_pickle=True)
                coef = data['coef']
                intercept = float(np.atleast_1d(data['intercept']).flat[0])
                # Return a small predictor object
                class _BGLRPredictor:
                    def predict(self, X):
                        return (np.asarray(X) @ coef) + intercept
                return _BGLRPredictor()
            except Exception as e:
                self.logger.error(f"Failed to load BGLR model {model_path}: {e}")
                return None
        # joblib
        try:
            return joblib.load(model_path)
        except Exception as e:
            self.logger.warning(f"joblib.load failed for {model_path}: {e}")
            # For BGLR, try same-name .npz in same dir (in case deployment saved .npz later or we have legacy .joblib)
            if model_path.stem in ('BayesA', 'BayesB', 'BayesCpi'):
                npz_path = model_path.with_suffix('.npz')
                if npz_path.exists():
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                        coef = data['coef']
                        intercept = float(np.atleast_1d(data['intercept']).flat[0])
                        class _BGLRPredictor:
                            def predict(self, X):
                                return (np.asarray(X) @ coef) + intercept
                        return _BGLRPredictor()
                    except Exception as e2:
                        self.logger.error(f"Failed to load BGLR .npz fallback {npz_path}: {e2}")
            return None
    
    def process_prediction_job(self, job_data, shared_data):
        """Process predictions for a chunk of animals"""
        
        chunk_id = job_data['chunk_id']
        animal_ids = job_data['animal_ids']
        X_chunk = job_data['X_chunk']
        available_models = job_data['available_models']
        n_animals_in_chunk = job_data['n_animals_in_chunk']
        
        # Get variance mask or training data for variance calculation
        variance_mask = shared_data.get('variance_mask')
        full_X = shared_data.get('full_X_for_variance')
        
        self.logger.info(f"Processing chunk {chunk_id}: {n_animals_in_chunk} animals")
        
        # Apply preprocessing
        # If variance_mask is None and full_X is None, data was already filtered during preparation
        # In that case, preprocessing should only handle missing values, not variance filtering
        if variance_mask is None and full_X is None:
            # Data is already filtered - only handle missing values
            X_processed = self.preprocess_data(X_chunk, variance_mask=None, full_X_for_variance=None)
        else:
            # Apply preprocessing using saved variance mask (preferred) or training data (fallback)
            X_processed = self.preprocess_data(X_chunk, full_X_for_variance=full_X, variance_mask=variance_mask)
        
        if X_processed is None:
            self.logger.error(f"Preprocessing failed for chunk {chunk_id}")
            return {}
        
        self.logger.info(f"Chunk {chunk_id} preprocessing: {X_chunk.shape[1]} -> {X_processed.shape[1]} features")
        
        chunk_results = {}
        
        # Process each trait
        for trait_name, trait_models in available_models.items():
            self.logger.info(f"  Predicting {trait_name}...")
            
            trait_results = {
                'animal_ids': animal_ids,
                'individual_predictions': {},
                'ensemble_predictions': {}
            }
            
            # Individual model predictions
            individual_preds = {}
            successful_models = 0
            
            for model_name, model_path in trait_models.items():
                model = self.load_model(model_path)
                if model is not None:
                    try:
                        predictions = model.predict(X_processed)
                        individual_preds[model_name] = predictions.tolist()
                        trait_results['individual_predictions'][model_name] = predictions.tolist()
                        
                        self.logger.info(f"    ✅ {model_name}: mean={np.mean(predictions):.2f}, "
                                       f"range=[{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
                        successful_models += 1
                        
                    except Exception as e:
                        self.logger.error(f"    ❌ {model_name} prediction failed: {e}")
                        continue
            
            # Ensemble predictions
            if len(individual_preds) > 1:
                try:
                    pred_matrix = np.column_stack(list(individual_preds.values()))
                    simple_avg = np.mean(pred_matrix, axis=1)
                    trait_results['ensemble_predictions']['Simple_Average'] = simple_avg.tolist()
                    
                    median_pred = np.median(pred_matrix, axis=1)
                    trait_results['ensemble_predictions']['Median'] = median_pred.tolist()
                    
                    weights = []
                    for model_name in individual_preds.keys():
                        if 'Ridge' in model_name or 'LASSO' in model_name or 'Bayes' in model_name:
                            weights.append(2.0)
                        elif 'RandomForest' in model_name or 'XGBoost' in model_name:
                            weights.append(1.5)
                        else:
                            weights.append(1.0)
                    
                    weights = np.array(weights) / np.sum(weights)
                    weighted_avg = np.average(pred_matrix, axis=1, weights=weights)
                    trait_results['ensemble_predictions']['Weighted_Average'] = weighted_avg.tolist()
                    
                    self.logger.info(f"    ✅ Ensembles: Simple={np.mean(simple_avg):.2f}, "
                                   f"Median={np.mean(median_pred):.2f}, Weighted={np.mean(weighted_avg):.2f}")
                    
                    # Stacking ensemble using Phase 1 weights
                    try:
                        trait_safe = trait_name.replace(' ', '_').replace('/', '_')
                        stacking_dir = Path(self.output_dir).parent.parent / 'Phase1_Learning_Benchmarking' / 'training_validation' / f'stacking_{trait_safe}'
                        weights_file = stacking_dir / 'stacking_weights_summary.csv'
                        
                        if weights_file.exists():
                            import pandas as pd_stack
                            stacking_weights_df = pd_stack.read_csv(weights_file)
                            
                            stacking_model_names = list(individual_preds.keys())
                            stacking_weights = []
                            for model_name in stacking_model_names:
                                row = stacking_weights_df[stacking_weights_df['model'] == model_name]
                                if len(row) > 0:
                                    stacking_weights.append(float(row['mean_weight'].iloc[0]))
                                else:
                                    stacking_weights.append(0.0)
                            
                            stacking_weights = np.array(stacking_weights)
                            weight_sum = np.sum(stacking_weights)
                            if weight_sum > 0:
                                stacking_weights = stacking_weights / weight_sum
                                stacking_pred = np.average(pred_matrix, axis=1, weights=stacking_weights)
                                trait_results['ensemble_predictions']['Stacking_NonNeg_Ridge'] = stacking_pred.tolist()
                                self.logger.info(f"    ✅ Stacking_NonNeg_Ridge: mean={np.mean(stacking_pred):.2f} "
                                               f"(using {np.sum(stacking_weights > 0.001)} non-zero weights from Phase 1)")
                            else:
                                self.logger.warning(f"    ⚠️  Stacking weights sum to zero for {trait_name} — skipping")
                        else:
                            self.logger.info(f"    ℹ️  No stacking weights found at {weights_file} — skipping Stacking_NonNeg_Ridge")
                    except Exception as e:
                        self.logger.warning(f"    ⚠️  Stacking ensemble failed: {e}")
                    
                except Exception as e:
                    self.logger.error(f"    ❌ Ensemble creation failed: {e}")
            
            if successful_models > 0:
                chunk_results[trait_name] = trait_results
                self.logger.info(f"  Completed {trait_name}: {successful_models} successful models")
        
        return chunk_results
    
    def prepare_prediction_jobs(self, new_X_file, models_dir, chunk_size=50, training_X_file=None):
        """Prepare prediction array jobs
        
        Args:
            new_X_file: Path to new animals genotype file (for prediction)
            models_dir: Directory containing trained models
            chunk_size: Number of animals per prediction job
            training_X_file: Path to training dataset (deprecated - now uses saved variance mask)
        """
        
        self.logger.info("Preparing prediction array jobs...")
        self.logger.info("Prediction phase: Using saved variance filter mask from deployment/training")
        
        # Load prediction data
        X_new_df = pd.read_csv(new_X_file, index_col=0)
        X_new = X_new_df.values
        animal_ids = X_new_df.index.tolist()
        
        self.logger.info(f"Prediction data: {X_new.shape[0]} animals, {X_new.shape[1]} original markers")
        
        # SNP ALIGNMENT (prediction): align new input to training SNP space before applying variance mask.
        # This mirrors deployment behavior and supports files with different SNP panels.
        models_dir_path = Path(models_dir)
        project_root = models_dir_path.parent.parent.parent
        training_geno_file = project_root / 'input' / 'Geno.csv'
        training_snp_names = None

        if training_geno_file.exists():
            training_snp_names = list(pd.read_csv(training_geno_file, index_col=0, nrows=0).columns)
            self.logger.info(f"Loaded training SNP names: {len(training_snp_names)} from {training_geno_file}")

            # Compare the SNP panel itself, not just its size: an input can match the
            # training marker count while carrying a different (or differently ordered)
            # panel, which must still be overlap-checked and realigned.
            if list(X_new_df.columns) != list(training_snp_names):
                input_snps = set(X_new_df.columns)
                training_snps = set(training_snp_names)
                common_snps = input_snps & training_snps
                extra_snps = input_snps - training_snps
                missing_snps = training_snps - input_snps

                n_common = len(common_snps)
                n_training = len(training_snps)
                overlap_pct = 100.0 * n_common / n_training if n_training > 0 else 0.0

                self.logger.info("SNP alignment (prediction input):")
                self.logger.info(f"  Input SNPs: {len(input_snps)}")
                self.logger.info(f"  Training SNPs: {n_training}")
                self.logger.info(f"  Common SNPs: {n_common} ({overlap_pct:.1f}%)")
                self.logger.info(f"  Extra SNPs (will drop): {len(extra_snps)}")
                self.logger.info(f"  Missing SNPs (will fill): {len(missing_snps)}")

                reject_threshold = 50.0
                warn_threshold = 80.0

                if overlap_pct < reject_threshold:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"❌ REJECTED: SNP overlap {overlap_pct:.1f}% is below minimum threshold {reject_threshold:.0f}%\n"
                        f"{'='*80}\n"
                        f"Input has {len(input_snps)} SNPs, training used {n_training} SNPs, only {n_common} in common.\n"
                        f"Predictions would be unreliable with this level of imputation.\n"
                        f"{'='*80}\n"
                    )
                    self.logger.error(error_msg)
                    print(error_msg)
                    raise ValueError(f"REJECTED: SNP overlap {overlap_pct:.1f}% below minimum {reject_threshold:.0f}%")

                if overlap_pct < warn_threshold:
                    self.logger.warning(f"⚠️  SNP overlap {overlap_pct:.1f}% is below warning threshold {warn_threshold:.0f}%")
                    print(f"⚠️  WARNING: SNP overlap {overlap_pct:.1f}% — predictions may be degraded")

                # Align to training SNP order and fill missing SNPs
                X_aligned = pd.DataFrame(index=X_new_df.index, columns=training_snp_names, dtype=float)
                for snp in training_snp_names:
                    if snp in X_new_df.columns:
                        X_aligned[snp] = X_new_df[snp].values
                    else:
                        X_aligned[snp] = np.nan

                for col in X_aligned.columns:
                    if X_aligned[col].isna().all():
                        X_aligned[col] = 0.0
                    elif X_aligned[col].isna().any():
                        X_aligned[col] = X_aligned[col].fillna(X_aligned[col].mean())

                X_new_df = X_aligned
                X_new = X_new_df.values.astype(float)
                self.logger.info(f"✅ SNP alignment complete: {X_new.shape[1]} features (training order)")
                self.logger.info(f"  Filled {len(missing_snps)} missing SNPs with mean/zero")
                self.logger.info(f"  Dropped {len(extra_snps)} extra SNPs not in training set")

        # Try to load saved variance filter mask from deployment directory FIRST
        # This must be done before G-matrix calculation and data processing
        deployment_dir = models_dir_path.parent  # models_dir is usually .../deployment/models, so parent is .../deployment
        variance_mask_file = deployment_dir / 'variance_filter_mask.npy'
        
        variance_mask = None
        if variance_mask_file.exists():
            self.logger.info(f"Loading saved variance filter mask from: {variance_mask_file}")
            variance_mask = np.load(variance_mask_file)
            self.logger.info(f"  Loaded mask: shape={variance_mask.shape}, kept features={np.sum(variance_mask)}/{len(variance_mask)}")
            
            # VALIDATION: Check that input data has the same number of features as the mask
            if variance_mask.shape[0] != X_new.shape[1]:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ FEATURE MISMATCH ERROR\n"
                    f"{'='*80}\n"
                    f"Prediction input data has {X_new.shape[1]} features, but deployment mask expects {variance_mask.shape[0]} features.\n"
                    f"\n"
                    f"This indicates the input dataset is different from the deployment/training dataset.\n"
                    f"All phases must use the SAME input dataset with the SAME features.\n"
                    f"\n"
                    f"Deployment mask file: {variance_mask_file}\n"
                    f"Prediction input file: {new_X_file}\n"
                    f"\n"
                    f"Please ensure:\n"
                    f"  1. The same Geno.csv file is used for training, deployment, and prediction\n"
                    f"  2. The file has not been modified between phases\n"
                    f"  3. The same preprocessing steps are applied\n"
                    f"\n"
                    f"Execution stopped to prevent inconsistent results.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                print(error_msg)
                raise ValueError(f"Feature mismatch: prediction input has {X_new.shape[1]} features, but deployment mask expects {variance_mask.shape[0]} features")
            else:
                self.logger.info(f"✅ Feature validation passed: input has {X_new.shape[1]} features (matches deployment mask)")
        else:
            # Fallback: try training directory (Phase 1)
            # Look for training output directory
            training_dir = project_root / 'Phase1_Learning_Benchmarking' / 'training_validation'
            training_mask_file = training_dir / 'variance_filter_mask.npy'
            
            if training_mask_file.exists():
                self.logger.info(f"Loading saved variance filter mask from training: {training_mask_file}")
                variance_mask = np.load(training_mask_file)
                self.logger.info(f"  Loaded mask: shape={variance_mask.shape}, kept features={np.sum(variance_mask)}/{len(variance_mask)}")
        
                # VALIDATION: Check that input data has the same number of features as the mask
                if variance_mask.shape[0] != X_new.shape[1]:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"❌ FEATURE MISMATCH ERROR\n"
                        f"{'='*80}\n"
                        f"Prediction input data has {X_new.shape[1]} features, but training mask expects {variance_mask.shape[0]} features.\n"
                        f"\n"
                        f"This indicates the input dataset is different from the training dataset.\n"
                        f"All phases must use the SAME input dataset with the SAME features.\n"
                        f"\n"
                        f"Training mask file: {training_mask_file}\n"
                        f"Prediction input file: {new_X_file}\n"
                        f"\n"
                        f"Please ensure:\n"
                        f"  1. The same Geno.csv file is used for training, deployment, and prediction\n"
                        f"  2. The file has not been modified between phases\n"
                        f"  3. The same preprocessing steps are applied\n"
                        f"\n"
                        f"Execution stopped to prevent inconsistent results.\n"
                        f"{'='*80}\n"
                    )
                    self.logger.error(error_msg)
                    print(error_msg)
                    raise ValueError(f"Feature mismatch: prediction input has {X_new.shape[1]} features, but training mask expects {variance_mask.shape[0]} features")
                else:
                    self.logger.info(f"✅ Feature validation passed: input has {X_new.shape[1]} features (matches training mask)")
            else:
                self.logger.warning("Variance filter mask not found in deployment or training directories")
                self.logger.warning(f"  Looked in: {variance_mask_file}")
                self.logger.warning(f"  And in: {training_mask_file}")
                self.logger.warning("  Falling back to using training dataset for variance calculation")
                
                # Fallback to old method if mask not found
                if training_X_file and Path(training_X_file).exists():
                    self.logger.info(f"Loading training dataset for variance calculation: {training_X_file}")
                    X_train_df = pd.read_csv(training_X_file, index_col=0)
                    X_train_full = X_train_df.values
                    self.logger.info(f"Training data: {X_train_full.shape[0]} animals, {X_train_full.shape[1]} markers")
                else:
                    self.logger.warning("Training dataset not provided - using prediction data for variance calculation")
                    X_train_full = X_new
        
        # Apply variance filtering to prediction data BEFORE G-matrix calculation
        # This ensures G-matrix uses the same features as the models
        # Note: Feature validation was already done above when loading the mask
        data_already_filtered = False
        if variance_mask is not None:
            # Double-check dimensions (should already be validated above, but verify again)
            if variance_mask.shape[0] != X_new.shape[1]:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"❌ FEATURE MISMATCH ERROR\n"
                    f"{'='*80}\n"
                    f"Variance mask dimension mismatch: mask has {variance_mask.shape[0]} features, but prediction data has {X_new.shape[1]} features.\n"
                    f"\n"
                    f"This should have been caught earlier. Please check the input data consistency.\n"
                    f"{'='*80}\n"
                )
                self.logger.error(error_msg)
                print(error_msg)
                return 0, None
            
            # Apply mask to filter the data
            self.logger.info(f"Applying variance filter mask: {X_new.shape[1]} -> {np.sum(variance_mask)} features")
            X_new_filtered = X_new[:, variance_mask]
            X_new_df_filtered = X_new_df.iloc[:, variance_mask]
            
            self.logger.info(f"Filtered prediction data: {X_new_filtered.shape[0]} animals, {X_new_filtered.shape[1]} features (from {X_new.shape[1]})")
            
            # Update X_new and X_new_df to use filtered data
            X_new = X_new_filtered
            X_new_df = X_new_df_filtered
            data_already_filtered = True
            self.logger.info("✅ Data filtered using saved variance mask - same features as deployment")
        else:
            # Fallback: will calculate variance from training data during preprocessing
            self.logger.info("No variance mask found - will calculate variance during preprocessing")
        
        # Calculate and save G-matrix for prediction data (using FILTERED data)
        if GMATRIX_AVAILABLE:
            self.logger.info("Calculating G-matrix for prediction data (using filtered features)...")
            try:
                gmatrix_dir = self.output_dir / 'gmatrix'
                gmatrix_dir.mkdir(parents=True, exist_ok=True)
                
                # Use filtered data for G-matrix calculation
                G, G_info = calculate_gmatrix(
                    X_new_df,  # Already filtered if mask was applied
                    method='vanRaden',
                    standardize=True,
                    save_intermediate=True,
                    output_dir=gmatrix_dir
                )
                
                self.logger.info(f"✅ G-matrix calculated: shape {G.shape}")
                self.logger.info(f"   Diagonal range: [{np.min(np.diag(G)):.4f}, {np.max(np.diag(G)):.4f}]")
                self.logger.info(f"   Saved to: {gmatrix_dir}")
                
                # Print summary
                print("\n" + "="*60)
                print("G-MATRIX CALCULATION SUMMARY (Prediction Phase)")
                print("="*60)
                print(f"Method: {G_info['method']}")
                print(f"Animals: {G_info['n_animals']}")
                print(f"Markers: {G_info['n_markers']}")
                print(f"Computation time: {G_info['computation_time']:.2f} seconds")
                print(f"Standardized: {G_info['standardized']}")
                print(f"G-matrix saved to: {gmatrix_dir}/Gmatrix.csv")
                print("="*60 + "\n")
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate G-matrix: {str(e)}")
        else:
            self.logger.warning("G-matrix calculation not available - skipping")
        
        # Find available models
        available_models = self.find_available_models(models_dir)
        if not available_models:
            self.logger.error("No trained models found")
            return 0, None

        # Audit available vs expected algorithms
        expected_set = set(EXPECTED_ALGORITHMS)
        audit = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "expected_algorithms": sorted(expected_set),
            "by_trait": {}
        }
        any_missing = False
        for trait, models in sorted(available_models.items()):
            available = sorted(models.keys())
            missing = sorted(expected_set - set(available))
            audit["by_trait"][trait] = {
                "available_algorithms": available,
                "missing_algorithms": missing,
                "available_count": len(available),
                "missing_count": len(missing)
            }
            if missing:
                any_missing = True
        audit_file = self.output_dir / "prediction_models_audit.json"
        with open(audit_file, "w") as f:
            json.dump(audit, f, indent=2)
        self.logger.info(f"Saved prediction models audit: {audit_file}")
        if any_missing:
            self.logger.warning("Some expected algorithms are missing for prediction. See prediction_models_audit.json")
            require_all = os.environ.get("BREEDAI_REQUIRE_ALL_ALGOS", "1") != "0"
            if require_all:
                raise RuntimeError(
                    "Missing expected algorithms for prediction. "
                    "Set BREEDAI_REQUIRE_ALL_ALGOS=0 to allow skipping. "
                    f"See {audit_file} for details."
                )
        
        # Test preprocessing
        # If mask was applied above, data is already filtered - test preprocessing without filtering
        # If mask was not found, test with fallback method
        sample_X = X_new[:min(5, X_new.shape[0])]
        
        if data_already_filtered:
            # Data was already filtered above, so test preprocessing without filtering (only missing value handling)
            sample_processed = self.preprocess_data(sample_X, variance_mask=None, full_X_for_variance=None)
            
            if sample_processed is None:
                self.logger.error("Preprocessing test failed")
                return 0, None
            
            self.logger.info(f"Preprocessing test: {sample_X.shape[1]} -> {sample_processed.shape[1]} features")
            self.logger.info("Using saved variance filter mask from deployment/training (already applied to data)")
        else:
            # Fallback: use training data for variance calculation
            X_train_clean = X_train_full.copy()
            if np.any(np.isnan(X_train_clean)):
                self.logger.info("Cleaning training dataset for variance calculation")
                for col in range(X_train_clean.shape[1]):
                    col_data = X_train_clean[:, col]
                    if np.any(np.isnan(col_data)):
                        unique_vals, counts = np.unique(col_data[~np.isnan(col_data)], return_counts=True)
                        if len(unique_vals) > 0:
                            mode_val = unique_vals[np.argmax(counts)]
                            X_train_clean[np.isnan(col_data), col] = mode_val
                        else:
                            X_train_clean[np.isnan(col_data), col] = 0
        
            # Test preprocessing using training data for variance
            sample_X = X_new[:min(5, X_new.shape[0])]
            sample_processed = self.preprocess_data(sample_X, full_X_for_variance=X_train_clean)
        
        if sample_processed is None:
            self.logger.error("Preprocessing test failed")
            return 0, None
        
        # Create prediction jobs (using filtered data if mask was applied)
        array_manager = PredictionArrayManager(self.output_dir)
        jobs = array_manager.create_prediction_jobs(X_new, animal_ids, available_models, chunk_size)
        
        # Create shared data
        # If mask was applied above, variance_mask is now None (data already filtered)
        # If mask was not found, variance_mask is None and we have X_train_clean for fallback
        shared_data = {
            'available_models': available_models,
            'models_dir': str(models_dir),
            'original_features': X_new.shape[1],  # This is now the filtered count if mask was applied
            'processed_features': sample_processed.shape[1]
        }
        
        # Use data_already_filtered flag to determine if data was filtered
        if data_already_filtered:
            # Data was already filtered above - don't filter again in preprocessing
            shared_data['variance_mask'] = None  # Signal that data is already filtered
            shared_data['full_X_for_variance'] = None
            self.logger.info("Shared data: variance mask already applied to data, skipping filtering in preprocessing")
        elif 'X_train_clean' in locals():
            # Mask not found - will use training data for variance calculation during preprocessing
            shared_data['full_X_for_variance'] = X_train_clean  # Fallback: training dataset
            shared_data['variance_mask'] = None
            self.logger.info("Shared data includes training dataset for variance calculation (fallback)")
        else:
            # No filtering method available
            shared_data['variance_mask'] = None
            shared_data['full_X_for_variance'] = None
            self.logger.warning("No variance filtering method available - using all features")
        
        # Save shared data
        array_manager.save_job_data(shared_data, 'shared', 'shared_data')
        
        # Save job data
        for job in jobs:
            array_manager.save_job_data(job, job['job_id'], 'job')
        
        self.logger.info(f"Prepared {len(jobs)} prediction jobs")
        self.logger.info(f"Using same dataset for training and prediction - should get identical feature filtering")
        
        return len(jobs), array_manager

def main():
    parser = argparse.ArgumentParser(description='Array Job Genomic Prediction')
    parser.add_argument('--mode', required=True, choices=['prepare', 'process_prediction_job', 'combine_predictions'])
    parser.add_argument('--new_X_file', help='Path to new animals genotype data (CSV)')
    parser.add_argument('--models_dir', help='Directory with trained models')
    parser.add_argument('--training_X_file', help='Path to training dataset (for variance calculation, like test phase)')
    parser.add_argument('--output_dir', default='./Phase2_Deployment_Prediction/prediction')
    parser.add_argument('--results_dir', default=None, help='Directory to save results (defaults to output_dir)')
    parser.add_argument('--job_id', type=int, help='Job ID for array processing')
    parser.add_argument('--n_jobs', type=int, help='Total number of jobs')
    parser.add_argument('--chunk_size', type=int, default=50, help='Animals per job')
    # ADD THESE TWO LINES TO ACCEPT THE ARGUMENTS (even if not used)
    parser.add_argument('--include_uncertainty', action='store_true', help='Include prediction uncertainty (not implemented)')
    parser.add_argument('--create_ensembles', action='store_true', help='Create ensemble predictions (always done)')
    
    args = parser.parse_args()
    # ... rest of main function remains the same
    
    predictor = GenomicPredictionArray(output_dir=args.output_dir)
    array_manager = PredictionArrayManager(args.output_dir)
    
    if args.mode == 'prepare':
        if not args.new_X_file or not args.models_dir:
            print("Error: new_X_file and models_dir required for prepare mode")
            return 1
        
        n_jobs, array_manager = predictor.prepare_prediction_jobs(
            args.new_X_file, args.models_dir, args.chunk_size, args.training_X_file
        )
        
        if n_jobs > 0:
            pred_script = array_manager.generate_slurm_prediction_array_script(n_jobs)
            combine_script = array_manager.generate_combine_predictions_script(n_jobs)
            
            print(f"Prediction jobs prepared: {n_jobs} jobs")
            print(f"SLURM prediction script: {pred_script}")
            print(f"Combine script: {combine_script}")
        
    elif args.mode == 'process_prediction_job':
        if args.job_id is None:
            print("Error: job_id required for process_prediction_job mode")
            return 1
        
        try:
            job_data = array_manager.load_job_data(args.job_id, 'job')
            shared_data = array_manager.load_job_data('shared', 'shared_data')
            
            results = predictor.process_prediction_job(job_data, shared_data)
            array_manager.save_prediction_results(results, args.job_id)
            
            print(f"Prediction job {args.job_id} completed successfully")
            
        except Exception as e:
            print(f"Prediction job {args.job_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.mode == 'combine_predictions':
        if not args.n_jobs:
            print("Error: n_jobs required for combine_predictions mode")
            return 1
        
        combined_results = array_manager.combine_prediction_results(args.n_jobs)
        
        if combined_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use output_dir if results_dir not specified
            if args.results_dir:
                results_dir = Path(args.results_dir)
            else:
                results_dir = Path(args.output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual CSV files and create Excel file
            excel_file = None
            excel_writer = None
            if EXCEL_AVAILABLE:
                excel_file = results_dir / f"all_predictions_{timestamp}.xlsx"
                excel_writer = pd.ExcelWriter(excel_file, engine='openpyxl')
            else:
                print("⚠️  openpyxl not available - skipping Excel file generation")
                print("   Install with: pip install openpyxl")
            
            # Create comprehensive data structures for reports
            all_animals = set()
            for trait_results in combined_results.values():
                all_animals.update(trait_results['animal_ids'])
            all_animals = sorted(list(all_animals))
            
            # 1. Create combined CSV with all algorithms for all traits (one row per animal)
            combined_all_df = pd.DataFrame(index=all_animals)
            
            # 2. Create ensemble-only CSV (Weighted Average ensemble for each trait)
            ensemble_only_df = pd.DataFrame(index=all_animals)
            
            for trait_name, trait_results in combined_results.items():
                animal_ids = trait_results['animal_ids']
                results_df = pd.DataFrame(index=animal_ids)
                
                for model_name, preds in trait_results['individual_predictions'].items():
                    results_df[model_name] = preds
                    # Add to combined dataframe with trait prefix (handle missing animals with NaN)
                    combined_all_df[f"{trait_name}_{model_name}"] = pd.Series(preds, index=animal_ids)
                
                for ensemble_name, preds in trait_results['ensemble_predictions'].items():
                    results_df[f'Ensemble_{ensemble_name}'] = preds
                    # Add to combined dataframe with trait prefix
                    combined_all_df[f"{trait_name}_Ensemble_{ensemble_name}"] = pd.Series(preds, index=animal_ids)
                    
                    # Add Weighted Average to ensemble-only CSV
                    if ensemble_name == 'Weighted_Average':
                        ensemble_only_df[f"{trait_name}_Weighted_Average"] = pd.Series(preds, index=animal_ids)
                
                trait_safe_name = trait_name.replace(' ', '_').replace('/', '_')
                
                # Save to CSV (individual file per trait)
                trait_file = results_dir / f"{trait_safe_name}_predictions.csv"
                results_df.to_csv(trait_file)
                
                # Save to Excel (one sheet per trait) if available
                if excel_writer:
                    # Excel sheet names are limited to 31 characters
                    sheet_name = trait_safe_name[:31] if len(trait_safe_name) > 31 else trait_safe_name
                    results_df.to_excel(excel_writer, sheet_name=sheet_name, index=True)
                
                print(f"Saved predictions for {trait_name}:")
                print(f"  CSV: {trait_file}")
                if excel_writer:
                    print(f"  Excel sheet: {sheet_name}")
                print(f"  Shape: {results_df.shape}")
                print(f"  Individual models: {len(trait_results['individual_predictions'])}")
                print(f"  Ensemble models: {len(trait_results['ensemble_predictions'])}")
            
            # Save combined CSV with all algorithms for all traits
            combined_all_file = results_dir / f"all_predictions_all_algorithms_{timestamp}.csv"
            combined_all_df.to_csv(combined_all_file)
            print(f"\n✅ Combined CSV with all algorithms for all traits: {combined_all_file}")
            print(f"   Shape: {combined_all_df.shape} (animals × (traits × algorithms))")
            
            # Save ensemble-only CSV (Weighted Average)
            ensemble_only_file = results_dir / f"ensemble_weighted_average_predictions_{timestamp}.csv"
            ensemble_only_df.to_csv(ensemble_only_file)
            print(f"✅ Ensemble-only CSV (Weighted Average): {ensemble_only_file}")
            print(f"   Shape: {ensemble_only_df.shape} (animals × traits)")
            
            if excel_writer:
                excel_writer.close()
                print(f"\n✅ Excel file with all predictions: {excel_file}")
            
            print(f"Combined predictions for {len(combined_results)} traits")
            
            summary_file = results_dir / f"prediction_summary_{timestamp}.json"
            summary = {
                'timestamp': timestamp,
                'total_traits': len(combined_results),
                'total_animals': len(list(combined_results.values())[0]['animal_ids']) if combined_results else 0,
                'traits': list(combined_results.keys()),
                'models_per_trait': {
                    trait: len(results['individual_predictions']) 
                    for trait, results in combined_results.items()
                }
            }

            audit_file = Path(args.output_dir) / "prediction_models_audit.json"
            if audit_file.exists():
                try:
                    with open(audit_file, "r") as f:
                        audit = json.load(f)
                    summary["expected_algorithms"] = audit.get("expected_algorithms", [])
                    summary["missing_algorithms_by_trait"] = {
                        trait: data.get("missing_algorithms", [])
                        for trait, data in audit.get("by_trait", {}).items()
                    }
                except Exception as e:
                    print(f"⚠️  Could not read prediction models audit: {e}")
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Summary report: {summary_file}")
            
            # Generate Phase 2 EDA/QC report (preprocessing on new prediction file)
            try:
                print("\n" + "="*70)
                print("GENERATING PHASE 2 EDA/QC REPORT")
                print("="*70)
                
                # Determine paths
                project_root = Path(args.output_dir).parent.parent
                notebooks_dir = project_root / 'notebooks' / 'Phase2_Deployment_Prediction'
                notebooks_dir.mkdir(parents=True, exist_ok=True)
                
                # Find dataset directory
                dataset_dir = project_root / 'input'
                if not dataset_dir.exists():
                    dataset_dir = project_root / 'data'
                
                # Deployment directory (for variance mask)
                deployment_dir = project_root / 'Phase2_Deployment_Prediction' / 'deployment'
                
                # Output file
                eda_qc_report_file = notebooks_dir / '2.1_Preprocessing_report.ipynb'
                
                print(f"📁 Dataset directory: {dataset_dir}")
                print(f"📁 Deployment directory: {deployment_dir}")
                print(f"📁 Output file: {eda_qc_report_file}")
                
                # Import and call prediction EDA report generator
                scripts_dir = Path(__file__).parent
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                
                try:
                    # Generate the EDA report notebook programmatically
                    report_module = _load_module_from_file("04b_phase2_report_preprocessing.py", "phase2_report_preprocessing")
                    create_prediction_eda_report = report_module.create_prediction_eda_report
                    
                    create_prediction_eda_report(
                        str(dataset_dir),
                        str(deployment_dir),
                        str(eda_qc_report_file)
                    )
                    print(f"✅ Phase 2 EDA/QC report generated: {eda_qc_report_file}")
                    print("   Report includes: EDA on new prediction file, missing data handling, feature filtering")
                except ImportError:
                    # Fallback: use subprocess to call the script
                    import subprocess
                    result = subprocess.run([
                        'python3',
                        str(scripts_dir / '04b_phase2_report_preprocessing.py'),
                        '--dataset_dir', str(dataset_dir),
                        '--deployment_dir', str(deployment_dir),
                        '--output_file', str(eda_qc_report_file)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✅ Phase 2 EDA/QC report generated: {eda_qc_report_file}")
                        print("   Report includes: EDA on new prediction file, missing data handling, feature filtering")
                    else:
                        raise RuntimeError(f"Failed to generate report: {result.stderr}")
            except Exception as e:
                print(f"⚠️  Could not generate Phase 2 EDA/QC report: {e}")
                import traceback
                traceback.print_exc()
                print("   You can generate it manually using:")
                project_root = Path(args.output_dir).parent.parent
                notebooks_dir = project_root / 'notebooks' / 'Phase2_Deployment_Prediction'
                dataset_dir = project_root / 'input' if (project_root / 'input').exists() else project_root / 'data'
                deployment_dir = project_root / 'Phase2_Deployment_Prediction' / 'deployment'
                print(f"   python3 {scripts_dir}/04b_phase2_report_preprocessing.py --dataset_dir {dataset_dir} --deployment_dir {deployment_dir} --output_file {notebooks_dir / '2.1_Preprocessing_report.ipynb'}")
        else:
            print("No prediction results found to combine")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
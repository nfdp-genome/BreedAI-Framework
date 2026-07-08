#!/bin/bash
#SBATCH --job-name=deployment_prep
#SBATCH --partition=batch
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=../logs/deployment/deployment_prep_%j.out
#SBATCH --error=../logs/deployment/deployment_prep_%j.err

# File: 03_phase2_deploy_models.sh

echo "=================================================================="
echo "MODEL DEPLOYMENT PREPARATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=================================================================="

# Project paths - automatically detect BreedAI-Framework directory
# Function to find project root by looking for environment.yml
find_project_dir() {
    local search_dir="$1"
    while [[ "$search_dir" != "/" ]]; do
        if [[ -f "$search_dir/environment.yml" ]] && [[ -d "$search_dir/scripts" ]]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done
    return 1
}

# Try multiple methods to find project directory
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    # SLURM sets this to the directory where sbatch was called
    PROJECT_DIR="$(find_project_dir "$SLURM_SUBMIT_DIR")"
fi

if [[ -z "$PROJECT_DIR" ]] && [[ -f "${BASH_SOURCE[0]}" ]]; then
    # Try from script location (works when not in SLURM spool)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [[ -n "$SCRIPT_DIR" ]] && [[ "$SCRIPT_DIR" != "/var/spool/slurm"* ]]; then
        PROJECT_DIR="$(find_project_dir "$SCRIPT_DIR")"
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    # Try from current working directory
    PROJECT_DIR="$(find_project_dir "$PWD")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: Could not find BreedAI-Framework directory"
    echo "Please ensure you're running from within the project or set PROJECT_DIR environment variable"
    exit 1
fi

SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/cattle_dataset/input"
DEPLOYMENT_ARRAY_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/deployment"
# Results saved directly in Phase2_Deployment_Prediction/deployment folder
RESULTS_DIR="$DEPLOYMENT_ARRAY_DIR"
BENCHMARK_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation"

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="$(conda info --base 2>/dev/null)"

module purge
module load R
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_ENV_PATH" || true
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$CONDA_PREFIX" ] || ( [ "$CONDA_PREFIX" != "$USER_CONDA_ENV_PATH" ] && [ "$(basename "$CONDA_PREFIX")" != "$(basename "$USER_CONDA_ENV_PATH")" ] ); then
    echo "ERROR: Failed to activate Conda environment"
    exit 1
fi

# Create directories
mkdir -p $DEPLOYMENT_ARRAY_DIR "$PROJECT_DIR/logs/deployment"
mkdir -p "$DEPLOYMENT_ARRAY_DIR/models"
cd $SCRIPTS_DIR

# Data files
X_FILE="$DATA_DIR/Geno.csv"
Y_FILE="$DATA_DIR/Pheno.csv"

echo "Configuration:"
echo "  Data files: X=$X_FILE, Y=$Y_FILE"
echo "  Deployment array dir: $DEPLOYMENT_ARRAY_DIR"
echo "  Results dir: $RESULTS_DIR"

if [[ ! -f "$X_FILE" ]] || [[ ! -f "$Y_FILE" ]]; then
    echo "ERROR: Data files not found"
    exit 1
fi

# Check for train-validate results
BENCHMARK_RESULTS=""
if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
    BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
    echo "Found train-validate results: $BENCHMARK_RESULTS"
else
    BENCHMARK_DIR="$PROJECT_DIR/train_validate_array"
    if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
        BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
        echo "Found train-validate results (fallback): $BENCHMARK_RESULTS"
    else
        echo "❌ ERROR: Phase 1 results not found."
        echo "Please run Phase 1 first to generate benchmarking results."
        echo "Checked locations:"
        echo "  - $PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/"
        echo "  - $PROJECT_DIR/train_validate_array/"
        exit 1
    fi
fi

echo "=================================================================="
echo "PREPARING DEPLOYMENT ARRAY JOBS"
echo "=================================================================="

# Prepare deployment array jobs
python 03a_phase2_deploy_array.py \
    --mode prepare \
    --X_file "$X_FILE" \
    --y_file "$Y_FILE" \
    --output_dir "$DEPLOYMENT_ARRAY_DIR" \
    --benchmark_results "$BENCHMARK_RESULTS" \
    --use_full_dataset \
    --random_state 42 \
    --n_cv_folds 0

PREP_STATUS=$?

if [[ $PREP_STATUS -eq 0 ]]; then
    echo "✅ Deployment array preparation successful"
    
    # Submit deployment array job
    DEPLOY_ARRAY_SCRIPT="$DEPLOYMENT_ARRAY_DIR/genomic_deployment_array.sh"
    if [[ -f "$DEPLOY_ARRAY_SCRIPT" ]]; then
        DEPLOY_ARRAY_JOB=$(sbatch --parsable --export=ALL "$DEPLOY_ARRAY_SCRIPT")
        echo "Deployment array job ID: $DEPLOY_ARRAY_JOB"
        
        # Submit model validation job
        VALIDATION_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:$DEPLOY_ARRAY_JOB \
            --job-name=model_validation \
            --time=02:00:00 \
            --mem=16G \
            --cpus-per-task=4 \
            --wrap="cd $SCRIPTS_DIR && python validate_trained_models.py --models_dir $DEPLOYMENT_ARRAY_DIR/models --output_dir $DEPLOYMENT_ARRAY_DIR/validation")
        
        echo "Model validation job ID: $VALIDATION_JOB"
        
        echo ""
        echo "=================================================================="
        echo "DEPLOYMENT PIPELINE JOBS SUBMITTED"
        echo "=================================================================="
        echo "  1. Deployment array: $DEPLOY_ARRAY_JOB"
        echo "  2. Model validation: $VALIDATION_JOB"
        echo ""
        echo "Monitor with: squeue -u $USER"
    fi
else
    echo "❌ Deployment array preparation failed"
    exit 1
fi

echo "=================================================================="
echo "Completed at: $(date)"
echo "=================================================================="
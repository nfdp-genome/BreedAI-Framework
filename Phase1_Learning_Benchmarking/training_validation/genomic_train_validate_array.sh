#!/bin/bash
#SBATCH --job-name=genomic_train_validate
#SBATCH --array=0-3
#SBATCH --time=14-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=batch
#SBATCH --output=$PROJECT_DIR/logs/train_validate/genomic_train_validate_%A_%a.out
#SBATCH --error=$PROJECT_DIR/logs/train_validate/genomic_train_validate_%A_%a.err

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
CONDA_BASE_PATH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_ENV_PATH" || true

# Fix for CXXABI error
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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
python3 02a_phase1_train_validate_array.py \
    --mode process_job \
    --job_id $SLURM_ARRAY_TASK_ID \
    --output_dir /ibex/project/c2293/BreedAI_Poster/BreedAI-public/Phase1_Learning_Benchmarking/training_validation \
    --n_jobs $SLURM_CPUS_PER_TASK  --stack_alpha 0.01 --stack_fit_intercept true --stack_standardize_cols true --stack_normalize_weights false --stack_n_splits 5 --stack_outer_splits 5 --stack_inner_splits 3

JOB_EXIT_CODE=$?

echo "=================================================================="
echo "Trait job $SLURM_ARRAY_TASK_ID completed with exit code: $JOB_EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================================================="

exit $JOB_EXIT_CODE

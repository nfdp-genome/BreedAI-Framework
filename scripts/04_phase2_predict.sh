#!/bin/bash
#SBATCH --job-name=pred_prep
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output=../logs/prediction/pred_prep_%j.out
#SBATCH --error=../logs/prediction/pred_prep_%j.err

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
DATA_DIR="$PROJECT_DIR/data"
MODELS_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/deployment/models"
NEW_ANIMALS_FILE="$DATA_DIR/Geno.csv"
PREDICTION_ARRAY_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/prediction"
# Results saved directly in Phase2_Deployment_Prediction/prediction folder
RESULTS_DIR="$PREDICTION_ARRAY_DIR"

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
_ep=$(CONDA_BASE_PATH="$CONDA_BASE_PATH" ENVNAME="$USER_CONDA_ENV_PATH" timeout 45 bash -c 'source "$CONDA_BASE_PATH/etc/profile.d/conda.sh" 2>/dev/null; conda activate "$ENVNAME" >/dev/null 2>&1; printf %s "$CONDA_PREFIX"' 2>/dev/null || true)
if [ -n "$_ep" ] && [ -x "$_ep/bin/python" ]; then export CONDA_PREFIX="$_ep"; export PATH="$_ep/bin:$PATH"; export CONDA_DEFAULT_ENV="$USER_CONDA_ENV_PATH"; fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Create directories
mkdir -p "$PROJECT_DIR/logs/prediction" "$PREDICTION_ARRAY_DIR"
cd "$SCRIPTS_DIR"

echo "=================================================================="
echo "GENOMIC PREDICTION PREPARATION"
echo "Started at: $(date)"
echo "=================================================================="

# Check required files
echo "Checking required files..."
if [[ ! -f "$NEW_ANIMALS_FILE" ]]; then
    echo "❌ ERROR: New animals file not found: $NEW_ANIMALS_FILE"
    exit 1
fi

if [[ ! -d "$MODELS_DIR" ]]; then
    echo "❌ ERROR: Models directory not found: $MODELS_DIR"
    echo "Expected path: $MODELS_DIR"
    echo "Please check if models were saved to Phase2_Deployment_Prediction/deployment/models/"
    exit 1
fi

echo "✅ New animals file: $NEW_ANIMALS_FILE"
echo "✅ Models directory: $MODELS_DIR"

# Show available models
echo "Available trained traits:"
if [[ -d "$MODELS_DIR" ]]; then
    ls -1 "$MODELS_DIR" 2>/dev/null | head -10 || echo "  (No traits found)"
    TRAIT_COUNT=$(ls -1 "$MODELS_DIR" 2>/dev/null | wc -l)
    if [[ $TRAIT_COUNT -gt 10 ]]; then
        echo "... and $((TRAIT_COUNT - 10)) more"
    fi
    echo "Total traits: $TRAIT_COUNT"
else
    echo "  (Models directory not found)"
fi

# Check data file
if [[ -f "$NEW_ANIMALS_FILE" ]]; then
    ANIMALS_COUNT=$(wc -l < "$NEW_ANIMALS_FILE" 2>/dev/null)
    if [[ $ANIMALS_COUNT -gt 1 ]]; then
        echo "✅ Data file contains $((ANIMALS_COUNT - 1)) animals"
    else
        echo "❌ ERROR: Data file appears empty or invalid"
        exit 1
    fi
fi

# Prepare prediction jobs
echo "Preparing prediction array jobs..."
python3 04a_phase2_predict_array.py \
    --mode prepare \
    --new_X_file "$NEW_ANIMALS_FILE" \
    --models_dir "$MODELS_DIR" \
    --output_dir "$PREDICTION_ARRAY_DIR" \
    --results_dir "$RESULTS_DIR" \
    --chunk_size 50

PREP_STATUS=$?

if [[ $PREP_STATUS -eq 0 ]]; then
    echo "✅ Prediction preparation completed successfully"
    
    # Submit prediction array job
    ARRAY_SCRIPT="${PREDICTION_ARRAY_DIR}/genomic_prediction_array.sh"
    if [[ -f "$ARRAY_SCRIPT" ]]; then
        PRED_ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT")
        echo "Prediction array job ID: $PRED_ARRAY_JOB"
        
        # Submit results combination job with dependency
        COMBINE_SCRIPT="${PREDICTION_ARRAY_DIR}/combine_predictions.sh"
        if [[ -f "$COMBINE_SCRIPT" ]]; then
            # Update combine script to use correct paths and add dependency
            sed -i "1a\\#SBATCH --dependency=afterok:$PRED_ARRAY_JOB" "$COMBINE_SCRIPT"
            COMBINE_JOB=$(sbatch --parsable "$COMBINE_SCRIPT")
            echo "Results combination job ID: $COMBINE_JOB"
            
    echo ""
            echo "=================================================================="
            echo "PREDICTION PIPELINE JOBS SUBMITTED"
            echo "=================================================================="
            echo "  1. Prediction array: $PRED_ARRAY_JOB"
            echo "  2. Results combination: $COMBINE_JOB (after array completion)"
    echo ""
            echo "Monitor with: squeue -u $USER"
            echo "Results will be in: $RESULTS_DIR"
        fi
    fi
else
    echo "❌ ERROR: Prediction preparation failed"
    exit 1
fi

echo "=================================================================="
echo "Finished at: $(date)"
echo "=================================================================="
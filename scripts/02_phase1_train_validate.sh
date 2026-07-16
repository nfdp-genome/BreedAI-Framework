#!/bin/bash
#SBATCH --job-name=train_validate_prep
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/train_validate/train_validate_prep_%j.out
#SBATCH --error=../logs/train_validate/train_validate_prep_%j.err

# File: 02_phase1_train_validate.sh

# Accept pipeline mode: "default" = GBLUP only; anything else = all algorithms + ensembles
BREEDAI_PIPELINE_MODE="${1:-default_plus_rnd}"

echo "=================================================================="
echo "PHASE 1: TRAIN-VALIDATE PIPELINE"
echo "Mode: $BREEDAI_PIPELINE_MODE"
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

# The #SBATCH --output/--error directives at the top of this file use ../logs, a path
# SLURM resolves against the job's working directory before this script runs -- so they
# cannot use $PROJECT_DIR. Submitting from anywhere but scripts/ therefore writes the
# SLURM logs outside the repo. The run itself is unaffected, so warn rather than exit.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    ACTUAL_LOG_DIR="$(cd "$PWD/.." && pwd)/logs"
    if [[ "$ACTUAL_LOG_DIR" != "$PROJECT_DIR/logs" ]]; then
        echo "⚠️  WARNING: SLURM logs for this job are being written OUTSIDE the repo:"
        echo "      $ACTUAL_LOG_DIR"
        echo "    Expected: $PROJECT_DIR/logs"
        echo "    Submit from the repo with:  sbatch -D scripts/ $(basename "${BASH_SOURCE[0]}")"
        echo "    or use the documented path: cd scripts && ./start_menu.sh"
    fi
fi

SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/data"
INPUT_DIR="$PROJECT_DIR/input"
ARRAY_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking"
# Create subdirectories for organization
QC_DIR="$ARRAY_DIR/QC"
TRAINING_DIR="$ARRAY_DIR/training_validation"
# Results saved in respective subfolders
RESULTS_DIR="$ARRAY_DIR"

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
_ep=$(CONDA_BASE_PATH="$CONDA_BASE_PATH" ENVNAME="$USER_CONDA_ENV_PATH" timeout 45 bash -c 'source "$CONDA_BASE_PATH/etc/profile.d/conda.sh" 2>/dev/null; conda activate "$ENVNAME" >/dev/null 2>&1; printf %s "$CONDA_PREFIX"' 2>/dev/null || true)
if [ -n "$_ep" ] && [ -x "$_ep/bin/python" ]; then export CONDA_PREFIX="$_ep"; export PATH="$_ep/bin:$PATH"; export CONDA_DEFAULT_ENV="$USER_CONDA_ENV_PATH"; fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$CONDA_PREFIX" ] || ( [ "$CONDA_PREFIX" != "$USER_CONDA_ENV_PATH" ] && [ "$(basename "$CONDA_PREFIX")" != "$(basename "$USER_CONDA_ENV_PATH")" ] ); then
    echo "ERROR: Failed to activate Conda environment"
    exit 1
fi

# #region agent log
debug_log() {
    local hypothesis_id="$1"
    local location="$2"
    local message="$3"
    local data="$4"
    local ts
    ts=$(date +%s%3N)
    printf '{"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"%s","location":"%s","message":"%s","data":%s,"timestamp":%s}\n' \
        "$hypothesis_id" "$location" "$message" "$data" "$ts" >> "${BREEDAI_DEBUG_LOG:-/dev/null}"
}

debug_log "H1" "02_phase1_train_validate.sh:86" "Env before data check" \
  "$(printf '{"BGLR_RSCRIPT":"%s","BREEDAI_REQUIRE_ALL_ALGOS":"%s","CONDA_PREFIX":"%s","PATH_has_Rscript":"%s"}' \
    "${BGLR_RSCRIPT:-}" "${BREEDAI_REQUIRE_ALL_ALGOS:-}" "${CONDA_PREFIX:-}" "$(command -v Rscript || true)")"
# #endregion agent log

# Create directories
mkdir -p $ARRAY_DIR $QC_DIR $TRAINING_DIR "$PROJECT_DIR/logs/train_validate"
cd $SCRIPTS_DIR

# Data files - check input/ first, then data/
X_FILE=""
Y_FILE=""

if [[ -f "$INPUT_DIR/Geno.csv" ]] && [[ -f "$INPUT_DIR/Pheno.csv" ]]; then
    X_FILE="$INPUT_DIR/Geno.csv"
    Y_FILE="$INPUT_DIR/Pheno.csv"
    echo "✓ Data files found in input/"
elif [[ -f "$DATA_DIR/Geno.csv" ]] && [[ -f "$DATA_DIR/Pheno.csv" ]]; then
    X_FILE="$DATA_DIR/Geno.csv"
    Y_FILE="$DATA_DIR/Pheno.csv"
    echo "✓ Data files found in data/ (fallback)"
else
    echo "❌ ERROR: Data files not found!"
    echo "  Checked: $INPUT_DIR/Geno.csv"
    echo "  Checked: $DATA_DIR/Geno.csv"
    echo ""
    echo "Please ensure Geno.csv and Pheno.csv exist in:"
    echo "  - input/ (preferred location)"
    echo "  - data/ (fallback)"
    exit 1
fi

echo "Preparing train-validate array jobs..."
echo "Data files: X=$X_FILE, Y=$Y_FILE"

# #region agent log
debug_log "H2" "02_phase1_train_validate.sh:118" "Launching phase1 prepare" \
  "$(printf '{"X_FILE":"%s","Y_FILE":"%s","SCRIPTS_DIR":"%s"}' "$X_FILE" "$Y_FILE" "$SCRIPTS_DIR")"
# #endregion agent log

# Build algorithm filter based on pipeline mode
ALGO_ARGS=""
ENSEMBLE_ARGS=""
if [[ "$BREEDAI_PIPELINE_MODE" == "default" ]]; then
    ALGO_ARGS="--algorithms GBLUP_Ridge GBLUP_RidgeCV"
    echo "Running default mode: GBLUP only"
    # Default track is pure Python (GBLUP) — never needs R/BGLR. Skip the R probe
    # so a slow/hanging R install can't stall the run.
    export BREEDAI_SKIP_BGLR_CHECK=1
else
    ENSEMBLE_ARGS="--ensemble simple_average median weighted_average stacking_nonneg_ridge"
    echo "Running default + R&D mode: all algorithms + ensembles"
fi

# Skip unavailable optional backends (e.g. R+BGLR, gpflow) instead of crashing
export BREEDAI_REQUIRE_ALL_ALGOS=0

# Prepare array jobs (with automatic G-matrix calculation)
# Split: 60% train, 20% validate, 20% test
python 02a_phase1_train_validate_array.py \
    --mode prepare \
    --X_file "$X_FILE" \
    --y_file "$Y_FILE" \
    --output_dir "$TRAINING_DIR" \
    --qc_dir "$QC_DIR" \
    --test_size 0.2 \
    --val_size 0.2 \
    --random_state 42 \
    --calculate_gmatrix \
    $ALGO_ARGS \
    $ENSEMBLE_ARGS

PREP_STATUS=$?

if [[ $PREP_STATUS -eq 0 ]]; then
    echo "✅ Array preparation successful"
    # #region agent log
    debug_log "H3" "02_phase1_train_validate.sh:140" "Prepare status" \
      "$(printf '{"PREP_STATUS":%s}' "$PREP_STATUS")"
    # #endregion agent log
    
    # Submit benchmarking array job
    ARRAY_SCRIPT="$TRAINING_DIR/genomic_train_validate_array.sh"
    if [[ -f "$ARRAY_SCRIPT" ]]; then
        BENCH_ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT")
        echo "Train-validate array job ID: $BENCH_ARRAY_JOB"

        # Submit results combination job (use afterany so combine runs even if some array tasks failed)
        COMBINE_SCRIPT="$TRAINING_DIR/genomic_combine.sh"
        if [[ -f "$COMBINE_SCRIPT" ]]; then
            sed -i '/^#SBATCH --dependency=/d' "$COMBINE_SCRIPT"
            sed -i "1a\\#SBATCH --dependency=afterany:$BENCH_ARRAY_JOB" "$COMBINE_SCRIPT"
            COMBINE_JOB=$(sbatch --parsable "$COMBINE_SCRIPT")
            echo "Results combination job ID: $COMBINE_JOB"
            
            echo ""
            echo "Train-validate pipeline jobs submitted:"
            echo "  Array job: $BENCH_ARRAY_JOB"
            echo "  Combine job: $COMBINE_JOB"
            echo ""
            echo "Monitor with: squeue -u $USER"
            
            # Output final job ID to file for pipeline orchestration
            # This is the job that must complete before Phase 2 can start
            FINAL_JOB_FILE="$TRAINING_DIR/final_job_id.txt"
            echo "$COMBINE_JOB" > "$FINAL_JOB_FILE"
            echo "FINAL_DEPENDENCY_JOB=$COMBINE_JOB"
        fi
    fi
else
    echo "❌ Array preparation failed"
    exit 1
fi

echo "=================================================================="
echo "Completed at: $(date)"
echo "=================================================================="
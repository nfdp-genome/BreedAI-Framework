#!/bin/bash
#SBATCH --job-name=predict_unified
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output=../logs/prediction/predict_unified_%j.out
#SBATCH --error=../logs/prediction/predict_unified_%j.err

# File: 05_phase2_predict_unified.sh
# Purpose: Unified prediction script that:
#   1. Checks if models exist
#   2. Deploys models if needed (Step 2)
#   3. Predicts on new data (Step 3)

echo "=================================================================="
echo "UNIFIED PREDICTION PIPELINE (Deploy + Predict)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=================================================================="

# Project paths - automatically detect BreedAI-Framework directory
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

if [[ -z "$PROJECT_DIR" ]] && [[ -f "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [[ -n "$SCRIPT_DIR" ]] && [[ "$SCRIPT_DIR" != "/var/spool/slurm"* ]]; then
        PROJECT_DIR="$(find_project_dir "$SCRIPT_DIR")"
    fi
fi

if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$(find_project_dir "$PWD")"
fi

if [[ -z "$PROJECT_DIR" ]]; then
    echo "ERROR: Could not find BreedAI-Framework directory"
    exit 1
fi

SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/input"
MODELS_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/deployment/models"
DEPLOYMENT_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/deployment"
PREDICTION_DIR="$PROJECT_DIR/Phase2_Deployment_Prediction/prediction"

# Get new animals file from environment variable or use default
NEW_ANIMALS_FILE="${NEW_X_FILE:-$DATA_DIR/Geno.csv}"

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

mkdir -p "$PROJECT_DIR/logs/prediction" "$PREDICTION_DIR"
cd "$SCRIPTS_DIR"

echo "Configuration:"
echo "  New animals file: $NEW_ANIMALS_FILE"
echo "  Models directory: $MODELS_DIR"
echo "  Deployment directory: $DEPLOYMENT_DIR"

# Check if new animals file exists
if [[ ! -f "$NEW_ANIMALS_FILE" ]]; then
    echo "❌ ERROR: New animals file not found: $NEW_ANIMALS_FILE"
    exit 1
fi

# STEP 1: Check if models exist
echo ""
echo "=================================================================="
echo "STEP 1: Checking for deployed models..."
echo "=================================================================="

MODELS_EXIST=false
if [[ -d "$MODELS_DIR" ]]; then
    TRAIT_COUNT=$(ls -1 "$MODELS_DIR" 2>/dev/null | wc -l)
    if [[ $TRAIT_COUNT -gt 0 ]]; then
        echo "✅ Found $TRAIT_COUNT trained trait models in $MODELS_DIR"
        MODELS_EXIST=true
    else
        echo "⚠️  Models directory exists but is empty"
    fi
else
    echo "⚠️  Models directory does not exist"
fi

# STEP 2: Deploy models if needed
if [[ "$MODELS_EXIST" == false ]]; then
    echo ""
    echo "=================================================================="
    echo "STEP 2: Deploying models (models not found)..."
    echo "=================================================================="
    
    # Check for train-validate results (check both possible locations)
    BENCHMARK_RESULTS=""
    # Primary location: Phase1_Learning_Benchmarking/training_validation/
    BENCHMARK_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation"
    if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
        BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
        echo "Found train-validate results: $BENCHMARK_RESULTS"
    else
        # Fallback location: train_validate_array/
        BENCHMARK_DIR="$PROJECT_DIR/train_validate_array"
        if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
            BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
            echo "Found train-validate results (fallback): $BENCHMARK_RESULTS"
        else
            echo "❌ ERROR: No train-validate results found. Please run Phase 1 first."
            echo "Checked locations:"
            echo "  - $PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/"
            echo "  - $PROJECT_DIR/train_validate_array/"
            exit 1
        fi
    fi
    
    # Check for data files
    # For deployment (model training), we need the TRAINING genotype + phenotype files
    # The new animals file is used only for prediction (Step 3), not for deployment
    X_FILE="$DATA_DIR/Geno.csv"   # Training genotype file for model deployment
    Y_FILE="$DATA_DIR/Pheno.csv"  # Training phenotype file for model deployment
    
    if [[ ! -f "$X_FILE" ]]; then
        echo "❌ ERROR: Training genotype file not found: $X_FILE"
        echo "   Deployment requires the original training Geno.csv to train models."
        exit 1
    fi
    
    if [[ ! -f "$Y_FILE" ]]; then
        echo "❌ ERROR: Phenotype file not found for deployment: $Y_FILE"
        exit 1
    fi
    
    echo "Preparing deployment..."
    echo "  Using training genotype file: $X_FILE (for model deployment)"
    echo "  Using phenotype file: $Y_FILE"
    echo "  New animals file for prediction: $NEW_ANIMALS_FILE (will be used in Step 3)"
    mkdir -p "$DEPLOYMENT_DIR" "$PROJECT_DIR/logs/deployment"
    
    # Save new animals file path for use in reports
    echo "$NEW_ANIMALS_FILE" > "$DEPLOYMENT_DIR/input_file_path.txt"
    
    # Prepare deployment array jobs
    # Deployment uses training data (Geno.csv + Pheno.csv) to train models on full dataset
    # calculate_gmatrix is not included (defaults to False)
    echo "Running deployment preparation..."
    echo "  X_file (training data): $X_FILE"
    echo "  Y_file (training data): $Y_FILE"
    echo "  Output dir: $DEPLOYMENT_DIR"
    echo "  Python: $(which python3)"
    echo "  Python version: $(python3 --version)"
    echo "  Conda env: $CONDA_PREFIX"
    
    # Run deployment preparation and capture both stdout and stderr
    python3 03a_phase2_deploy_array.py \
        --mode prepare \
        --X_file "$X_FILE" \
        --y_file "$Y_FILE" \
        --output_dir "$DEPLOYMENT_DIR" \
        --benchmark_results "$BENCHMARK_RESULTS" \
        --use_full_dataset \
        --random_state 42 \
        --n_cv_folds 0 2>&1 | tee "$PROJECT_DIR/logs/deployment/deployment_prep_${SLURM_JOB_ID}.log"
    
    DEPLOY_PREP_STATUS=${PIPESTATUS[0]}
    
    if [[ $DEPLOY_PREP_STATUS -ne 0 ]]; then
        echo "❌ Deployment preparation failed with exit code: $DEPLOY_PREP_STATUS"
        echo "Check log file: $PROJECT_DIR/logs/deployment/deployment_prep_${SLURM_JOB_ID}.log"
        echo "Check Python environment and dependencies"
        exit 1
    fi
    
    if [[ $DEPLOY_PREP_STATUS -eq 0 ]]; then
        echo "✅ Deployment preparation successful"
        
        # Create Phase 2 notebook directory so reports have a place to go
        mkdir -p "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction"
        echo "📁 Phase 2 reports will be in: $PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/"
        
        # Submit deployment array job
        DEPLOY_ARRAY_SCRIPT="$DEPLOYMENT_DIR/genomic_deployment_array.sh"
        COMBINE_SCRIPT="$DEPLOYMENT_DIR/combine_models.sh"
        
        if [[ -f "$DEPLOY_ARRAY_SCRIPT" ]]; then
            DEPLOY_ARRAY_JOB=$(sbatch --parsable "$DEPLOY_ARRAY_SCRIPT")
            echo "Deployment array job ID: $DEPLOY_ARRAY_JOB"
            
            # Submit combine job with dependency on array job (export PROJECT_DIR so notebook path is correct)
            # Use afterany so combine runs even if some array tasks failed (still organizes saved models + report)
            COMBINE_JOB=""
            if [[ -f "$COMBINE_SCRIPT" ]]; then
                # Remove any existing dependency line so we don't accumulate or use stale job ID
                sed -i '/^#SBATCH --dependency=/d' "$COMBINE_SCRIPT"
                sed -i "1a\\#SBATCH --dependency=afterany:$DEPLOY_ARRAY_JOB" "$COMBINE_SCRIPT"
                COMBINE_JOB=$(sbatch --parsable --export=ALL,PROJECT_DIR="$PROJECT_DIR" "$COMBINE_SCRIPT")
                echo "Deployment combine job ID: $COMBINE_JOB"
                
            # The final job that must complete is the combine job (which generates the report)
            DEPLOYMENT_FINAL_JOB=$COMBINE_JOB
        else
            echo "⚠️  Combine script not found, using array job as final"
            DEPLOYMENT_FINAL_JOB=$DEPLOY_ARRAY_JOB
        fi
        
        # Define deployment report path
        DEPLOYMENT_REPORT="$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb"
        
        echo "Waiting for deployment to complete (final job: $DEPLOYMENT_FINAL_JOB)..."
        echo "Deployment report will be generated at: $DEPLOYMENT_REPORT"
        echo "Monitor with: squeue -j $DEPLOYMENT_FINAL_JOB"
        
        # Store the final job ID for prediction dependency
        DEPLOYMENT_JOB_ID=$DEPLOYMENT_FINAL_JOB
        else
            echo "❌ ERROR: Deployment array script not found"
            exit 1
        fi
    else
        echo "❌ ERROR: Deployment preparation failed"
        exit 1
    fi
else
    echo "✅ Models already exist, skipping deployment"
    DEPLOYMENT_JOB_ID=""
    # Ensure Phase 2 notebook directory and 2.2 report exist (e.g. from a previous run that didn't run combine)
    mkdir -p "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction"
    if [[ ! -f "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb" ]]; then
        echo "Generating deployment report notebook (2.2) since it was missing..."
        python3 03b_phase2_report_deployment.py \
            --models_dir "$MODELS_DIR" \
            --deployment_dir "$DEPLOYMENT_DIR" \
            --output_file "$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb" 2>/dev/null || true
    fi
fi

# STEP 3: Predict on new data
echo ""
echo "=================================================================="
echo "STEP 3: Predicting on new data..."
echo "=================================================================="

# Define deployment report path
DEPLOYMENT_REPORT="$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb"

# Verify models exist and deployment report is ready before prediction
if [[ ! -d "$MODELS_DIR" ]] || [[ $(ls -1 "$MODELS_DIR" 2>/dev/null | wc -l) -eq 0 ]]; then
    if [[ -n "$DEPLOYMENT_JOB_ID" ]]; then
        echo "⚠️  Models not ready yet. Prediction will be submitted as dependent job."
        echo "   Waiting for deployment job $DEPLOYMENT_JOB_ID to complete..."
        echo "   Deployment report will be checked at: $DEPLOYMENT_REPORT"
        
        # Create a wrapper script that waits for deployment and then prepares prediction
        PRED_PREP_SCRIPT="$PREDICTION_DIR/prepare_prediction_after_deploy.sh"
        cat > "$PRED_PREP_SCRIPT" << 'PRED_PREP_EOF'
#!/bin/bash
#SBATCH --job-name=pred_prep
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --dependency=afterok:__DEPLOYMENT_JOB_ID__
#SBATCH --output=__PROJECT_DIR__/logs/prediction/pred_prep_%j.out
#SBATCH --error=__PROJECT_DIR__/logs/prediction/pred_prep_%j.err

# Project paths
PROJECT_DIR="__PROJECT_DIR__"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/input"
MODELS_DIR="__MODELS_DIR__"
NEW_ANIMALS_FILE="__NEW_ANIMALS_FILE__"
PREDICTION_DIR="__PREDICTION_DIR__"
TRAINING_X_FILE="$DATA_DIR/Geno.csv"

# Wait a bit to ensure models are fully written
sleep 5

# Verify models exist
if [[ ! -d "$MODELS_DIR" ]] || [[ $(ls -1 "$MODELS_DIR" 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "❌ ERROR: Models still not available after deployment"
    echo "Models directory: $MODELS_DIR"
        exit 1
    fi

# Verify deployment report exists
DEPLOYMENT_REPORT="$PROJECT_DIR/notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb"
if [[ ! -f "$DEPLOYMENT_REPORT" ]]; then
    echo "⚠️  WARNING: Deployment report not found at $DEPLOYMENT_REPORT"
    echo "   Continuing with prediction anyway..."
else
    echo "✅ Deployment report found: $DEPLOYMENT_REPORT"
fi

echo "✅ Models found, preparing prediction jobs..."
cd "$SCRIPTS_DIR"

# Prepare prediction jobs
# Use training dataset for variance calculation (like test phase does)
TRAINING_X_FILE="$DATA_DIR/Geno.csv"
python3 04a_phase2_predict_array.py \
    --mode prepare \
    --new_X_file "$NEW_ANIMALS_FILE" \
    --models_dir "$MODELS_DIR" \
    --training_X_file "$TRAINING_X_FILE" \
    --output_dir "$PREDICTION_DIR" \
    --results_dir "$PREDICTION_DIR" \
    --chunk_size 50

PREP_STATUS=$?
    
if [[ $PREP_STATUS -eq 0 ]]; then
    echo "✅ Prediction preparation completed"
    # Submit prediction array job
    ARRAY_SCRIPT="$PREDICTION_DIR/genomic_prediction_array.sh"
    if [[ -f "$ARRAY_SCRIPT" ]]; then
            PRED_ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT")
        echo "Prediction array job ID: $PRED_ARRAY_JOB"
        
        # Submit results combination job
        COMBINE_SCRIPT="$PREDICTION_DIR/combine_predictions.sh"
        if [[ -f "$COMBINE_SCRIPT" ]]; then
            sed -i "1a\\#SBATCH --dependency=afterok:$PRED_ARRAY_JOB" "$COMBINE_SCRIPT"
            COMBINE_JOB=$(sbatch --parsable "$COMBINE_SCRIPT")
            echo "Results combination job ID: $COMBINE_JOB"
        fi
    fi
else
    echo "❌ ERROR: Prediction preparation failed"
    exit 1
fi
PRED_PREP_EOF
        
        # Replace placeholders with actual values
        sed -i "s|__DEPLOYMENT_JOB_ID__|$DEPLOYMENT_JOB_ID|g" "$PRED_PREP_SCRIPT"
        sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PRED_PREP_SCRIPT"
        sed -i "s|__MODELS_DIR__|$MODELS_DIR|g" "$PRED_PREP_SCRIPT"
        sed -i "s|__NEW_ANIMALS_FILE__|$NEW_ANIMALS_FILE|g" "$PRED_PREP_SCRIPT"
        sed -i "s|__PREDICTION_DIR__|$PREDICTION_DIR|g" "$PRED_PREP_SCRIPT"
        # Note: TRAINING_X_FILE is already set in the script template
        
        chmod +x "$PRED_PREP_SCRIPT"
        
        # Submit the preparation job
        PRED_PREP_JOB=$(sbatch --parsable "$PRED_PREP_SCRIPT")
        echo "Prediction preparation job ID: $PRED_PREP_JOB (depends on deployment $DEPLOYMENT_JOB_ID)"
            
            echo ""
            echo "=================================================================="
            echo "PREDICTION PIPELINE SUBMITTED"
            echo "=================================================================="
                echo "  1. Deployment array: $DEPLOY_ARRAY_JOB"
                if [[ -n "$COMBINE_JOB" ]]; then
                    echo "  2. Deployment combine: $COMBINE_JOB"
                    echo "  3. Prediction preparation: $PRED_PREP_JOB (depends on deployment combine)"
                else
                    echo "  2. Prediction preparation: $PRED_PREP_JOB (depends on deployment array)"
                fi
                echo "  3. Prediction array: Will be submitted by preparation job"
                echo ""
                echo "Monitor with: squeue -u $USER"
                echo "Results will be in: $PREDICTION_DIR"
        
    else
        echo "❌ ERROR: No models available and no deployment job ID"
        exit 1
    fi
else
    # Models exist, check if deployment report exists
    if [[ -f "$DEPLOYMENT_REPORT" ]]; then
        echo "✅ Models found and deployment report exists"
    else
        echo "✅ Models found, but deployment report not found (this is OK if deployment was done previously)"
    fi
    
    echo "Preparing prediction jobs..."
    echo "  New animals file: $NEW_ANIMALS_FILE"
    echo "  Models directory: $MODELS_DIR"
    echo "  Training file (for variance): $DATA_DIR/Geno.csv"
    echo "  Output directory: $PREDICTION_DIR"
    
    # Use training dataset for variance calculation (like test phase does)
    TRAINING_X_FILE="$DATA_DIR/Geno.csv"
    
    # Run prediction preparation and capture output
    python3 04a_phase2_predict_array.py \
        --mode prepare \
        --new_X_file "$NEW_ANIMALS_FILE" \
        --models_dir "$MODELS_DIR" \
        --training_X_file "$TRAINING_X_FILE" \
        --output_dir "$PREDICTION_DIR" \
        --results_dir "$PREDICTION_DIR" \
        --chunk_size 50 2>&1 | tee "$PROJECT_DIR/logs/prediction/pred_prep_${SLURM_JOB_ID:-manual}.log"
    
    PRED_PREP_STATUS=${PIPESTATUS[0]}
    
    if [[ $PRED_PREP_STATUS -eq 0 ]]; then
        echo "✅ Prediction preparation completed successfully"
        
        # Submit prediction array job
        ARRAY_SCRIPT="${PREDICTION_DIR}/genomic_prediction_array.sh"
        if [[ -f "$ARRAY_SCRIPT" ]]; then
            echo "Submitting prediction array job..."
            PRED_ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT" 2>&1)
            ARRAY_SUBMIT_EXIT=$?
            
            if [[ $ARRAY_SUBMIT_EXIT -ne 0 ]] || [[ -z "$PRED_ARRAY_JOB" ]] || ! [[ "$PRED_ARRAY_JOB" =~ ^[0-9]+$ ]]; then
                echo "❌ ERROR: Failed to submit prediction array job"
                echo "sbatch output: $PRED_ARRAY_JOB"
                exit 1
            fi
            
            echo "✅ Prediction array job submitted: $PRED_ARRAY_JOB"
            
            # Submit results combination job with dependency
            COMBINE_SCRIPT="${PREDICTION_DIR}/combine_predictions.sh"
            if [[ -f "$COMBINE_SCRIPT" ]]; then
                echo "Submitting results combination job..."
                # Update combine script to add dependency
                sed -i "1a\\#SBATCH --dependency=afterok:$PRED_ARRAY_JOB" "$COMBINE_SCRIPT"
                COMBINE_JOB=$(sbatch --parsable "$COMBINE_SCRIPT" 2>&1)
                COMBINE_SUBMIT_EXIT=$?
                
                if [[ $COMBINE_SUBMIT_EXIT -ne 0 ]] || [[ -z "$COMBINE_JOB" ]] || ! [[ "$COMBINE_JOB" =~ ^[0-9]+$ ]]; then
                    echo "⚠️  WARNING: Failed to submit combine job, but prediction array is running"
                    echo "sbatch output: $COMBINE_JOB"
                else
                    echo "✅ Results combination job submitted: $COMBINE_JOB"
                fi
                
                echo ""
                echo "=================================================================="
                echo "PREDICTION PIPELINE SUBMITTED"
                echo "=================================================================="
                echo "  1. Prediction array: $PRED_ARRAY_JOB"
                if [[ -n "$COMBINE_JOB" ]] && [[ "$COMBINE_JOB" =~ ^[0-9]+$ ]]; then
                    echo "  2. Results combination: $COMBINE_JOB (depends on prediction)"
            fi
            echo ""
            echo "Monitor with: squeue -u $USER"
            echo "Results will be in: $PREDICTION_DIR"
            else
                echo "⚠️  WARNING: Combine script not found: $COMBINE_SCRIPT"
                echo "Prediction array job is running: $PRED_ARRAY_JOB"
    fi
else
            echo "❌ ERROR: Prediction array script not found: $ARRAY_SCRIPT"
            echo "Prediction preparation may have failed to generate the script"
    exit 1
        fi
    else
        echo "❌ ERROR: Prediction preparation failed with exit code: $PRED_PREP_STATUS"
        echo "Check log file: $PROJECT_DIR/logs/prediction/pred_prep_${SLURM_JOB_ID:-manual}.log"
        exit 1
    fi
fi

echo "=================================================================="
echo "Completed at: $(date)"
echo "=================================================================="


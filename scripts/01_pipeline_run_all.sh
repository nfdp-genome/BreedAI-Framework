#!/bin/bash
# File: 01_pipeline_run_all.sh
# Purpose: Pipeline execution engine (called by start_menu.sh)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (BreedAI-Framework)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
LOGS_DIR="$PROJECT_DIR/logs"

# Get the phase choice from command line argument
if [[ $# -eq 0 ]]; then
    echo "Error: 01_pipeline_run_all.sh requires a phase choice argument (1-4)"
    echo "Usage: ./01_pipeline_run_all.sh <option>"
    echo "Or use: ./start_menu.sh (for interactive menu)"
    exit 1
fi

phase_choice="$1"
BREEDAI_CLI="$PROJECT_DIR/src/breedai/cli.py"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

case $phase_choice in
    1)
        echo "=================================================================="
        echo "PHASE 1: MODEL DEVELOPMENT & BENCHMARKING (ALGORITHM SELECTION & FULL REPORT)"
        echo "=================================================================="
        echo ""
        echo "This step will:"
        echo "  - Split data into train (60%), validate (20%), and test (20%)"
        echo "  - Evaluate all algorithms on all three sets"
        echo "  - Generate comprehensive report notebook"
        echo "  - Save results to CSV files"
        echo ""
        read -p "Mode [default | default_plus_rnd] (default=default_plus_rnd): " run_mode
        run_mode="${run_mode:-default_plus_rnd}"
        read -p "Species (e.g., cattle): " species
        species="${species:-cattle}"
        read -p "Goal (e.g., growth or milk): " goal
        goal="${goal:-growth}"
        read -p "Input type [fastq|vcf|plink] (default=vcf): " input_type
        input_type="${input_type:-vcf}"

        if [[ "$run_mode" == "default" ]]; then
            if [[ ! -f "$BREEDAI_CLI" ]]; then
                echo "ERROR: BreedAI CLI not found at $BREEDAI_CLI"
                exit 1
            fi
            python3 "$BREEDAI_CLI" run \
                --phase 1 \
                --mode default \
                --species "$species" \
                --goal "$goal" \
                --input-type "$input_type" \
                --submit --slurm
            CLI_EXIT=$?
            if [[ $CLI_EXIT -ne 0 ]]; then
                echo "❌ ERROR: default pipeline submission failed"
                exit $CLI_EXIT
            fi
            echo "✅ Default pipeline submission created under $PROJECT_DIR/runs/"
        else
            if [[ ! -f "02_phase1_train_validate.sh" ]]; then
                echo "ERROR: 02_phase1_train_validate.sh not found!"
                exit 1
            fi
            PREP_JOB=$(sbatch --parsable --export=ALL 02_phase1_train_validate.sh)
            echo "Train-validate preparation job: $PREP_JOB"
            echo "Monitor with: squeue -j $PREP_JOB"
        fi
        echo ""
        echo "Results will be in: $PROJECT_DIR/Phase1_Learning_Benchmarking/"
        echo "  - QC files: $PROJECT_DIR/Phase1_Learning_Benchmarking/QC/"
        echo "  - Training results: $PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/"
        echo "Report notebook will be generated automatically after completion"
        ;;
    2)
        echo "=================================================================="
        echo "PHASE 2: PREDICTION DEPLOYMENT & INFERENCE (AUTO-DEPLOY IF NEEDED)"
        echo "=================================================================="
        echo ""
        echo "This step will:"
        echo "  - Check if models are already deployed"
        echo "  - Deploy models automatically if needed (using full dataset)"
        echo "  - Predict on new animals (without ground truth)"
        echo "  - Save predictions to CSV"
        echo ""
        read -p "Mode [default | default_plus_rnd] (default=default_plus_rnd): " run_mode
        run_mode="${run_mode:-default_plus_rnd}"
        read -p "Species (e.g., cattle): " species
        species="${species:-cattle}"
        read -p "Goal (e.g., growth or milk): " goal
        goal="${goal:-growth}"
        read -p "Input type [fastq|vcf|plink] (default=vcf): " input_type
        input_type="${input_type:-vcf}"

        if [[ "$run_mode" == "default" ]]; then
            python3 "$BREEDAI_CLI" run \
                --phase 2 \
                --mode default \
                --species "$species" \
                --goal "$goal" \
                --input-type "$input_type" \
                --submit --slurm
            CLI_EXIT=$?
            if [[ $CLI_EXIT -ne 0 ]]; then
                echo "❌ ERROR: default pipeline submission failed"
                exit $CLI_EXIT
            fi
            echo "✅ Default pipeline submission created under $PROJECT_DIR/runs/"
            exit 0
        fi

        # Check for default-track deployment option first
        DEFAULT_TRACK_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/default_track"
        if [[ -f "$DEFAULT_TRACK_DIR/model_card.json" ]] && [[ "$run_mode" == "default" ]]; then
            echo "Default track results found — deploying GBLUP model"
            python3 "$SCRIPTS_DIR/08_deploy_default_track.py"
            echo "✅ Default deployment complete"
            echo "Results: $PROJECT_DIR/Phase2_Deployment_Prediction/deployment/default_track/"
            exit 0
        fi

        # Existing R&D path for default_plus_rnd
        BENCHMARK_RESULTS=""
        BENCHMARK_DIR="$PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation"
        if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
            BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
        else
            BENCHMARK_DIR="$PROJECT_DIR/train_validate_array"
            if [[ -f "$BENCHMARK_DIR/combined_train_validate_results.csv" ]]; then
                BENCHMARK_RESULTS="$BENCHMARK_DIR/combined_train_validate_results.csv"
            fi
        fi
        if [[ -z "$BENCHMARK_RESULTS" ]]; then
            echo "❌ ERROR: Phase 1 results not found."
            echo "Please run Phase 1 first to generate benchmarking results."
            echo "Checked locations:"
            echo "  - $PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/"
            echo "  - $PROJECT_DIR/train_validate_array/"
            exit 1
        fi
        
        read -p "Enter new animals genotype file path (or press Enter for test mode): " new_file
        
        if [[ -z "$new_file" ]]; then
            # Use reference data for testing
            new_file="$PROJECT_DIR/dataset/input/Geno.csv"
            echo "Using reference data for test predictions: $new_file"
        fi
        
        if [[ ! -f "$new_file" ]]; then
            echo "❌ ERROR: File not found: $new_file"
            exit 1
        fi
        
        if [[ ! -f "$SCRIPTS_DIR/05_phase2_predict_unified.sh" ]]; then
            echo "ERROR: 05_phase2_predict_unified.sh not found at $SCRIPTS_DIR/05_phase2_predict_unified.sh"
            exit 1
        fi
        
        # Submit the job from scripts directory
        cd "$SCRIPTS_DIR"
        
        echo "Submitting prediction job..."
        echo "  Script: $SCRIPTS_DIR/05_phase2_predict_unified.sh"
        echo "  Input file: $new_file"
        
        PRED_JOB=$(sbatch --parsable --export=ALL,NEW_X_FILE="$new_file" "$SCRIPTS_DIR/05_phase2_predict_unified.sh" 2>&1)
        SBATCH_EXIT=$?
        
        # Check if submission was successful
        if [[ $SBATCH_EXIT -ne 0 ]]; then
            echo "❌ ERROR: sbatch command failed with exit code $SBATCH_EXIT"
            echo "sbatch output: $PRED_JOB"
            exit 1
        fi
        
        if [[ -z "$PRED_JOB" ]]; then
            echo "❌ ERROR: No job ID returned from sbatch"
            exit 1
        fi
        
        # Check if output looks like a job ID (numeric)
        if ! [[ "$PRED_JOB" =~ ^[0-9]+$ ]]; then
            echo "❌ ERROR: Invalid job ID returned: $PRED_JOB"
            echo "This might indicate an error from sbatch"
            exit 1
        fi
        
        echo "✅ Unified prediction job submitted successfully: $PRED_JOB"
        echo "Monitor with: squeue -j $PRED_JOB"
        echo ""
        echo "Predictions will be in: $PROJECT_DIR/Phase2_Deployment_Prediction/prediction/"
        ;;
    3)
        echo "=================================================================="
        echo "JOB STATUS CHECK"
        echo "=================================================================="
        echo ""
        squeue -u $USER
        ;;
    4)
        echo "=================================================================="
        echo "SETUP TEST AND JOB COUNT ESTIMATION"
        echo "=================================================================="

        # Check data files - check dataset/input/ first, then data/
        if [[ -f "$PROJECT_DIR/dataset/input/Geno.csv" ]] && [[ -f "$PROJECT_DIR/dataset/input/Pheno.csv" ]]; then
            DATA_DIR="$PROJECT_DIR/dataset/input"
        elif [[ -f "$PROJECT_DIR/data/Geno.csv" ]] && [[ -f "$PROJECT_DIR/data/Pheno.csv" ]]; then
            DATA_DIR="$PROJECT_DIR/data"
        else
            DATA_DIR="$PROJECT_DIR/data"
        fi

        X_FILE="$DATA_DIR/Geno.csv"
        Y_FILE="$DATA_DIR/Pheno.csv"

        if [[ -f "$Y_FILE" ]]; then
            N_TRAITS=$(python3 -c "import pandas as pd; df=pd.read_csv('$Y_FILE', index_col=0); print(len(df.columns))" 2>/dev/null)
            echo "Data found:"
            echo "  Genotype file: $X_FILE"
            echo "  Phenotype file: $Y_FILE"
            echo "  Number of traits: $N_TRAITS"
            echo ""
            echo "Expected job counts (ONE JOB PER TRAIT approach):"
            echo "  Train-validate jobs: $N_TRAITS (instead of ~518!)"
            echo "  Deployment jobs: $N_TRAITS"
            echo "  Prediction jobs: depends on chunk size (default: animals/50)"
        else
            echo "Data files not found in $DATA_DIR"
            echo "Please check your data file paths"
        fi

        echo ""
        echo "Required scripts status:"
        for script in 02_phase1_train_validate.sh 03_phase2_deploy_models.sh 04_phase2_predict.sh; do
            if [[ -f "$script" ]]; then
                echo "  ✅ $script"
            else
                echo "  ❌ $script (missing)"
            fi
        done
        ;;
    5)
        echo "=================================================================="
        echo "PHASE 1 ORCHESTRATOR (same as menu Phase 1 / public benchmark reproduction)"
        echo "=================================================================="
        echo ""
        echo "Requires: $PROJECT_DIR/dataset/input/Geno.csv and Pheno.csv"
        echo "Copy from dataset/public_datasets/cattle/processed/... (see cattle README)."
        echo ""
        if [[ ! -f "$PROJECT_DIR/dataset/input/Geno.csv" ]] || [[ ! -f "$PROJECT_DIR/dataset/input/Pheno.csv" ]]; then
            echo "ERROR: Missing Geno.csv or Pheno.csv under dataset/input/"
            exit 1
        fi
        read -p "Mode [default | default_plus_rnd] (default=default_plus_rnd): " poc_mode
        poc_mode="${poc_mode:-default_plus_rnd}"

        if command -v sbatch &>/dev/null; then
            mkdir -p "$PROJECT_DIR/logs/poc"
            POC_JOB=$(sbatch --parsable --export=ALL "$SCRIPTS_DIR/07_run_poc.sh" "$poc_mode")
            echo "✅ Job submitted: $POC_JOB"
            echo "Monitor with: squeue -j $POC_JOB"
        else
            echo "Running interactively (no SLURM detected)..."
            python3 "$SCRIPTS_DIR/07_run_poc.py" --mode "$poc_mode"
        fi
        echo ""
        echo "Results will be in:"
        echo "  Core dataset: $PROJECT_DIR/Phase1_Learning_Benchmarking/core_dataset/"
        echo "  Default track: $PROJECT_DIR/Phase1_Learning_Benchmarking/training_validation/default_track/"
        ;;
    6)
        echo "=================================================================="
        echo "BUILD CORE DATASET FROM CONFIG"
        echo "=================================================================="
        echo ""
        read -p "Config YAML path (relative to project root): " config_path
        if [[ -z "$config_path" ]]; then
            echo "Using default: configs/run_configs/cattle_vandenberg_default.yaml"
            config_path="configs/run_configs/cattle_vandenberg_default.yaml"
        fi
        python3 "$SCRIPTS_DIR/05_build_core_dataset.py" --config "$config_path"
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=================================================================="
echo "MONITORING COMMANDS"
echo "=================================================================="
echo "  Check all jobs:     squeue -u $USER"
echo "  Cancel all jobs:    scancel -u $USER"
echo "  View recent logs:   find $LOGS_DIR -name '*.out' -mtime -1 -exec tail -5 {} \\;"
echo "  Model Development results: $PROJECT_DIR/Phase1_Learning_Benchmarking"
echo "  Prediction results:   $PROJECT_DIR/Phase2_Deployment_Prediction/prediction"
echo "=================================================================="
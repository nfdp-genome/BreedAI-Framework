#!/bin/bash
#SBATCH --dependency=afterany:48474826
#SBATCH --job-name=genomic_combine
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output=$PROJECT_DIR/logs/train_validate/genomic_combine_%j.out
#SBATCH --error=$PROJECT_DIR/logs/train_validate/genomic_combine_%j.err

echo "=================================================================="
echo "GENOMIC TRAIN-VALIDATE RESULTS COMBINATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=================================================================="

mkdir -p logs

# Environment setup
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; [ -n "$CONDA_BASE_PATH" ] || CONDA_BASE_PATH="$(timeout 20 conda info --base 2>/dev/null || true)"

module purge 2>/dev/null || true
module load R 2>/dev/null || true
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "$USER_CONDA_ENV_PATH" || true
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$CONDA_PREFIX" ] || ( [ "$CONDA_PREFIX" != "$USER_CONDA_ENV_PATH" ] && [ "$(basename "$CONDA_PREFIX")" != "$(basename "$USER_CONDA_ENV_PATH")" ] ); then
    echo "ERROR: Failed to activate Conda environment"
    exit 1
fi

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

echo "Combining results from 4 trait jobs..."

python3 02a_phase1_train_validate_array.py \
    --mode combine_results \
    --n_jobs 4 \
    --output_dir /ibex/project/c2293/BreedAI_Poster/BreedAI-public/Phase1_Learning_Benchmarking/training_validation

COMBINE_EXIT_CODE=$?

if [[ $COMBINE_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Generating comprehensive reports..."

    # Create Phase 1 folder for results
    mkdir -p "$PROJECT_DIR/notebooks/Phase_1_Learning_Benchmarking"

    # Generate Jupyter notebook report
    python3 02b_phase1_report_benchmarking.py \
        --results_file /ibex/project/c2293/BreedAI_Poster/BreedAI-public/Phase1_Learning_Benchmarking/training_validation/combined_train_validate_results.csv \
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
    
    python3 02c_phase1_report_preprocessing.py \
        --dataset_dir "$DATASET_DIR" \
        --gmatrix_dir "$GMATRIX_DIR" \
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

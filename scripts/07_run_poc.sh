#!/bin/bash
#SBATCH --job-name=breedai_poc
#SBATCH --partition=batch
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=../logs/poc/poc_%j.out
#SBATCH --error=../logs/poc/poc_%j.err

# Phase 1 orchestrator (requires dataset/input/Geno.csv + Pheno.csv)
# Submit: sbatch scripts/07_run_poc.sh [default|default_plus_rnd]
# Or run interactively: bash scripts/07_run_poc.sh

echo "=================================================================="
echo "BREEDAI PHASE 1 (dataset/input)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Started at: $(date)"
echo "=================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Environment
USER_CONDA_ENV_PATH="genomic_pred"
CONDA_BASE_PATH="$(conda info --base 2>/dev/null)"

if command -v module &>/dev/null; then module purge; fi
[ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$USER_CONDA_ENV_PATH" 2>/dev/null || true

mkdir -p "$PROJECT_DIR/logs/poc" "$PROJECT_DIR/logs/train_validate"

cd "$SCRIPT_DIR"

MODE="${1:-default_plus_rnd}"

python 07_run_poc.py --mode "$MODE"

echo "=================================================================="
echo "Completed at: $(date)"
echo "=================================================================="

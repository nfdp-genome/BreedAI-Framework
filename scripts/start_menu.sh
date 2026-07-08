#!/usr/bin/env bash
set -euo pipefail

# =========================
# BreedAI menu configuration
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${REPO_ROOT}/logs"
PHASE1_DIR="${REPO_ROOT}/Phase1_Learning_Benchmarking"
PHASE2_DIR="${REPO_ROOT}/Phase2_Deployment_Prediction"

mkdir -p "${LOG_DIR}"

# Adjust this if your conda env path changes
DEFAULT_CONDA_ENV="genomic_pred"

# Default SLURM resources (generous buffer for long / memory-heavy runs; press Enter to accept)
# Override by exporting before running the menu, e.g.: export SBATCH_DEFAULT_MEM=256G
SBATCH_DEFAULT_PARTITION="${SBATCH_DEFAULT_PARTITION:-batch}"
SBATCH_DEFAULT_CPUS="${SBATCH_DEFAULT_CPUS:-64}"
SBATCH_DEFAULT_MEM="${SBATCH_DEFAULT_MEM:-512G}"
SBATCH_DEFAULT_TIME="${SBATCH_DEFAULT_TIME:-7-00:00:00}"

# Your SLURM account/allocation. sbatch reads SBATCH_ACCOUNT natively, so setting it
# here (or exporting it before running: export SBATCH_ACCOUNT=<your-account>) applies to
# every job the menu submits. Leave empty to use your cluster's default account.
# Find yours with: sacctmgr show assoc user=$USER format=account
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
export SBATCH_ACCOUNT

# =========================
# Helper functions
# =========================
print_header() {
  echo "=================================================================="
  echo "BREEDAI GENOMIC PREDICTION PIPELINE"
  echo "=================================================================="
  echo "Repository : ${REPO_ROOT}"
  echo "Logs       : ${LOG_DIR}"
  echo "Phase 1    : ${PHASE1_DIR}"
  echo "Phase 2    : ${PHASE2_DIR}"
  echo "=================================================================="
  echo
}

pause_line() {
  echo
  read -r -p "Press Enter to continue..." _
}

print_paths_summary() {
  echo
  echo "Output locations:"
  echo "  Logs                     : ${LOG_DIR}"
  echo "  Phase 1 outputs          : ${PHASE1_DIR}"
  echo "  Phase 2 outputs          : ${PHASE2_DIR}"
  echo
}

load_conda_hint() {
  cat <<EOF

Before running BreedAI, activate the recommended environment if needed:
  conda activate ${DEFAULT_CONDA_ENV}

If conda is not already initialized in your shell, run:
  source ~/miniconda3/etc/profile.d/conda.sh

EOF
}

submit_with_sbatch() {
  local job_name="$1"
  local command="$2"
  local partition="$3"
  local cpus="$4"
  local mem="$5"
  local time_limit="$6"

  local sbatch_file
  # Write the generated sbatch script into logs/ (gitignored), NOT into scripts/,
  # so it never lands in the tracked tree — it contains SBATCH_ACCOUNT and other
  # run-specific values that should stay local.
  sbatch_file="$(mktemp "${LOG_DIR}/.${job_name}.XXXXXX.sbatch")"

  cat > "${sbatch_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err

set -euo pipefail

cd "${SCRIPT_DIR}"

module purge
module load R
CONDA_BASE_PATH="$(conda info --base 2>/dev/null)"
if [ -f "\${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
  source "\${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
fi

conda activate "${DEFAULT_CONDA_ENV}" || true

export NEW_X_FILE="${NEW_X_FILE:-}"
export BREEDAI_REQUIRE_ALL_ALGOS=0
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT}"

${command}
EOF

  echo
  echo "Submitting job with sbatch..."
  local submit_output
  submit_output="$(sbatch "${sbatch_file}")"
  echo "${submit_output}"
  echo "SBATCH script: ${sbatch_file}"
  print_paths_summary
}

run_locally() {
  local command="$1"
  echo
  echo "Running locally:"
  echo "  ${command}"
  echo
  cd "${SCRIPT_DIR}"
  eval "${command}"
}

ask_execution_mode() {
  # Sets the global EXEC_MODE variable (not captured via subshell to avoid stdin issues).
  echo
  echo "Where to run this job?"
  echo "  1) SLURM — submit with sbatch (recommended on HPC; runs in background)"
  echo "  2) Local — run now in this shell (blocks until finished; good for quick tests)"
  echo
  local run_where
  read -r -p "SLURM or local [1-2] (default=1): " run_where
  case "${run_where:-1}" in
    1) EXEC_MODE="sbatch" ;;
    2) EXEC_MODE="local" ;;
    *) EXEC_MODE="sbatch" ;;
  esac
}

ask_sbatch_resources() {
  local default_partition="${1:-${SBATCH_DEFAULT_PARTITION}}"
  local default_cpus="${2:-${SBATCH_DEFAULT_CPUS}}"
  local default_mem="${3:-${SBATCH_DEFAULT_MEM}}"
  local default_time="${4:-${SBATCH_DEFAULT_TIME}}"

  echo >&2
  echo >&2 "SLURM resource defaults (includes buffer for long runs; press Enter to keep each):"
  echo >&2 "  partition=${default_partition}  cpus=${default_cpus}  mem=${default_mem}  time=${default_time}"
  echo >&2

  read -r -p "SLURM partition [${default_partition}]: " SLURM_PARTITION
  read -r -p "CPUs per task [${default_cpus}]: " SLURM_CPUS
  read -r -p "Memory [${default_mem}]: " SLURM_MEM
  read -r -p "Time limit (D-HH:MM:SS) [${default_time}]: " SLURM_TIME
  read -r -p "SLURM account (Enter = ${SBATCH_ACCOUNT:-cluster default}): " _acct_in

  SLURM_PARTITION="${SLURM_PARTITION:-${default_partition}}"
  SLURM_CPUS="${SLURM_CPUS:-${default_cpus}}"
  SLURM_MEM="${SLURM_MEM:-${default_mem}}"
  SLURM_TIME="${SLURM_TIME:-${default_time}}"
  # sbatch reads SBATCH_ACCOUNT from the environment; exporting it here applies to the
  # menu's submission and to any array jobs the pipeline submits downstream.
  SBATCH_ACCOUNT="${_acct_in:-${SBATCH_ACCOUNT}}"
  export SBATCH_ACCOUNT
}

run_phase1() {
  echo
  echo "You selected: Phase 1 — Learning & Benchmarking"
  echo "Train and benchmark models on a dataset."
  echo
  echo "Standard inputs must be present under:"
  echo "  ${REPO_ROOT}/input/Geno.csv"
  echo "  ${REPO_ROOT}/input/Pheno.csv"
  echo
  echo "Copy benchmark files from cattle_dataset/processed/ when reproducing a public run"
  echo "(see cattle_dataset/README.md for the cattle benchmark)."
  echo

  if [[ ! -f "${REPO_ROOT}/input/Geno.csv" ]] || [[ ! -f "${REPO_ROOT}/input/Pheno.csv" ]]; then
    echo "ERROR: Geno.csv and/or Pheno.csv missing under ${REPO_ROOT}/input/"
    echo "Add both files before running Phase 1."
    return
  fi

  echo "Species:"
  echo "  1) cattle"
  echo "  2) sheep"
  echo "  3) goat"
  echo "  4) camel"
  echo "  5) horse"
  echo
  read -r -p "Species [1-5] (default=1): " species_choice
  case "${species_choice:-1}" in
    1) species="cattle" ;;
    2) species="sheep" ;;
    3) species="goat" ;;
    4) species="camel" ;;
    5) species="horse" ;;
    *) species="cattle" ;;
  esac

  read -r -p "Breeding goal (default=growth): " goal
  goal="${goal:-growth}"

  echo
  echo "Phase 1 pipeline type:"
  echo "  1) Default — literature-aligned baseline (GBLUP) only"
  echo "  2) Default + R&D — default track plus optional benchmarking layer"
  echo
  read -r -p "Pipeline [1-2] (default=2): " mode_choice

  local run_mode
  case "${mode_choice:-2}" in
    1) run_mode="default" ;;
    2) run_mode="default_plus_rnd" ;;
    *) run_mode="default_plus_rnd" ;;
  esac

  echo
  echo "Selected settings:"
  echo "  Species : ${species}"
  echo "  Goal    : ${goal}"
  echo "  Mode    : ${run_mode}"
  ask_execution_mode

  if [[ "${EXEC_MODE}" == "sbatch" ]]; then
    ask_sbatch_resources
  fi

  mkdir -p "${PHASE1_DIR}"
  local cmd="bash 02_phase1_train_validate.sh ${run_mode}"
  local job_name="breedai_phase1"

  echo
  read -r -p "Proceed with Phase 1? [y/N]: " confirm
  if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    return
  fi

  if [[ "${EXEC_MODE}" == "sbatch" ]]; then
    submit_with_sbatch "${job_name}" "${cmd}" "${SLURM_PARTITION}" "${SLURM_CPUS}" "${SLURM_MEM}" "${SLURM_TIME}"
  else
    run_locally "${cmd}"
  fi
}

run_phase2() {
  echo
  echo "You selected: Phase 2 — Deployment & Prediction"
  echo "Deploy models on full dataset and predict breeding values for new animals."
  echo

  # Check Phase 1 completed
  if [[ ! -d "${PHASE1_DIR}/training_validation" ]]; then
    echo "❌ ERROR: Phase 1 results not found at ${PHASE1_DIR}/training_validation"
    echo "   Please run Phase 1 first before deploying models."
    return
  fi

  read -r -p "Path to new animals genotype CSV file (Enter = use input/Geno.csv): " new_geno
  if [[ -z "${new_geno}" ]]; then
    new_geno="${REPO_ROOT}/input/Geno.csv"
    echo "Using default: ${new_geno}"
  fi

  if [[ ! -f "${new_geno}" ]]; then
    echo "❌ ERROR: File not found: ${new_geno}"
    return
  fi

  mkdir -p "${PHASE2_DIR}"
  local cmd="bash 05_phase2_predict_unified.sh"

  echo
  echo "Selected settings:"
  echo "  New genotype file : ${new_geno}"
  echo "  Mode              : deploy (if needed) + predict"
  echo
  echo "This will:"
  echo "  1. Check if models are already deployed"
  echo "  2. Deploy models automatically if needed (using full dataset)"
  echo "  3. Predict on new animals"
  echo "  4. Generate reports"

  ask_execution_mode

  if [[ "${EXEC_MODE}" == "sbatch" ]]; then
    ask_sbatch_resources
  fi

  echo
  read -r -p "Proceed with Phase 2? [y/N]: " confirm
  if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    return
  fi

  # Export the new genotype file path for predict_unified.sh
  export NEW_X_FILE="${new_geno}"

  if [[ "${EXEC_MODE}" == "sbatch" ]]; then
    submit_with_sbatch "breedai_phase2" "${cmd}" "${SLURM_PARTITION}" "${SLURM_CPUS}" "${SLURM_MEM}" "${SLURM_TIME}"
  else
    run_locally "${cmd}"
  fi
}

check_status_and_results() {
  echo
  echo "You selected: Check job status and results"
  echo

  echo "Recent SLURM jobs:"
  squeue -u "${USER}" || true

  echo
  echo "Recent log files:"
  ls -lt "${LOG_DIR}" 2>/dev/null | head -20 || echo "No logs found."

  print_paths_summary
  pause_line
}

test_setup() {
  echo
  echo "You selected: Test setup"
  echo

  echo "Checking Python..."
  python --version || true

  echo
  echo "Checking key scripts..."
  for f in \
    "${SCRIPT_DIR}/02_phase1_train_validate.sh" \
    "${SCRIPT_DIR}/02a_phase1_train_validate_array.py" \
    "${SCRIPT_DIR}/05_phase2_predict_unified.sh"
  do
    if [[ -f "${f}" ]]; then
      echo "  OK  ${f}"
    else
      echo "  MISSING  ${f}"
    fi
  done

  echo
  echo "Expected job groups:"
  echo "  - Phase 1 default / default_plus_rnd (input)"
  echo "  - Phase 2 deployment"
  print_paths_summary
  pause_line
}

# =========================
# Main menu
# =========================
print_header
load_conda_hint

echo "Choose what you want to run:"
echo
echo "  1) Phase 1 — Learning & Benchmarking"
echo "     Train and compare models on a dataset"
echo
echo "  2) Phase 2 — Deployment & Prediction"
echo "     Predict breeding values for new animals"
echo
echo "  3) Check job status and results"
echo "     View SLURM jobs, logs, and output folders"
echo
echo "  4) Test setup"
echo "     Validate environment and show expected job groups"
echo
echo "  5) Exit"
echo

read -r -p "Enter your choice [1-5]: " main_choice

case "${main_choice}" in
  1) run_phase1 ;;
  2) run_phase2 ;;
  3) check_status_and_results ;;
  4) test_setup ;;
  5)
    echo
    echo "Goodbye."
    exit 0
    ;;
  *)
    echo "Invalid choice. Please rerun the script and choose 1-5."
    exit 1
    ;;
esac

# BreedAI User Guide

## Prerequisites

- Conda (or pip) to create the environment
- For SLURM/HPC runs: access to a cluster with a SLURM scheduler (the pipeline can
  also run locally via the menu's "Local" option)
- This repository cloned into your project directory

---

## Installation

### 1. Core environment (required)

```bash
cd /path/to/BreedAI-Framework          # the cloned repo
conda env create -f environment.yml    # creates the `genomic_pred` env
conda activate genomic_pred
# …or, without conda:  pip install -r requirements.txt
```

This is everything you need for the **default GBLUP track** and the **15 Python
R&D models** (penalized linear, tree/boosting, kernel/GP, neural) plus ensembles.

### 2. R + BGLR (optional — Bayesian methods only)

The Bayesian alphabet models — **BayesA / BayesB / BayesCpi** — are the *only* part
of BreedAI that needs R (via the **BGLR** package). Everything else runs without R.
If R/BGLR isn't available, BreedAI **skips those three methods and the run still
completes** — it never blocks the pipeline.

To enable them, create the separate optional R environment and point BreedAI at it:

```bash
conda env create -f environment-r.yml                          # creates `breedai_r` (R + BGLR)
export BGLR_RSCRIPT="$(conda run -n breedai_r which Rscript)"   # tell BreedAI which R to use
conda run -n breedai_r Rscript -e 'library(BGLR); cat("BGLR OK\n")'   # verify
```

`BGLR_RSCRIPT` can point at **any** R that has BGLR — a conda env, or a module /
system R (on an HPC: `module load R` then `export BGLR_RSCRIPT=$(which Rscript)`).

Notes:
- Keeping R in a **separate** env means a failed R install never blocks the core
  pipeline — you only lose the three Bayesian methods.
- The probe that detects BGLR is time-bounded (`BREEDAI_BGLR_CHECK_TIMEOUT`,
  default 60s), and the default track skips it entirely, so a slow R can't stall a run.

### 3. SLURM account (HPC only)

Set your account once before submitting — see [Quick Start](#quick-start):
`export SBATCH_ACCOUNT=<your-account>`.

---

## Quick Start

```bash
ssh <YOUR_USER>@<YOUR_HPC_LOGIN_HOST>
cd <PROJECT_DIR>/scripts

source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
conda activate genomic_pred

# Set your SLURM account (find it: sacctmgr show assoc user=$USER format=account).
# Optional if your cluster has a default account; the menu also prompts for it.
export SBATCH_ACCOUNT=<your-account>

./start_menu.sh
```

Menu options:

```
1) Phase 1 — Learning & Benchmarking
2) Phase 2 — Deployment & Prediction
3) Check job status and results
4) Test setup
5) Exit
```

---

## Input Data

Phase 1 reads two files from `input/`:

| File | What it is |
|------|-----------|
| **`Geno.csv`** — genotypes | The SNP marker matrix: one row per animal, one column per SNP, coded **0 / 1 / 2** (number of copies of the alternate allele). This is the genomic data the models learn from. |
| **`Pheno.csv`** — phenotypes | The observed trait values: one row per animal, one column per trait (e.g. milk yield, growth). These are the targets the models are trained to predict and are evaluated against. |

The two files are matched by animal ID. That is all BreedAI needs — drop your own
cohort's `Geno.csv` / `Pheno.csv` into `input/` and run.

### Reproducing the shipped cattle example

The repo ships the public **Van den Berg** cattle benchmark under
[`cattle_dataset/`](cattle_dataset/README.md). You don't use its files directly —
you first generate the BreedAI-format `Geno.csv` / `Pheno.csv` from the raw data,
then copy them into `input/`. The **[`cattle_dataset/README.md`](cattle_dataset/README.md)**
has the full walkthrough (dataset description, the QTL300 / r_g = 0.8 scenario,
and how to build other replicates).

For the public cattle benchmark, first build the processed files from the raw
data, then copy them in:

```bash
# From the repository root — builds Geno_QTL300_rg8.csv / Pheno_QTL300_rg8.csv.
# No arguments are needed: the defaults reproduce the poster scenario.
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py

# Copy the generated files into the runtime input folder:
cp cattle_dataset/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv  input/Geno.csv
cp cattle_dataset/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv input/Pheno.csv
```

> **Scope:** this public companion repo starts from genotypes. VCF / PLINK / FASTQ
> entry (upstream sequencing → VCF QC) is out of scope here.

---

## Two Modes

### Default Mode
Literature/production-aligned pipeline:
- **GBLUP** (ridge regression on VanRaden G-matrix)
- **ssGBLUP** (if pedigree exists)

### Default + R&D Mode
Extends default with the full algorithm suite on the **same splits and preprocessing**:

| Family | Algorithms |
|--------|-----------|
| **Linear (default)** | GBLUP_Ridge, GBLUP_RidgeCV |
| **Penalized linear** | LASSO, LASSO_CV, ElasticNet, ElasticNet_CV, BayesianRidge |
| **Bayesian alphabet** | BayesA, BayesB, BayesCpi (via R/BGLR) |
| **Tree / boosting** | RandomForest, XGBoost, LightGBM |
| **Kernel / probabilistic** | SVR_RBF, SVR_Linear, GP_RBF, GP_Matern52 |
| **Neural** | NeuralNet_MLP |
| **Ensembles** | Simple average, Median, Weighted average, Non-negative ridge stacking |

**Total:** 18 individual algorithms + 4 ensemble methods.

---

## Menu Options

### Option 1: Phase 1 — Learning & Benchmarking

Trains and evaluates all algorithms on the dataset.

1. Loads Geno.csv + Pheno.csv
2. Runs QC (variance filtering, G-matrix calculation)
3. Splits data: 60% train / 20% validation / 20% test
4. Trains all algorithms per trait (SLURM array jobs)
5. Creates ensemble predictions (stacking)
6. Generates report notebooks

### Option 2: Phase 2 — Deployment & Prediction

Deploys models and predicts on new animals.

1. Checks if models are already deployed
2. Deploys models on full dataset if needed
3. Predicts breeding values for new animals
4. Generates deployment and prediction reports

### Option 3: Check Job Status

Shows current SLURM jobs and recent log files.

### Option 4: Test Setup

Validates environment and prints output locations.

---

## Running Without the Menu

```bash
cd scripts
conda activate genomic_pred

# Phase 1 R&D (all algorithms + ensembles)
sbatch 02_phase1_train_validate.sh default_plus_rnd

# Phase 2 (deploy + predict)
export NEW_X_FILE="/path/to/NewGeno.csv"
bash 05_phase2_predict_unified.sh
```

---

## Where Outputs Go

| Output | Location |
|--------|----------|
| QC / G-matrix | `Phase1_Learning_Benchmarking/QC/` |
| Training results (all algorithms) | `Phase1_Learning_Benchmarking/training_validation/` |
| Combined results CSV | `Phase1_Learning_Benchmarking/training_validation/combined_train_validate_results.csv` |
| Stacking weights | `Phase1_Learning_Benchmarking/training_validation/stacking_Trait_*/` |
| Variance filter mask | `Phase1_Learning_Benchmarking/training_validation/variance_filter_mask.npy` |
| Deployed models | `Phase2_Deployment_Prediction/deployment/models/` |
| Prediction results | `Phase2_Deployment_Prediction/prediction/` |
| Phase 1 report notebooks | `notebooks/Phase_1_Learning_Benchmarking/` |
| Phase 2 report notebooks | `notebooks/Phase2_Deployment_Prediction/` |
| SLURM logs | `logs/` (subdirs: `train_validate/`, `deployment/`, `prediction/`) |

---

## Reports

| Report | Location |
|--------|----------|
| Preprocessing QC | `notebooks/Phase_1_Learning_Benchmarking/1.1_Preprocessing_report.ipynb` |
| Benchmarking | `notebooks/Phase_1_Learning_Benchmarking/1.2_Learning_Benchmarking_report.ipynb` |
| Deployment | `notebooks/Phase2_Deployment_Prediction/2.2_Deployment_report.ipynb` |
| Prediction results | `notebooks/Phase2_Deployment_Prediction/2.3_Prediction_report.ipynb` |

---

## Repository Structure

```
BreedAI-Framework/
├── input/                            ← Put Geno.csv + Pheno.csv here (any cohort)
├── cattle_dataset/                   Shipped example dataset (Van den Berg cattle)
│   ├── raw/vandenberg/               raw data (as downloaded)
│   └── processed/                    BreedAI-format Geno/Pheno (rebuilt locally)
├── Phase1_Learning_Benchmarking/     Phase 1 outputs (QC, training, stacking)
├── Phase2_Deployment_Prediction/     Phase 2 outputs (deployment, prediction)
├── notebooks/                        Generated Jupyter report notebooks
├── scripts/                          All pipeline scripts
│   ├── start_menu.sh                 Interactive menu (main entry point)
│   ├── 02_phase1_train_validate.sh   Phase 1 SLURM launcher
│   ├── 02a_phase1_train_validate_array.py  R&D training engine
│   ├── 03a_phase2_deploy_array.py    Phase 2 deployment
│   ├── 04a_phase2_predict_array.py   Phase 2 prediction
│   ├── 05_phase2_predict_unified.sh  Phase 2 unified pipeline
│   ├── ensembles/                    Stacking, non-negative ridge
│   └── public_dataset/vandenberg/    Dataset adapters and converters
├── logs/                             SLURM job logs
├── USER_GUIDE.md                     This guide
└── README.md                         Project overview + reproduction steps
```

---

## SLURM Resource Defaults

| Resource | Value |
|----------|-------|
| Partition | batch |
| Account | `$SBATCH_ACCOUNT` (unset → cluster default) |
| Memory | 512 GB |
| CPUs | 64 |
| Time limit | 7 days |

Customizable when prompted by the menu. Set your account with
`export SBATCH_ACCOUNT=<your-account>` before running (see Quick Start).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python: can't open file '.../scripts/...': No such file or directory` | You're not in the repo root. `cd` into the cloned repository first, then run the command (all paths are relative to the repo root). |
| `ModuleNotFoundError: numpy` | Activate conda env first |
| `sbatch: command not found` | Normal on login nodes — sbatch still works |
| Job stuck at BayesianRidge | Normal for large datasets; check memory with `sacct -j JOBID --format=MaxRSS` |
| `Feature mismatch` in Phase 2 | Different SNP panels are supported (aligned automatically). If overlap < 50%, prediction is rejected |
| `No Phase 1 results found` | Run Phase 1 before Phase 2 |
| Bayesian methods (BayesA/B/Cpi) missing from results | Optional — they need R + BGLR. Install `environment-r.yml` and `export BGLR_RSCRIPT=$(conda run -n breedai_r which Rscript)` (see Installation §2). The rest of the run is unaffected. |
| Run seems to hang right after `module load R` | A slow R was blocking the BGLR probe. It's now time-bounded (`BREEDAI_BGLR_CHECK_TIMEOUT`, default 60s); update to the latest scripts, or `export BREEDAI_SKIP_BGLR_CHECK=1` to skip R entirely. |

---

*Last updated: 2026-07-07*

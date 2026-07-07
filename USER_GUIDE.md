# BreedAI User Guide

## Prerequisites

- Access to Ibex (KAUST HPC)
- Conda environment `genomic_pred`
- Project directory: `<PROJECT_DIR>/`

---

## Quick Start

```bash
ssh <YOUR_USER>@<YOUR_HPC_LOGIN_HOST>
cd <PROJECT_DIR>/scripts

source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh
conda activate genomic_pred

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

Phase 1 expects **`dataset/input/Geno.csv`** and **`dataset/input/Pheno.csv`**.

For the public cattle benchmark:

```bash
# From repository root
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv dataset/input/Geno.csv
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv dataset/input/Pheno.csv
```

See `dataset/public_datasets/cattle/README.md` for details.

### Supported Input Types

| Type | Description | Status |
|------|-------------|--------|
| **genotype** | Geno.csv + Pheno.csv (0/1/2 matrix) | **Working** |

> The public companion repo starts from genotypes. VCF / PLINK / FASTQ entry
> (upstream sequencing → VCF QC) is out of scope here.

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
├── dataset/                          Input data and public benchmarks
│   ├── input/                        ← Put Geno.csv + Pheno.csv here
│   └── public_datasets/cattle/       Van den Berg benchmark
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

## Pipeline Stages

Stages 1–3 (raw sequencing → VCF) are **upstream and out of scope** for this public
companion repo; BreedAI starts from genotypes (stage 4 onward).

| # | Stage | Status |
|---|-------|--------|
| 1 | FASTQ QC | Upstream — out of scope |
| 2 | Alignment | Upstream — out of scope |
| 3 | Joint SNP calling | Upstream — out of scope |
| 4 | Sample / SNP QC | **Implemented** |
| 5 | Imputation | Mean-fill working; Beagle optional |
| 6 | Genotype matrix | **Implemented** |
| 7 | GRM (VanRaden G-matrix) | **Implemented** |
| 8 | H-matrix / ssGBLUP | Implemented (skips if no pedigree) |
| 9 | Fixed effects | Implemented (intercept-only for POC) |
| 10 | Default GBLUP | **Implemented** |
| 11 | GEBV + reliability | **Implemented** |
| 12 | Selection index | Implemented (needs economic weights) |
| 13 | Monitoring / drift | Implemented (needs baseline run) |

---

## SLURM Resource Defaults

| Resource | Value |
|----------|-------|
| Partition | batch |
| Account | YOUR_SLURM_ACCOUNT |
| Memory | 512 GB |
| CPUs | 64 |
| Time limit | 7 days |

Customizable when prompted by the menu.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: numpy` | Activate conda env first |
| `sbatch: command not found` | Normal on login nodes — sbatch still works |
| Job stuck at BayesianRidge | Normal for large datasets; check memory with `sacct -j JOBID --format=MaxRSS` |
| `Feature mismatch` in Phase 2 | Different SNP panels are supported (aligned automatically). If overlap < 50%, prediction is rejected |
| `No Phase 1 results found` | Run Phase 1 before Phase 2 |
| BGLR/Rscript not found | Ensure `module load R` is in the SLURM script |

---

*Last updated: 2026-07-07*

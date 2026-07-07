# Public Benchmark Datasets for BreedAI Validation

This directory contains public benchmark datasets used to validate the BreedAI genomic prediction framework against published methods in the literature.

---

## Overview

The validation framework uses publicly available genomic prediction datasets to:
1. **Download** raw benchmark datasets from public repositories
2. **Preprocess** datasets to BreedAI format (Geno.csv, Pheno.csv)
3. **Run** BreedAI framework on the processed datasets
4. **Benchmark** BreedAI performance against published methods from the literature

---

## Standard Runtime Path (all public benchmarks)

BreedAI **does not** read processed files from this folder automatically. The single runtime input location is:

- **`dataset/input/Geno.csv`**
- **`dataset/input/Pheno.csv`**

Optional: `dataset/input/metadata.csv`, `dataset/input/pedigree.csv` when you use those features.

**Workflow:** choose a processed scenario under `dataset/public_datasets/cattle/processed/…`, **copy** (or symlink) the CSVs into `dataset/input/` with the standard names, then run Phase 1 from `scripts/start_menu.sh` (option **1**). See `dataset/input/README.md`.

---

## Quick Start: Reproducing the Cattle Benchmark (QTL300, r_g = 0.8)

**Current reference scenario:** **`processed/vandenberg_QTL300_rg8/`** (300 QTL, genetic correlation 0.8, replicate 1).

For full details see **`processed/vandenberg_QTL300_rg8/PREPARATION.md`**.

```bash
# From repository root
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv dataset/input/Geno.csv
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv dataset/input/Pheno.csv

cd scripts
./start_menu.sh
# Choose 1) Phase 1 — Learning & Benchmarking
```

**Output:** `Phase1_Learning_Benchmarking/`, `notebooks/Phase_1_Learning_Benchmarking/`.

---

## Dataset: Van den Berg et al. (2020)

### Citation

**Van den Berg, I., et al. (2020)**  
"Across population genomic prediction scenarios in which Bayesian variable selection outperforms GBLUP"  
*BMC Genomics*, 21, 492.  
**Paper DOI:** [10.1186/s12864-020-06906-0](https://doi.org/10.1186/s12864-020-06906-0)  
**Data DOI (Dryad):** [10.5061/dryad.rq80k](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k)

### Dataset Description

Simulated Holstein cattle dataset designed to evaluate genomic prediction methods across different genetic architectures:

- **Genotype Data:** 1,285 animals × 26,503 SNPs (coding: 0/1/2)
- **Phenotype Data:** True Breeding Values (TBV) for 4 traits per replicate
- **Scenarios:** 4 QTL counts × 3 genetic correlations × 100 replicates = 1,200 phenotype files

### QTL-to-Replicate Mapping

Each `Phenotypes_GenCor_*/` directory contains 100 replicate files. Per the original paper, the replicates map to QTL scenarios as follows:

| Replicates | QTL Count | Genetic Architecture |
|-----------|-----------|---------------------|
| 1–25 | **3 QTLs** | Highly oligogenic (few large-effect loci) |
| 26–50 | **30 QTLs** | Oligogenic |
| 51–75 | **300 QTLs** | Moderately polygenic |
| 76–100 | **3,000 QTLs** | Highly polygenic (many small-effect loci) |

Combined with genetic correlations (0.4, 0.8, 1.0), this gives 12 distinct scenarios.

### Dataset Structure

```
dataset/public_datasets/cattle/
├── raw/
│   └── vandenberg/
│       ├── Genotypes_26503SNPs.txt          # Genotype data (33 MB)
│       ├── ID_Breed.txt                     # Breed identifiers
│       └── Phenotypes_GenCor_0.8/           # r_g = 0.8, the poster scenario (replicates 1-100)
│
├── processed/
│   └── vandenberg_QTL300_rg8/              # Reference scenario
│       ├── Geno_QTL300_rg8.csv             # → copy as dataset/input/Geno.csv
│       ├── Pheno_QTL300_rg8.csv            # → copy as dataset/input/Pheno.csv
│       ├── metadata_QTL300_rg8.json
│       ├── QC_REPORT.md
│       └── PREPARATION.md
│
└── input/                                   # Symlinks or copies for test runs
    ├── Geno.csv
    └── Pheno.csv
```

---

## Validation Workflow

### Step 1: Download Raw Data

**Script:** `scripts/public_dataset/vandenberg/01_download_vandenberg.py`

```bash
python scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --dataset vandenberg \
    --output-dir dataset/public_datasets/cattle/raw
```

> **Note:** Dryad uses dynamic file loading. Manual URL extraction may be required — see `scripts/public_dataset/vandenberg/00_quick_start.md`. Alternatively, transfer files manually (see `scripts/public_dataset/vandenberg/99_transfer_commands.md`).

### Step 2: Preprocess to BreedAI Format

**Script:** `scripts/public_dataset/vandenberg/02_prepare_vandenberg.py`

Converts raw files to BreedAI format. No separate QC step is needed — BreedAI runs QC during Phase 1.

```bash
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py \
    --geno-file dataset/public_datasets/cattle/raw/vandenberg/Genotypes_26503SNPs.txt \
    --pheno-file dataset/public_datasets/cattle/raw/vandenberg/Phenotypes_GenCor_0.8/Phenotypes_replicate_1.txt \
    --output-dir dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8 \
    --output-suffix _QTL300_rg8
```

### Step 3: Run BreedAI

Copy processed files into the standard input folder, then run via the menu:

```bash
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv dataset/input/Geno.csv
cp dataset/public_datasets/cattle/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv dataset/input/Pheno.csv

cd scripts
./start_menu.sh
```

- **Option 1** — Phase 1: Learning & Benchmarking (train/validate/test, algorithm comparison)
- **Option 2** — Phase 2: Deployment & Prediction (deploy models, predict on new animals)

### Step 4: Benchmark Against Published Methods

**Script:** `scripts/validation/compare_frameworks.py` *(planned — not yet implemented)*

Will compare BreedAI results against published methods:

| Method | Type | Reference |
|--------|------|-----------|
| GBLUP | Linear mixed model baseline | Van den Berg et al. (2020) |
| BayesA / BayesB / BayesCpi | Bayesian variable selection | Van den Berg et al. (2020) |
| Random Forest / XGBoost / MLP | Machine learning | BreedAI implementation |

**Metrics:** Prediction accuracy (Pearson *r*), MSE, bias.

---

## Simulated Test Scenarios for Phase 2 Validation

To test Phase 2's robustness to SNP overlap mismatches, three synthetic genotype files were generated from the training data. These simulate real-world situations where new animals are genotyped on a different chip or with missing/extra markers.

### How They Were Generated

From the training `Geno.csv` (1,285 animals × 26,479 SNPs), a subset of animals was randomly selected, renamed `NEW_1`, `NEW_2`, etc., and their SNP columns were manipulated:

```python
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("dataset/input")
geno = pd.read_csv(INPUT_DIR / "Geno.csv", index_col=0)
original_snps = list(geno.columns)
n_total_snps = len(original_snps)

def create_test_file(name, n_holdout, n_drop, n_extra, seed):
    np.random.seed(seed)
    holdout_idx = np.random.choice(len(geno), size=n_holdout, replace=False)
    new_geno = geno.iloc[holdout_idx].copy()
    new_geno.index = [f"NEW_{i+1}" for i in range(n_holdout)]

    # Drop some original SNPs
    if n_drop > 0:
        drop_idx = np.random.choice(n_total_snps, size=min(n_drop, n_total_snps - 1), replace=False)
        keep_snps = [s for i, s in enumerate(original_snps) if i not in drop_idx]
        new_geno = new_geno[keep_snps]

    # Add fake extra SNPs
    if n_extra > 0:
        extra_data = np.random.choice([0, 1, 2], size=(n_holdout, n_extra))
        extra_df = pd.DataFrame(extra_data, index=new_geno.index,
                                columns=[f"EXTRA_{i}" for i in range(n_extra)])
        new_geno = pd.concat([new_geno, extra_df], axis=1)

    # Shuffle column order and save
    cols = list(new_geno.columns)
    np.random.shuffle(cols)
    new_geno[cols].to_csv(INPUT_DIR / f"{name}.csv")

# Test 1: Normal overlap (~98%)
create_test_file("Geno_test1_normal_overlap", n_holdout=50, n_drop=500, n_extra=200, seed=42)

# Test 2: Low overlap (~60%)
create_test_file("Geno_test2_low_overlap", n_holdout=30, n_drop=int(n_total_snps*0.40), n_extra=100, seed=77)

# Test 3: Very low overlap (~30%)
create_test_file("Geno_test3_very_low_overlap", n_holdout=20, n_drop=int(n_total_snps*0.70), n_extra=50, seed=55)
```

### Test Files Summary

| File | Animals | Total SNPs | Common with training | Overlap | Expected Phase 2 Outcome |
|------|---------|-----------|---------------------|---------|--------------------------|
| `Geno_test1_normal_overlap.csv` | 50 | 26,179 | 25,979 | **98.1%** | Pass cleanly |
| `Geno_test2_low_overlap.csv` | 30 | 15,988 | 15,888 | **60.0%** | Warn but still predict |
| `Geno_test3_very_low_overlap.csv` | 20 | 7,994 | 7,944 | **30.0%** | Reject (below 50% threshold) |

**Location:** `dataset/input/Geno_test*.csv`

### Usage

After Phase 1 completes, run Phase 2 with each test file:

```bash
cd scripts
./start_menu.sh
# Choose 2) Phase 2 — Deployment & Prediction
# Enter path: dataset/input/Geno_test1_normal_overlap.csv
```

The pipeline should:
- **Test 1:** Predict without issues
- **Test 2:** Print a warning about low SNP overlap but still produce predictions
- **Test 3:** Reject the input and stop (overlap below 50% threshold)

---

## Validation Scenarios

### Scenario Matrix

| QTL Count | Genetic Correlation | Replicates | Status |
|-----------|-------------------|------------|--------|
| 3 | 0.4 | 1–25 | To be processed |
| 3 | 0.8 | 1–25 | To be processed |
| 3 | 1.0 | 1–25 | To be processed |
| 30 | 0.4 | 26–50 | To be processed |
| 30 | 0.8 | 26–50 | To be processed |
| 30 | 1.0 | 26–50 | To be processed |
| 300 | 0.4 | 51–75 | To be processed |
| 300 | 0.8 | 51–75 | ✅ Processed (replicate 1) |
| 300 | 1.0 | 51–75 | To be processed |
| 3000 | 0.4 | 76–100 | To be processed |
| 3000 | 0.8 | 76–100 | To be processed |
| 3000 | 1.0 | 76–100 | To be processed |

**Total:** 12 scenarios × 25 replicates each = 300 unique phenotype configurations (across 3 genetic correlations = 1,200 files total).

---

## Results Directory Structure *(planned)*

```
results/validation/
├── benchmarking/          # BreedAI benchmarking results
│   └── vandenberg/
│       └── QTL300_rg8/
│           └── replicate_1/
│
├── comparisons/           # Comparison with published methods
│   └── vandenberg/
│       └── QTL300_rg8/
│           ├── breedai_vs_gblup.csv
│           ├── breedai_vs_bayes.csv
│           └── summary_statistics.json
│
└── plots/                 # Visualization plots
    └── vandenberg/
        ├── accuracy_comparison.png
        ├── mse_comparison.png
        └── scenario_heatmap.png
```

---

## Literature References

### Primary Reference

**Van den Berg, I., Bowman, P. J., MacLeod, I. M., Hayes, B. J., Wang, T., Bolormaa, S., & Goddard, M. E. (2019)**  
"Multi-breed genomic prediction using Bayes R with sequence data and dropping variants with a small effect"  
*Genetics Selection Evolution*, 51, 70.  
**Paper DOI:** [10.1186/s12711-019-0522-8](https://doi.org/10.1186/s12711-019-0522-8)  
**Data DOI (Dryad):** [10.5061/dryad.rq80k](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k)

### Key Findings

1. **Bayesian methods** (BayesB, BayesCpi) outperformed GBLUP when QTL count was low (3, 30) and genetic correlation was moderate (0.4, 0.8)
2. **GBLUP** performed comparably or better when QTL count was high (3,000) and genetic correlation was high (1.0)
3. **Performance gap** between methods decreased with increasing QTL count (more polygenic traits)

### BreedAI Validation Goals

- Verify if BreedAI's ML approaches (ensemble stacking, XGBoost, etc.) can match or exceed Bayesian methods
- Determine optimal scenarios for each algorithm family
- Assess computational efficiency compared to Bayesian MCMC methods

---

## Scripts and Tools

### Data Preparation
- `scripts/public_dataset/vandenberg/01_download_vandenberg.py` — Download from Dryad
- `scripts/public_dataset/vandenberg/02_prepare_vandenberg.py` — Convert to BreedAI format (Geno.csv, Pheno.csv)

### Validation *(planned)*
- `scripts/validation/validate_predictions.py` — Run full validation pipeline
- `scripts/validation/compare_frameworks.py` — Compare with published methods

### Documentation
- `scripts/public_dataset/vandenberg/00_quick_start.md` — Quick start guide
- `scripts/public_dataset/vandenberg/99_transfer_commands.md` — File transfer instructions
- `processed/vandenberg_QTL300_rg8/PREPARATION.md` — Detailed reproduction steps

---

## Next Steps

1. ✅ **Download raw data** — Completed
2. ✅ **Create preprocessing scripts** — Completed
3. ✅ **Process initial dataset** — Completed (QTL300, r_g=0.8, replicate 1)
4. ✅ **Generate simulated test scenarios** — Completed (3 overlap test files)
5. ⏳ **Process additional scenarios** — In progress
6. ⏳ **Run BreedAI on all scenarios** — Pending
7. ⏳ **Implement comparison scripts** — Pending
8. ⏳ **Generate comparison reports** — Pending

---

## Notes

- All processed datasets maintain the original animal IDs for traceability
- **BreedAI runs QC** (variance filtering, G-matrix, imputation) as part of Phase 1; no separate QC step is required for clean datasets
- **Move** (or copy) `Geno.csv` and `Pheno.csv` into `dataset/input/` before running the menu
- Results can be aggregated across replicates for statistical robustness
- The validation framework is extensible to other public datasets and species

---

*Last Updated: 2026-03-30*  
*BreedAI — cattle public benchmark*

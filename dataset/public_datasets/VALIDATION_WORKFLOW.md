# Public Dataset Validation Workflow

This document explains the proper train/validation/test workflow for validating BreedAI against published methods using public benchmark datasets.

---

## Overview

For proper validation against published methods, we need a **three-way split**:

1. **Training Set (60%)**: Used for initial model training
2. **Validation Set (20%)**: Used for model selection and hyperparameter tuning during benchmarking
3. **Test Set (20%)**: **Held-out set** for final performance evaluation (never used during training)

This ensures:
- Fair comparison with published methods
- No data leakage
- Unbiased performance estimates

---

## Workflow Steps

### Step 1: Split the Dataset

**Script:** `scripts/validation/split_public_dataset.py`

```bash
cd <PROJECT_DIR>

python scripts/validation/split_public_dataset.py \
    --geno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv \
    --pheno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv \
    --output-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits \
    --train-size 0.6 \
    --val-size 0.2 \
    --test-size 0.2 \
    --random-state 42
```

**Output:**
```
data/public_datasets/processed/vandenberg_QTL300_rg8/splits/
├── train/
│   ├── Geno.csv      # 771 animals (60%)
│   └── Pheno.csv
├── validation/
│   ├── Geno.csv      # 257 animals (20%)
│   └── Pheno.csv
├── test/
│   ├── Geno.csv      # 257 animals (20%) - HELD OUT
│   └── Pheno.csv
└── split_info.json   # Split metadata
```

---

### Step 2: Benchmarking Phase (Train + Validation)

**Purpose:** Compare algorithms and select best models

**Data Used:**
- **Training set**: Train models
- **Validation set**: Evaluate and compare models (for model selection)

**Setup:**
```bash
# Link training data for benchmarking
cd <PROJECT_DIR>
rm data/Geno.csv data/Pheno.csv  # Remove current links
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train/Geno.csv data/Geno.csv
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train/Pheno.csv data/Pheno.csv
```

**Run Benchmarking:**
```bash
cd scripts
./start_menu.sh  # Select option 1: Phase 1 - Benchmarking
```

**What Happens:**
- Models are trained on the **training set** (771 animals)
- Models are evaluated on a **validation split** (20% of training data, ~154 animals)
- Performance metrics are compared across algorithms
- Best models are selected for each trait

**Output:** `benchmarking_array/` with algorithm comparison results

---

### Step 3: Deployment Phase (Train + Validation Combined)

**Purpose:** Train final models on full training data

**Data Used:**
- **Training + Validation sets combined**: Full training data (1,028 animals = 80%)

**Setup:**
```bash
# Combine train + validation for deployment
cd <PROJECT_DIR>
python scripts/validation/combine_train_val.py \
    --train-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train \
    --val-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits/validation \
    --output-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train_val

# Link combined data for deployment
rm data/Geno.csv data/Pheno.csv
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train_val/Geno.csv data/Geno.csv
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/train_val/Pheno.csv data/Pheno.csv
```

**Run Deployment:**
```bash
cd scripts
./start_menu.sh  # Select option 2: Phase 2 - Deployment
```

**What Happens:**
- Final models are trained on **train + validation** (1,028 animals)
- Models are saved for prediction
- No test set is used here

**Output:** `deployment_array/models/` with trained models

---

### Step 4: Final Testing (Test Set Only)

**Purpose:** Evaluate final model performance on held-out test set

**Data Used:**
- **Test set only**: Held-out data (257 animals = 20%)

**Setup:**
```bash
# Link test data for final evaluation
cd <PROJECT_DIR>
rm data/Geno.csv data/Pheno.csv
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/test/Geno.csv data/Geno.csv
# Note: Pheno.csv is only for reference - predictions will be compared to it
ln -s $(pwd)/data/public_datasets/processed/vandenberg_QTL300_rg8/splits/test/Pheno.csv data/Pheno.csv
```

**Run Prediction:**
```bash
cd scripts
./start_menu.sh  # Select option 3: Phase 3 - Prediction
```

**What Happens:**
- Trained models from deployment phase are loaded
- Predictions are generated for **test set animals**
- Predictions are compared to true values (from test Pheno.csv)
- Final performance metrics are calculated

**Output:** `prediction_array/` with GEBV predictions and performance metrics

---

## Performance Evaluation

After Step 4, you can evaluate final performance:

**Metrics to Calculate:**
- **Prediction Accuracy**: Correlation between predicted and true breeding values
- **Mean Squared Error (MSE)**: Prediction error
- **Bias**: Systematic prediction bias
- **R² Score**: Coefficient of determination

**Compare with Published Methods:**
- GBLUP performance from Van den Berg et al. (2020)
- Bayesian methods (BayesA, BayesB, BayesCπ) performance
- Determine when BreedAI outperforms traditional methods

---

## Important Notes

1. **Test Set is NEVER Used During Training**
   - Test set is only used in Step 4 for final evaluation
   - This ensures unbiased performance estimates

2. **Reproducibility**
   - Use `--random-state 42` for consistent splits
   - Save `split_info.json` to track which animals are in which set

3. **Validation Set Purpose**
   - Used during benchmarking for model selection
   - Combined with training set for final model deployment
   - Ensures models are trained on maximum available data

4. **Fair Comparison**
   - This workflow matches standard machine learning practices
   - Ensures fair comparison with published methods
   - Prevents overfitting and data leakage

---

## Quick Reference

```bash
# 1. Split dataset
python scripts/validation/split_public_dataset.py \
    --geno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv \
    --pheno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv \
    --output-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits

# 2. Benchmarking (use train set)
# Link train/Geno.csv and train/Pheno.csv to data/
cd scripts && ./start_menu.sh  # Option 1

# 3. Deployment (use train+val combined)
# Link train_val/Geno.csv and train_val/Pheno.csv to data/
cd scripts && ./start_menu.sh  # Option 2

# 4. Final Testing (use test set)
# Link test/Geno.csv and test/Pheno.csv to data/
cd scripts && ./start_menu.sh  # Option 3
```

---

*Last Updated: 2025-12-09*



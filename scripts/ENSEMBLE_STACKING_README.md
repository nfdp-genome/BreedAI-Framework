# Stacking Ensemble (Non-Negative Ridge, OOF)

This document describes the stacking ensemble used in Phase 1 train-validate benchmarking.

## Nested CV design

Current implementation uses **nested CV** for stacking rigor:

- **Outer CV (K=5 by default)**: pure evaluation folds
- **Inner CV (J=3 by default)**: model tuning + OOF generation for stacker training

Per outer fold:

1. Tune each base model on outer-train with inner CV
2. Build inner OOF prediction matrix on outer-train
3. Fit non-negative ridge stacker on inner OOF
4. Retrain tuned base models on full outer-train
5. Predict outer-test and apply stacker

## What is implemented

BreedAI now includes a stacking ensemble named:

- `Ensemble_Stacking_NonNeg_Ridge`

It is trained using:

- **OOF base-model predictions** from the train split
- **non-negative ridge regression** (`w >= 0`)
- optional intercept and column standardization

Objective:

- minimize `||y - P w||^2 + alpha ||w||^2`, subject to `w >= 0`

where:

- `P` is the OOF prediction matrix (samples x base models)
- `y` is the target vector for the train split

## Why OOF stacking

Each OOF row is predicted by base models that were not trained on that row, which avoids direct leakage in stacker training.

## Where it runs

Phase 1 script:

- `scripts/02a_phase1_train_validate_array.py`

Stacking is created together with other ensembles during trait-level ensemble generation.

## Configuration

CLI options supported by `02a_phase1_train_validate_array.py`:

- `--stack_alpha` (default `0.01`)
- `--stack_fit_intercept` (default `true`)
- `--stack_standardize_cols` (default `true`)
- `--stack_normalize_weights` (default `false`)
- `--stack_outer_splits` (default `5`)
- `--stack_inner_splits` (default `3`)
- `--stack_n_splits` (deprecated alias for outer folds)

## Model inclusion/exclusion policy

Stacking uses only successful base models with complete OOF coverage.

- If a model fails in any OOF fold, it is excluded.
- Exclusions are logged with reason.

## Artifacts per trait

For each trait, files are written under:

- `Phase1_Learning_Benchmarking/training_validation/stacking_<trait>/`

Artifacts:

- `stacking_weights_by_fold.csv`
- `stacking_weights_summary.csv`
- `stacking_family_weights_summary.csv`
- `oof_predictions_stacking.npz`

## Model family groups used in summaries

- `linear`: GBLUP_Ridge, GBLUP_RidgeCV, LASSO, LASSO_CV, ElasticNet, ElasticNet_CV, BayesianRidge
- `tree_boosting`: RandomForest, XGBoost, LightGBM
- `kernel_prob`: SVR_RBF, SVR_Linear, GP_RBF, GP_Matern52
- `neural`: NeuralNet_MLP
- `bayesian_alphabet`: BayesA, BayesB, BayesCpi

## Optimizer backend

- Preferred: `scipy.optimize.lsq_linear` with ridge augmentation
- Fallback: deterministic projected gradient descent when SciPy is unavailable

## Reporting

Phase 1 report notebook includes a stacking preference section when artifacts are present:

- top weighted models
- family-level weight totals

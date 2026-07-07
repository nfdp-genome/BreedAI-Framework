# Standard runtime input folder

BreedAI Phase 1 reads **your active analysis files** from this directory.

## Required files

| File | Description |
|------|-------------|
| `Geno.csv` | Samples × SNPs matrix. Rows: `Animal_ID` (or first column = ID). Columns: `SNP_1`, `SNP_2`, … or SNP IDs. Values: **0 / 1 / 2** (additive coding). |
| `Pheno.csv` | Samples × traits. Must include `Animal_ID` and one column per trait (e.g. `Trait_1`, …). |

## Optional files

| File | Description |
|------|-------------|
| `metadata.csv` | Sample covariates (breed, herd, sex, …) when available; used when the pipeline supports fixed-effect expansion. |
| `pedigree.csv` | Pedigree for ssGBLUP / H-matrix when available (format as expected by `scripts/stages/s08_h_matrix.py`). |

## Workflow

1. Copy or symlink prepared `Geno.csv` / `Pheno.csv` here (see `dataset/public_datasets/` for benchmark examples).
2. Run Phase 1 from `scripts/start_menu.sh` or `python scripts/07_run_poc.py`.
3. Outputs are written under `Phase1_Learning_Benchmarking/` (see top-level `README.md`).

**Note:** Files in `dataset/public_datasets/` are **reference benchmarks**; they are not read automatically. You copy the scenario you want into `dataset/input/`.

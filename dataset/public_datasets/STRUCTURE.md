# Public Datasets Folder Structure

This document explains the expected folder structure for public benchmark datasets in the BreedAI framework.

---

## Directory Structure

```
data/public_datasets/
├── README.md                    # Overview of public datasets & validation plan
├── VALIDATION_WORKFLOW.md      # Detailed train/val/test workflow guide
├── STRUCTURE.md                # This file - explains folder structure
├── raw/                         # Raw downloaded files (original format)
│   └── vandenberg/              # Van den Berg et al. (2020) dataset
│       ├── Genotypes_26503SNPs.txt
│       ├── ID_Breed.txt
│       └── Phenotypes_GenCor_*/  # Different genetic correlation scenarios
│           └── Phenotypes_replicate_*.txt
└── processed/                  # Converted to BreedAI format
    └── vandenberg_QTL300_rg8/  # Processed dataset (QTL300, rg=0.8)
        ├── Geno_QTL300_rg8.csv
        ├── Pheno_QTL300_rg8.csv
        ├── QC_REPORT.md         # Quality control report
        ├── qc_report_QTL300_rg8.json
        ├── metadata_QTL300_rg8.json
        └── splits/              # Train/validation/test splits
            ├── train/           # 60% - Training set
            │   ├── Geno.csv
            │   └── Pheno.csv
            ├── validation/      # 20% - Validation set
            │   ├── Geno.csv
            │   └── Pheno.csv
            ├── train_val/       # 80% - Combined train+val (for deployment)
            │   ├── Geno.csv
            │   └── Pheno.csv
            └── test/            # 20% - Test set (held-out)
                ├── Geno.csv
                └── Pheno.csv
```

---

## Documentation Files

### 1. `README.md`
**Purpose:** Overview and introduction to public datasets
- Explains what public datasets are used for
- Describes the validation framework
- Provides dataset citations and descriptions
- Documents the overall workflow

**Why you need it:** Provides context and documentation for anyone using or adding public datasets.

### 2. `VALIDATION_WORKFLOW.md`
**Purpose:** Detailed workflow guide for proper train/validation/test splitting
- Explains the three-way split (60% train, 20% val, 20% test)
- Documents the workflow steps
- Provides command examples
- Ensures no data leakage

**Why you need it:** Critical for proper validation methodology - ensures fair comparison with published methods.

---

## Scripts Location

Dataset-specific scripts are organized under:
```
scripts/public_dataset/
└── vandenberg/                  # Van den Berg dataset scripts
    ├── 01_download_vandenberg.py     # Download from Dryad
    ├── 02_prepare_vandenberg.py      # Convert to BreedAI format
    ├── 00_get_dryad_urls.py          # Helper for URL extraction
    ├── 01_run_download.sh            # Download script
    ├── 00_download_instructions.md   # Download guide
    ├── 00_quick_start.md             # Quick start guide
    └── vandenberg_urls.json.template
```

General validation scripts remain in:
```
scripts/validation/
├── quality_control.py           # Reusable QC functions
├── split_public_dataset.py      # General splitting script
├── detect_splits.py             # Detect existing splits
└── ...
```

---

## Adding New Datasets

When adding a new public dataset:

1. **Create raw data folder:**
   ```
   data/public_datasets/raw/new_dataset_name/
   ```

2. **Create processed data folder:**
   ```
   data/public_datasets/processed/new_dataset_name/
   ```

3. **Create dataset-specific scripts:**
   ```
   scripts/public_dataset/new_dataset_name/
   ├── 01_download_new_dataset.py
   ├── 02_prepare_new_dataset.py
   └── ...
   ```

4. **Update documentation:**
   - Add dataset info to `README.md`
   - Document workflow in `VALIDATION_WORKFLOW.md` if needed

---

## Usage

1. **Download raw data:**
   ```bash
   python scripts/public_dataset/vandenberg/01_download_vandenberg.py --dataset vandenberg
   ```

2. **Convert to BreedAI format:**
   ```bash
   python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py \
       --geno-file data/public_datasets/raw/vandenberg/Genotypes_26503SNPs.txt \
       --pheno-file data/public_datasets/raw/vandenberg/Phenotypes_GenCor_0.8/Phenotypes_replicate_1.txt \
       --output-dir data/public_datasets/processed/vandenberg_QTL300_rg8
   ```

3. **Split into train/val/test:**
   ```bash
   python scripts/validation/split_public_dataset.py \
       --geno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv \
       --pheno-file data/public_datasets/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv \
       --output-dir data/public_datasets/processed/vandenberg_QTL300_rg8/splits
   ```

4. **Run validation pipeline:**
   Use `scripts/01_pipeline_run_all.sh` Option 4, which automatically detects and uses the splits.


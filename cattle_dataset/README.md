# BreedAI — Cattle Benchmark Dataset

Everything you need to obtain the benchmark data, build BreedAI's input files
(`Geno.csv` / `Pheno.csv`), and run the pipeline.

This public companion repo ships a single cattle benchmark — the **Van den Berg
et al. simulated Holstein** dataset used in the ISMB 2026 poster. BreedAI starts
from genotypes; upstream sequencing (FASTQ → VCF) is out of scope.

This is the shipped example dataset. The pipeline's runtime inputs live in the
repo-root **`input/`** folder (§3), not here.

```
cattle_dataset/
├── README.md                     # This guide
├── raw/                          # Shipped raw data (as downloaded from Dryad)
│   └── vandenberg/
│       ├── Genotypes_26503SNPs.txt      # 1,285 animals × 26,503 SNPs (33 MB)
│       ├── ID_Breed.txt                 # breed identifiers
│       ├── DATASET_INFO.txt             # file inventory
│       └── Phenotypes_GenCor_0.8/       # r_g = 0.8 scenario, 100 replicate files
└── processed/                    # §2 writes Geno/Pheno CSVs + metadata here
                                   # (generated; contents are git-ignored)
```

---

## 1. The data

**Van den Berg, I., *et al.* (2020).** "Across population genomic prediction
scenarios in which Bayesian variable selection outperforms GBLUP."
*BMC Genomics* 21, 492. Paper DOI:
[10.1186/s12864-020-06906-0](https://doi.org/10.1186/s12864-020-06906-0) ·
Data (Dryad) DOI:
[10.5061/dryad.rq80k](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k)

A simulated Holstein cattle dataset for evaluating genomic prediction methods
across genetic architectures:

- **Genotypes:** 1,285 animals × 26,503 SNPs, additive-coded **0 / 1 / 2**.
- **Phenotypes:** True Breeding Values (TBVs); **4 traits** per replicate.
- **Scenarios:** simulated across QTL counts × genetic correlations, 100
  replicates each. This repo ships the **genetic correlation r_g = 0.8** set
  (`Phenotypes_GenCor_0.8/`); the r_g = 0.4 and 1.0 sets are available from the
  same Dryad record.

**Poster reference scenario — `vandenberg_QTL300_rg8`:** the **300-QTL**
architecture (a moderately polygenic trait) at genetic correlation **r_g = 0.8**,
replicate 1. After BreedAI's Phase-1 QC this yields **1,285 animals × 26,479
SNPs**, split 771 / 257 / 257 (60 / 20 / 20). Fixed effects are intercept-only
for this proof of concept.

---

## 2. Build `Geno.csv` / `Pheno.csv`

The processed CSVs are **not committed** — you regenerate them from the shipped
raw files (or from a fresh Dryad download) in one command.

### Option A — use the shipped raw data (default)

The defaults already point at the shipped files, so reproducing the poster
scenario takes **no arguments** and runs from **any** directory:

```bash
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py
```

This writes `Geno_QTL300_rg8.csv` and `Pheno_QTL300_rg8.csv` into
`cattle_dataset/processed/vandenberg_QTL300_rg8/`. (BreedAI runs its own QC in
Phase 1, so no separate QC step is needed.)

To convert a different replicate, override the paths (relative paths resolve
against the repo root):

```bash
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py \
    --pheno-file cattle_dataset/raw/vandenberg/Phenotypes_GenCor_0.8/Phenotypes_replicate_50.txt \
    --output-dir cattle_dataset/processed/vandenberg_rep50 \
    --output-suffix _rep50
```

### Option B — download from Dryad first

```bash
python scripts/public_dataset/vandenberg/01_download_vandenberg.py \
    --output-dir cattle_dataset/raw
```

> Dryad serves files via dynamic links; manual URL extraction may be required —
> see [`scripts/public_dataset/vandenberg/00_quick_start.md`](../scripts/public_dataset/vandenberg/00_quick_start.md).
> Then run the Option A command to convert.

### File format

Both files are matched by animal ID (first column).

| File | Rows | Columns | Values |
|------|------|---------|--------|
| **`Geno.csv`** | one per animal (`AnimalID`) | one per SNP (`SNP_1`, `SNP_2`, …) | `0` / `1` / `2` — copies of the alternate allele |
| **`Pheno.csv`** | one per animal (`AnimalID`, matching Geno.csv) | one per trait (`Trait_1`, …) | numeric phenotype / breeding value |

```csv
# Geno.csv                     # Pheno.csv
AnimalID,SNP_1,SNP_2,SNP_3     AnimalID,Trait_1,Trait_2
1,0,1,2                        1,45.2,12.3
2,1,1,0                        2,42.1,11.8
```

Optional input: `metadata.csv` (sample covariates — breed, herd, sex — for
fixed-effect modeling). A `pedigree.csv` (`animal, sire, dam`) is accepted but
not yet used; pedigree-based single-step (ssGBLUP / H-matrix) is future work.

---

## 3. Run BreedAI

BreedAI reads its inputs from the repo-root **`input/`** folder (it does *not*
read this processed folder automatically). Copy the built CSVs there under the
standard names:

```bash
cp cattle_dataset/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv  input/Geno.csv
cp cattle_dataset/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv input/Pheno.csv

cd scripts
./start_menu.sh
```

Menu:

```
1) Phase 1 — Learning & Benchmarking
2) Phase 2 — Deployment & Prediction
3) Check job status and results
4) Test setup
5) Exit
```

- **Option 1 — Phase 1:** loads `Geno.csv` + `Pheno.csv`, runs QC (variance
  filtering, VanRaden G-matrix), makes a seed-controlled **60 / 20 / 20** split,
  trains the default GBLUP baseline and the full R&D algorithm suite + stacking
  ensembles on the *same* splits, and generates report notebooks.
- **Option 2 — Phase 2:** deploys models on the combined train+validation data
  and predicts breeding values for new animals, with SNP-overlap guardrails
  (warn below 80 %, reject below 50 %).

Outputs land under `Phase1_Learning_Benchmarking/` and
`Phase2_Deployment_Prediction/`; see the top-level
[`README.md`](../README.md) and [`USER_GUIDE.md`](../USER_GUIDE.md) for the full
output map, the algorithm list, and troubleshooting.

---

## 4. Test the deployment SNP-overlap guardrails (optional)

Phase 2 aligns new-animal genotypes to the reference panel and guards against
mismatched SNP sets — it **warns** below 80 % overlap and **rejects** below 50 %.
To see all three outcomes, generate three synthetic new-animal files from
`input/Geno.csv` (markers dropped/added + columns shuffled):

```bash
python scripts/public_dataset/vandenberg/03_make_overlap_test_files.py
```

This writes into `input/` (both files are gitignored):

| File | Overlap | Expected Phase-2 outcome |
|------|---------|--------------------------|
| `Geno_test1_normal_overlap.csv`   | ~98 % | passes cleanly |
| `Geno_test2_low_overlap.csv`      | ~60 % | warns, still predicts |
| `Geno_test3_very_low_overlap.csv` | ~30 % | rejected (below 50 %) |

Then run Phase 2 (menu → option 2) and, when prompted for the new-animal
genotype file, point it at each `input/Geno_test*.csv` in turn.

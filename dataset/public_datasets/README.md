# Public Benchmark Datasets for BreedAI Validation

This directory contains public benchmark datasets used to validate the **BreedAI** genomic prediction framework across species and against published methods in the literature.

---

## Plan to benchmark BreedAI with public datasets

BreedAI will be benchmarked on public datasets from multiple livestock species. Status and scope by species:

| Species | Dataset | Source / Accession | Scope | Status |
|--------|---------|--------------------|--------|--------|
| **Cattle** | Van den Berg et al. (2020) | Dryad (DOI: 10.5061/dryad.rq80k) | Full genomic prediction (GBLUP, Bayesian, ML); multiple QTL/scenario replicates | ✅ In use |
| **Sheep** | Hu sheep body weight | **GSE152717** (Gene Expression Omnibus) | Genomic prediction for body weight | 📋 Planned |
| **Goats** | American Alpine dairy goats, milk | **GSE145419** (GEO) | Milk trait genomic prediction | 📋 Planned |
| **Camels** | Bitaraf Sani et al. (2021) | TBD — confirm data repository or obtain permission | Genomic prediction (once data access confirmed) | 📋 Pending access |
| **Horses** | Equine multi-breed SNP panel | TBD | **Population genetics demo** (e.g. clustering, structure); treat as prediction benchmark only if/when GWAS data with **open phenotypes** are secured | 📋 Demo / future |

- **Cattle:** Primary benchmark; Van den Berg scenario matrix (QTL count × genetic correlation) fully supported.
- **Sheep, Goats:** Planned benchmarks; prepare BreedAI input (Geno.csv, Pheno.csv) from GEO/processed sources and run Phase 1–2.
- **Camels:** Pending confirmation of public repository or author permission for Bitaraf Sani 2021 data.
- **Horses:** Use equine SNP panel for population-structure/clustering demos; full prediction benchmarking deferred until open phenotype data are available.

---

## Overview

The validation framework uses publicly available genomic (and, where applicable, phenotypic) datasets to:

1. **Download** raw data from public repositories (Dryad, GEO, or species-specific sources).
2. **Preprocess** into BreedAI format (`Geno.csv`, `Pheno.csv`) — see [VALIDATION_WORKFLOW.md](VALIDATION_WORKFLOW.md).
3. **Run** BreedAI Phase 1 (Learning & Benchmarking) and Phase 2 (Deployment & Prediction).
4. **Benchmark** BreedAI performance against published methods and literature.

---

## Directory structure (by species)

```
dataset/public_datasets/
├── README.md                    # This file — plan and overview
├── VALIDATION_WORKFLOW.md       # Train/val/test workflow
├── STRUCTURE.md                  # Folder structure details
├── cattle/                       # Cattle benchmarks
│   ├── raw/
│   │   └── vandenberg/           # Van den Berg et al. (2020)
│   └── processed/
│       └── vandenberg_QTL300_rg8/
├── sheep/                        # Sheep benchmarks
│   └── GSE152717/                # Hu sheep body weight (GSE152717)
├── goat/                         # Goat benchmarks
│   └── README.md                 # American Alpine milk (GSE145419) — planned
├── camels/                       # (Planned) Bitaraf Sani 2021 — pending repo/permission
└── horses/                       # (Planned) Equine SNP panel — population genetics demo
```

See [STRUCTURE.md](STRUCTURE.md) for full layout and split conventions.

---

## Cattle: Van den Berg et al. (2020)

### Citation
**Van den Berg, I., et al. (2020)**  
"Across population genomic prediction scenarios in which Bayesian variable selection outperforms GBLUP"  
**DOI:** [10.5061/dryad.rq80k](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k)  
**Dryad:** https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k

### Description
- **Genotypes:** 1,285 animals × 26,503 SNPs (0/1/2).
- **Phenotypes:** Multiple QTL scenarios (3, 30, 300, 3000 QTLs) × genetic correlations (0.4, 0.8, 1.0); 100 replicates per scenario; 4 traits per replicate (TBVs).

### Cattle dataset structure

```
dataset/public_datasets/cattle/
├── raw/
│   └── vandenberg/
│       ├── Genotypes_26503SNPs.txt
│       ├── ID_Breed.txt
│       └── Phenotypes_GenCor_0.8/
│           └── Phenotypes_replicate_*.txt
└── processed/
    └── vandenberg_QTL300_rg8/
        ├── Geno_*.csv, Pheno_*.csv   # BreedAI format
        ├── QC_REPORT.md
        └── metadata_*.json
```

### Scripts (cattle / vandenberg)
- **Download:** `scripts/public_dataset/vandenberg/01_download_vandenberg.py` (or manual transfer — see `00_quick_start.md`, `99_transfer_commands.md`).
- **Prepare:** `scripts/public_dataset/vandenberg/02_prepare_vandenberg.py` — convert raw files to BreedAI `Geno.csv` / `Pheno.csv`.

Then run BreedAI from project root: `cd scripts` → `./start_menu.sh` → Phase 1 (and Phase 2 as needed), using `dataset/input/` symlinked or copied from the chosen processed cattle scenario.

---

## Sheep: Hu sheep body weight (GSE152717)

- **Source:** Gene Expression Omnibus — **GSE152717** (Hu sheep body weight).
- **Planned use:** Body weight trait; prepare Geno/Pheno from GEO/processed matrices, then run BreedAI Phase 1–2 and benchmark against literature.
- **Status:** Planned; data under `dataset/public_datasets/sheep/GSE152717/`.

---

## Goats: American Alpine dairy goats, milk (GSE145419)

- **Source:** GEO — **GSE145419** (American Alpine dairy goats, milk dataset).
- **Planned use:** Milk-related traits; same workflow as sheep (prepare → BreedAI → benchmark).
- **Status:** Planned; see `dataset/public_datasets/goat/README.md`.

---

## Camels: Bitaraf Sani et al. (2021)

- **Planned use:** Genomic prediction benchmark once data are accessible.
- **Status:** Pending — confirm public data repository or obtain author permission before adding to `dataset/public_datasets/camels/` and running BreedAI.

---

## Horses: Equine multi-breed SNP panel

- **Planned use:** Multi-breed SNP panel for **population genetics demos** (e.g. clustering, structure). Not treated as a full genomic prediction benchmark unless/until **GWAS or other data with open phenotypes** are secured.
- **Status:** Demo/future; full prediction benchmarking deferred.

---

## Validation workflow (all species)

1. **Download** raw data into `dataset/public_datasets/<species>/raw/` (or species-specific subdir).
2. **Preprocess** to BreedAI format (Geno.csv, Pheno.csv) — see [VALIDATION_WORKFLOW.md](VALIDATION_WORKFLOW.md).
3. **Run BreedAI:** `cd scripts` → `./start_menu.sh` → **1** (Phase 1) and/or **2** (Phase 2), with input data in `dataset/input/` (or path used by pipeline).
4. **Benchmark:** Compare BreedAI metrics (R², Pearson r, RMSE, MAE, bias) to published methods (e.g. GBLUP, Bayesian variable selection, ML) as in the cattle section and literature.

---

## Literature references

- **Van den Berg, I., et al. (2020).** Dryad 10.5061/dryad.rq80k — cattle benchmark.
- **Meuwissen et al. (2001).** *Genetics* 157, 1819–1829 — Bayesian alphabet.
- **VanRaden, P. M. (2008).** *J. Dairy Sci.* 91(11), 4414–4423 — G-matrix, GBLUP.
- **de los Campos et al. (2013).** *Genetics* 193, 327–345 — BGLR.
- **González-Recio et al. (2014).** *Front. Genet.* — ML in genomic selection.

(Add sheep/goat/camel/horse references as datasets are integrated.)

---

## Next steps

1. ✅ **Cattle (Van den Berg)** — in use; extend to more QTL/correlation scenarios as needed.
2. 📋 **Sheep (GSE152717)** — prepare BreedAI inputs; run Phase 1–2; document.
3. 📋 **Goats (GSE145419)** — same as sheep.
4. 📋 **Camels (Bitaraf Sani 2021)** — confirm repository or permission; then add and run.
5. 📋 **Horses** — use SNP panel for population genetics demo; add prediction benchmark when open phenotype data are available.

---

*Last updated: 2026-02*  
*BreedAI public datasets — multi-species benchmarking plan*

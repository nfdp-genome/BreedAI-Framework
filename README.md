# BreedAI

**An open-source framework for fair and standardized machine-learning-based
genomic prediction in livestock.**

Nuha BinTayyash, M. S. Alarawi, Aroob Abdullah Alhumaidy, Areej Almuhayya,
Mashael F. Alghuraybi, Othman I. Aljurayyad, Yara Altuwaijri, Saidan Mohammed
Alotaibi, Norah M. Alharbi, Muruj I. Tukruni, Alanoud T. Alharthi, Osama A.
Alshehri, Hend Alotaibi, Nouf F. Alharbi, Shahad A. Alsalman, Hatim Almutairi

*National Livestock and Fisheries Development Program (NLFDP), Ministry of
Environment, Water and Agriculture (MEWA), Kingdom of Saudi Arabia.*

📄 **Poster (ISMB 2026):** [`poster/BreedAI_poster_ISMB_2026.pdf`](poster/BreedAI_poster_ISMB_2026.pdf)
📖 **User Guide:** [`USER_GUIDE.md`](USER_GUIDE.md)

> This repository is the **public companion to the ISMB 2026 poster** — the
> proof-of-concept version of BreedAI benchmarked on a public simulated cattle
> dataset. It reproduces the poster's fair-benchmarking results end to end.

---

## What is BreedAI?

Genomic prediction is central to modern livestock breeding, yet workflows are
hard to compare, reproduce, and deploy. BreedAI provides a **unified, fair, and
reproducible** framework that holds preprocessing and data splits constant across
every model, so accuracy differences are interpretable and results are auditable.

**One shared backbone, two tracks:**

- **(A) Core dataset** — one shared backbone for every method: sample & SNP QC ·
  imputation · VanRaden G-matrix · fixed-effects modeling · seed-controlled
  60/20/20 split (optional pedigree → H-matrix / ssGBLUP).
- **(B) Default track** — literature-aligned **GBLUP** baseline → GEBV + reliability.
- **(C) R&D track** — **18 models** (Bayesian · linear · kernel · tree · neural) +
  stacking ensembles, on the *same* data & splits.
- **(D) Deployment** — new-animal prediction with SNP alignment, imputation, and
  overlap guardrails (warn < 80 %, reject < 50 %).

*Scope: sequencing (FASTQ → VCF) is upstream / out of scope; BreedAI starts at
genotypes.*

---

## Results — fair benchmarking on a public dataset

Public **Van den Berg simulated Holstein** cattle (QTL300, rg = 0.8). After QC:
**1,285 animals · 26,479 SNPs · 4 traits**; seed-controlled split 771 / 257 / 257.
Fixed effects: intercept-only (proof of concept).

| Category | Mean test Pearson r | Range |
|---|---|---|
| Default GBLUP (RidgeCV) | 0.886 | 0.835–0.912 |
| Best single model (penalized linear) | 0.911 | up to 0.956 |
| Best ensemble (stacking, non-neg. ridge) | **0.912** | up to **0.957** |
| Gain over GBLUP | **+3.0 %** | — |

Penalized linear models and stacking outperform the default GBLUP on moderately
polygenic traits; tree-based and neural methods underperform here — exactly the
kind of insight fair benchmarking is meant to surface. All metrics (R², Pearson r,
RMSE, MAE, bias) are reported per model in machine-readable CSV/JSON.

---

## How to run

BreedAI runs on **any** cohort — put your two input files in `input/` and launch
the pipeline. Preprocessing and data splits are held constant across every model,
so accuracy differences are interpretable and results are comparable.

```bash
# 1. Enter the repository — run every command below from the repo root
cd /path/to/BreedAI-Framework            # the directory you cloned

# 2. Environment (conda)
conda env create -f environment.yml      # creates the `genomic_pred` env
conda activate genomic_pred
#    …or, without conda:  pip install -r requirements.txt

# 3. Provide your data — drop two files into input/ (matched by animal ID):
#      input/Geno.csv     animals × SNPs, coded 0/1/2
#      input/Pheno.csv    animals × traits
#    Optional: input/metadata.csv (covariates), input/pedigree.csv (ssGBLUP)

# 4. Run the fair benchmark (default GBLUP + 18-model R&D track, same splits),
#    locally or on SLURM/HPC via the interactive menu:
cd scripts
./start_menu.sh
```

Per-model accuracy tables and figures are written under the run's results folder
(CSV/JSON). See the [`USER_GUIDE.md`](USER_GUIDE.md) for input formats, menu
options, and output locations.

> **HPC note:** before submitting, set your SLURM account once —
> `export SBATCH_ACCOUNT=<your-account>` (find it with
> `sacctmgr show assoc user=$USER format=account`). `sbatch` reads this variable
> natively, so it applies to every job; the menu also prompts for it. Leave it
> unset to use your cluster's default account. The scripts use the conda env
> `genomic_pred` and auto-detect the conda base (`conda info --base`).

---

## Reproduce the poster results

The repo ships a worked example — the public **Van den Berg** simulated Holstein
cattle dataset (Dryad DOI
[10.5061/dryad.rq80k](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rq80k)) —
so you can reproduce the poster end to end. Build its `Geno.csv` / `Pheno.csv`
from the shipped raw data and copy them into `input/`, then run as above:

```bash
# Run from the repo root (cd into it first — see "How to run" step 1).
# No arguments needed — the defaults reproduce the poster scenario (QTL300, r_g=0.8):
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py

cp cattle_dataset/processed/vandenberg_QTL300_rg8/Geno_QTL300_rg8.csv  input/Geno.csv
cp cattle_dataset/processed/vandenberg_QTL300_rg8/Pheno_QTL300_rg8.csv input/Pheno.csv
```

Full details — the dataset, the scenario, and how to build other replicates — are
in [`cattle_dataset/README.md`](cattle_dataset/README.md).

---

## Documentation

See the **[`USER_GUIDE.md`](USER_GUIDE.md)** for prerequisites, menu options, the two
tracks, the algorithm list, output locations, and troubleshooting. Method details and
the full write-up live in the poster (and, later, the paper).

---

## Citation

If you use BreedAI, please cite the ISMB 2026 poster:

> BinTayyash N, Alarawi MS, Alhumaidy AA, *et al.* **BreedAI: An open-source
> framework for fair and standardized machine-learning-based genomic prediction
> in livestock.** ISMB 2026 (poster). NLFDP, MEWA, Kingdom of Saudi Arabia.

## License

Released under the [MIT License](LICENSE) — © 2026 NLFDP / MEWA.

## Acknowledgement

We thank the **KAUST Supercomputing Core Laboratory** for access to the **Ibex**
high-performance computing cluster, on which all analyses were run.

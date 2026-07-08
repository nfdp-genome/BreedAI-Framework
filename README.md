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

## Reproduce it

```bash
# 1. Environment (conda)
conda env create -f environment.yml      # creates the `genomic_pred` env
conda activate genomic_pred
#    …or, without conda:  pip install -r requirements.txt

# 2. Get the data — Van den Berg et al. (2020), Dryad DOI: 10.5061/dryad.rq80k
#    See cattle_dataset/README.md for details, then
#    build the BreedAI Geno.csv / Pheno.csv:
python scripts/public_dataset/vandenberg/02_prepare_vandenberg.py   # see script header for args

# 3. Run the fair benchmark (default GBLUP + 18-model R&D track, same splits)
#    Locally or on SLURM/HPC via the menu:
bash scripts/start_menu.sh
```

Per-model accuracy tables and figures are written under the run's results folder
(CSV/JSON). See the [`USER_GUIDE.md`](USER_GUIDE.md).

> **HPC note:** the SLURM scripts read the conda env name from `genomic_pred` and
> auto-detect the conda base (`conda info --base`); set your own SLURM account
> (`YOUR_SLURM_ACCOUNT` placeholder in the configs) before submitting.

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

# BreedAI — methods summary (paper draft)

*Aligned with the repository as of 2026-03-30. Wording is conservative: it describes what the code does, not aspirational scope.*

---

## Study design and software

**BreedAI** is an open-source, policy- and config-driven framework for genomic prediction in livestock breeding. It is organized into **Phase 1 (learning and benchmarking)** and **Phase 2 (deployment-style prediction)** on a shared **core dataset** contract so that default and research tracks share identical preprocessing and evaluation splits when run from the same configuration.

Implementation is primarily Python 3 (NumPy, pandas, scikit-learn) with optional R-backed Bayesian alphabet models in the R&D track (`scripts/02a_phase1_train_validate_array.py`), SLURM job scripts for Ibex, and optional Nextflow pipelines for VCF-level QC (`pipelines/step4_genotype_qc_grm_ready/`).

---

## Data and benchmark

**Cattle proof of concept** uses the public Van den Berg et al. simulated Holstein dataset (Dryad doi:10.5061/dryad.rq80k), genotype-level entry (no FASTQ in this validation).

All metrics (default GBLUP and R&D benchmarking) are from a **single Phase 1 run** on the processed scenario **`vandenberg_QTL300_rg8`** (1,285 animals, 26,479 SNPs after QC, 4 traits). Both default and R&D tracks use the **same data, same QC, same 60/20/20 train/val/test splits**. Full results are in `docs/results/cattle_results_paper_table.md`.

---

## Preprocessing and quality control

1. **Sample / SNP QC** (genotype matrix path): thresholds include MAF, variance, and call-rate filters as implemented in `scripts/stages/s04_sample_snp_qc.py` and orchestrated by `scripts/05_build_core_dataset.py`.
2. **Genotype matrix**: validated 0/1/2 coding, sample alignment (`scripts/stages/s06_genotype_matrix.py`).
3. **Genomic relationship matrix**: VanRaden method via `scripts/stages/s07_grm.py` and `scripts/01a_utils_calculate_gmatrix.py`.
4. **Pedigree / H-matrix**: optional blending for ssGBLUP when pedigree is supplied (`scripts/stages/s08_h_matrix.py`); **skipped** for the cattle POC (no pedigree).
5. **Fixed effects**: intercept-only for the cattle POC (`scripts/stages/s09_fixed_effects.py`).

---

## Phase 1 — default (literature-aligned) track

**GBLUP** is implemented as **ridge regression on the genomic relationship kernel** (equivalently, ridge on a quadratic form of centered genotypes). Hyperparameters: **RidgeCV** alpha selection for the default POC (`scripts/models/gblup.py`, `scripts/stages/s10_default_gblup.py`).  

Outputs include train/validation/test metrics (e.g., R², Pearson *r*, RMSE, MAE, bias), per-trait summaries, and timing. **GEBV-style values and a CV-based reliability estimate** are computed in `scripts/stages/s11_gebv_reliability.py`.

---

## Phase 1 — optional R&D track

The R&D layer runs a **broader model zoo** (linear, penalized regression, trees, boosting, SVR, neural network, Bayesian alphabet models where available, etc.) and **simple ensembles** (average, median, weighted average) plus **non-negative ridge stacking on out-of-fold predictions** (`scripts/ensembles/stacking.py`), orchestrated by `scripts/02a_phase1_train_validate_array.py`.  

Nested cross-validation for stacking is implemented where enabled in that script; the exact flags should match the run configuration used to generate archived results.

---

## Phase 2 — deployment-style prediction

`scripts/08_deploy_default_track.py` loads training SNP order and allele frequencies from the Phase 1 default track, **aligns** new genotypes (reorder, drop extra SNPs, handle missing), optionally runs **Beagle** imputation when requested and when `BREEDAI_BEAGLE_JAR` (or discovered jar) is available, otherwise **mean-fills** missing genotypes from training frequencies.  

**SNP overlap guardrails** record overlap percentage and apply configurable **warning** and **rejection** thresholds; decisions are written to `prediction_report.json` and `model_card.json`.  

Operational testing uses `scripts/09_test_new_animal_prediction.py` (simulated held-out / perturbed genotypes).

---

## Reporting

- **Machine-readable** outputs (CSV, JSON) are the source of truth for metrics and deployment.
- **Jupyter notebooks** under `notebooks/` are human-readable summaries; they may be regenerated after runs.

---

## Limitations (must appear in discussion)

- **No end-to-end FASTQ-to-GEBV** validation in this milestone (upstream stages are scaffolds).
- **Cattle POC** is one public benchmark; sheep/goat pilots are planned, not reported here.
- **ssGBLUP** not validated on data with pedigree in this milestone.
- **Single-scenario validation**: current results are from one QTL/correlation scenario; generalizability across genetic architectures is not yet demonstrated.
- **Beagle** is optional and environment-dependent; mean-fill remains the portable fallback.

---

## Reproducibility

Recommended entry point:

```bash
cd scripts
./start_menu.sh
# Option 1: Phase 1 — Learning & Benchmarking
# Option 2: Phase 2 — Deployment & Prediction
```

Or non-interactive:

```bash
sbatch 02_phase1_train_validate.sh default_plus_rnd   # Phase 1
bash 05_phase2_predict_unified.sh                      # Phase 2
```

Exact environment (conda env `genomic_pred` on Ibex) and commit hash should be cited in the paper supplement.

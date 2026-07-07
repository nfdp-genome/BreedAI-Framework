# BreedAI — Abstract Outline

*Updated: 2026-04-01. Positioning: ISMB poster/presentation abstract first, then expanded full manuscript track (Nature-family target). Claims must match `docs/results/cattle_results_paper_table.md` and `docs/paper/paper_methods_summary.md`.*

---

## Title Options

1. **BreedAI: An open-source and fair framework for standardized genomic prediction in livestock breeding**

2. **BreedAI: Transparent and reproducible genomic prediction for livestock breeding with fair benchmarking and deployment guardrails**

3. **BreedAI: Standardizing livestock genomic prediction through open workflows, shared preprocessing, and transparent reporting**

4. **Open and fair genomic prediction in livestock breeding: the BreedAI framework for reproducible benchmarking and deployment**

---

## Novelty Statement (4 key points)

1. **Open-source transparency:** BreedAI is fully open and audit-friendly, with documented, reproducible steps rather than black-box prediction behavior.

2. **Fair and standardized evaluation:** all methods use one shared preprocessing backbone and identical train/validation/test splits (60/20/20), removing split/preprocessing confounds.

3. **Transparent reporting (current):** standardized outputs and notebook reports expose model behavior and metrics (R2, Pearson *r*, RMSE, MAE, bias) for each algorithm and ensemble.

4. **Standardized operational prediction:** Phase 2 deployment/prediction uses consistent VanRaden G-matrix conventions, reproducible seeds, SNP alignment, and overlap guardrails for safer use on new genotypes.

---

## Abstract (~250 words)

**Background.** Genomic prediction is central to modern livestock breeding, yet many workflows remain difficult to compare, reproduce, and operationalize. Production settings often rely on default GBLUP without broad benchmarking, while research studies use inconsistent preprocessing and split strategies that reduce fairness and traceability. A practical framework is needed that is open, fair, transparent, and standardized for both benchmarking and deployment.

**Methods.** We present BreedAI, an open-source genomic prediction framework built on four principles: transparency, fair standardization, interpretable reporting, and reproducibility. BreedAI enforces a shared preprocessing backbone (sample/SNP QC, VanRaden G-matrix conventions, fixed effects, and seed-based 60/20/20 train/validation/test splits), then runs a literature-aligned default GBLUP baseline (RidgeCV) and an optional R&D benchmark of 18 algorithms across 6 families with 4 ensemble methods including non-negative ridge stacking. Phase 2 supports prediction on new animals with SNP alignment, mean-fill or optional Beagle imputation, and overlap-based warn/reject guardrails. We validate the genotype-level workflow on the public Van den Berg simulated Holstein dataset (Dryad doi:10.5061/dryad.rq80k).

**Results.** On the QTL300, r_g=0.8 scenario (1,285 animals, 26,479 SNPs after QC, 4 traits), default GBLUP achieved test Pearson *r* = 0.835-0.912 (mean 0.886). Penalized linear models (LASSO_CV, ElasticNet_CV) outperformed GBLUP on moderately polygenic traits (*r* up to 0.956), while non-negative ridge stacking reached *r* = 0.957, improving mean accuracy to 0.913 (+3.0% vs default). Standardized reporting captured R2, Pearson *r*, RMSE, MAE, and bias across all models and ensembles, with reproducible notebook outputs.

**Conclusions.** BreedAI provides an open, fair, and transparent framework for standardized livestock genomic prediction, bridging reproducible benchmarking and deployment-ready inference.

---

## Safe Claims (supported by current results)

- Dual-mode design: **default GBLUP** vs **optional R&D** on a shared preprocessing backbone
- **18 algorithms** across 6 families + **4 ensemble methods** (including non-negative ridge stacking)
- All comparisons use **same data, same QC, same splits** — no preprocessing confounds
- Default GBLUP: test *r* = 0.835–0.912 (mean 0.886)
- Best R&D: test *r* = 0.839–0.956 (mean 0.911)
- Best ensemble: test *r* = 0.839–0.957 (mean 0.913, +3.0% over GBLUP)
- **Phase 2 deployment** with SNP alignment, mean-fill imputation, and overlap guardrails
- **Reproducible** via `start_menu.sh` on Ibex or direct script invocation
- **Honest scope:** genotype-level path validated; FASTQ stages are scaffolds

## Claims to Avoid

- "End-to-end **FASTQ** pipeline validated"
- "**ssGBLUP** validated on pedigree data"
- "**Universally best** accuracy vs all frameworks"
- "**Multi-species** validation complete" (cattle only so far)
- "**Beagle imputation** validated on all environments" (optional; jar-dependent)
- "**Selection index** results reported" (implemented but not exercised with real weights)
- "**Built-in model explainability/XAI is complete**" (feature-level explanation module not yet implemented)

## Mandatory Qualifiers

- Dataset is **simulated** cattle (Van den Berg), not real production data
- Scenario is **QTL300, r_g=0.8** — results may differ for other genetic architectures
- Phase 2 validation used **simulated** new animals (same-SNP full genotype), not an independent external lab chip
- Phase 2 simulated overlap scenarios (98%/60%/30%) have SNP alignment implemented but are not yet finalized (tests running)

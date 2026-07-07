# ISMB 2026 — BreedAI Submission Plan

## Target Venue

**ISMB 2026** — Intelligent Systems for Molecular Biology  
**Submission type:** Poster abstract (with presentation track if accepted)  
**Paper target:** Nature Methods / Nature Biotechnology (separate, longer timeline)

---

## Deliverables

### 1. Poster Abstract (ISMB)

| Item | Status |
|------|--------|
| Title | Draft ready — see `../abstract_outline.md` |
| Abstract (~250 words) | Draft ready — see `../abstract_outline.md` |
| Pipeline figure | Generated — see `../pipeline_figure_text.md` |
| Results table | Ready — see `../../results/cattle_results_abstract_table.md` |
| Submission deadline | April 4, 2026 |

### 2. Poster

| Item | Status |
|------|--------|
| Poster design | Pending |
| Key panels: pipeline figure, results table, key finding | Pending |
| Conference date | TBD |

### 3. Nature Paper

| Item | Status | Notes |
|------|--------|-------|
| Title | Draft | Choose from options in `../abstract_outline.md` |
| Abstract | Draft | Expand from ISMB version |
| Introduction | Not started | Gap: no unified framework for default + R&D genomic prediction |
| Methods | Draft ready | See `../paper_methods_summary.md` |
| Results — Cattle benchmark | Ready | See `../../results/cattle_results_paper_table.md` |
| Results — Phase 2 deployment | In progress | Full-genotype run pending confirmation |
| Results — Sheep benchmark | Not started | Next priority after abstract |
| Results — Goat benchmark | Not started | |
| Results — Saudi pilot | Not started | If data available |
| Discussion | Not started | |
| Figures | Partial | Pipeline figure done; results figures from notebooks |
| Supplementary | Not started | Full algorithm tables, stacking weights, reproducibility |
| Submission target | Nature Methods or Nature Biotechnology | |

---

## Key Results for Both Submissions

**Dataset:** Van den Berg et al. simulated Holstein cattle (1,285 animals, 26,479 SNPs, 4 traits)

| Category | Mean test Pearson *r* |
|----------|:--------------------:|
| Default GBLUP (RidgeCV) | 0.886 |
| Best R&D individual | 0.911 |
| Best ensemble (stacking) | 0.913 (+3.0%) |

**Key finding:** Penalized linear models and non-negative ridge stacking outperform GBLUP on moderately polygenic traits; tree-based and neural methods underperform on this dataset.

**Scope note for claims:** Current strength is openness, fairness, standardization, and transparent reporting. A dedicated feature-level explainability module (e.g., SHAP/permutation importance) is planned for the full-paper phase.

---

## Timeline

### ISMB Poster Abstract
- [x] Results tables generated
- [x] Abstract skeleton drafted
- [x] Pipeline figure generated
- [ ] Final abstract wording
- [ ] Final title
- [ ] Author list and affiliations
- [ ] Submit by April 4

### Nature Paper (after ISMB)
- [ ] Confirm Phase 2 deployment results
- [ ] Fix simulated overlap scenario bugs
- [ ] Sheep benchmark
- [ ] Goat benchmark
- [ ] Saudi pilot (if data available)
- [ ] Full methods section
- [ ] Full results section (multi-species)
- [ ] Discussion and conclusions
- [ ] Figures and supplementary
- [ ] Internal review
- [ ] Submit

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `README.md` | This plan |
| `poster_draft.pptx` | Poster layout (when created) |
| `ismb_abstract_final.md` | Final submitted abstract (when finalized) |
| `nature_paper_draft.md` | Paper manuscript (when started) |

---

## References

- Pipeline figure: `docs/paper/pipeline_figure_text.md`
- Abstract outline: `docs/paper/abstract_outline.md`
- Methods summary: `docs/paper/paper_methods_summary.md`
- Results (paper): `docs/results/cattle_results_paper_table.md`
- Results (abstract): `docs/results/cattle_results_abstract_table.md`
- Deployment validation: `docs/results/cattle_option2_deployment_table.md`

---

*Created: 2026-03-30*

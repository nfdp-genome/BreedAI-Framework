# BreedAI Documentation

**Last updated:** 2026-03-30

---

## Table of Contents

### User Guide


| Document                                         | Description                                                                                                    |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| [Getting Started](user_guide/getting_started.md) | How to run BreedAI: prerequisites, menu options, algorithms, output locations, repo structure, troubleshooting |


### Results


| Document                                                       | Description                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [Paper Table](results/cattle_results_paper_table.md)           | Full train/val/test metrics for default, best R&D, and best ensemble per trait |
| [Abstract Table](results/cattle_results_abstract_table.md)     | Compact test *r* table, interpretation, recommended abstract sentence          |
| [Deployment Table](results/cattle_option2_deployment_table.md) | Phase 2 simulated overlap test scenarios and status                            |
| [POC: Cattle Benchmark](results/poc_vandenberg.md)             | Dataset summary, reproduction steps, current results                           |


### Paper Drafts


| Document                                          | Description                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| [Abstract Outline](paper/abstract_outline.md)     | Title options, novelty statement, abstract skeleton, safe vs avoided claims |
| [Methods Summary](paper/paper_methods_summary.md) | Paper-ready methods text                                                    |
| [Pipeline Figure](paper/pipeline_figure_text.md)  | Figure caption, panel legend, color semantics                               |
| [ISMB 2026 Plan](paper/ISMB_2026/README.md)       | ISMB poster abstract + Nature paper submission plan and timeline            |


### Planning


| Document                                                                  | Description                                                              |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Current Status & Readiness](planning/00_current_status.md)               | What is done, what is pending, readiness checklists, known issues        |
| [Roadmap](planning/roadmap.md)                                            | Milestones (Now/Next/Later/Long-term), immediate actions, deferred items |
| [Abstract Submission Plan](planning/abstract_submission_april4.md)        | April 4 deadline timeline and deliverables                               |
| [Breeding Program Pipeline](planning/BREEDING_PROGRAM_PIPELINE_README.md) | Reference architecture                                                   |


---

## Quick Links


| I want to...                     | Go to                                                |
| -------------------------------- | ---------------------------------------------------- |
| Run BreedAI for the first time   | [Getting Started](user_guide/getting_started.md)     |
| See the cattle benchmark results | [Paper Table](results/cattle_results_paper_table.md) |
| Draft the abstract               | [Abstract Outline](paper/abstract_outline.md)        |
| Check what is done vs pending    | [Current Status](planning/00_current_status.md)      |
| See full roadmap and next steps  | [Roadmap](planning/roadmap.md)                       |


---

## Key Results

**Dataset:** Van den Berg et al. cattle (1,285 animals, 26,479 SNPs, 4 traits)


| Category                 | Mean test Pearson *r* |
| ------------------------ | --------------------- |
| Default GBLUP (RidgeCV)  | 0.886                 |
| Best R&D individual      | 0.911                 |
| Best ensemble (stacking) | 0.913                 |



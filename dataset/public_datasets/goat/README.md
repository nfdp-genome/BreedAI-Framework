## Goat public datasets

This folder contains public goat datasets for benchmarking.
Current dataset: **American Alpine dairy goats milk (GSE145419)**.

### Dataset description

- Source: NCBI GEO Series `GSE145419`
- Organism: *Capra hircus*
- Traits: milk yield and composition traits
- Genotyping: Illumina GoatIGGC_conf_60K BeadChip

### Reproducibility guide

1. **Keep original files**  
   Use the downloaded GEO files as-is and record file names and sizes.

2. **Match Phase 1 preprocessing**  
   Apply the same variance filter mask and missing‑data handling so the
   prediction feature space matches training.

3. **Confirm phenotype alignment**  
   Ensure `Animal_ID` alignment between genotype and phenotype inputs.

4. **Keep outputs**  
   Save Phase 1 QC and benchmarking reports for comparison across runs.

### Benchmarking with BreedAI

Run Phase 1 to benchmark all algorithms + ensembles and identify top models.
Run Phase 2 to deploy final models and produce prediction outputs.

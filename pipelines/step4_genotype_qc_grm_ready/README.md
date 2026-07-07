# Step 4: Genotype Filtration & QC for GRM Readiness

This isolated pipeline converts a cohort VCF into a **GRM-ready SNP dataset** and generates a reproducible **HTML + JSON QC report**.

## What it produces

- `results/step4/grm_ready.vcf.gz` + `.tbi`
- `results/step4/grm_ready.pgen/.pvar/.psam`
- `results/step4/pca_pruned.*`
- `results/step4/sample_qc.tsv`
- `results/step4/variant_qc.tsv`
- `results/step4/relatedness.tsv`
- `results/step4/pca_scores.tsv`
- `results/step4/pca_loadings.tsv`
- `results/reports/step4_genotype_qc_report.html`
- `results/reports/step4_genotype_qc_report.json`

## Example run

```bash
nextflow run pipelines/step4_genotype_qc_grm_ready/main.nf \
  -profile docker \
  --vcf results/step3/cohort.filtered_snps.vcf.gz \
  --metadata data/metadata.csv \
  --outdir results/step4 \
  --report_dir results/reports
```

## Quick dry run

```bash
nextflow run pipelines/step4_genotype_qc_grm_ready/main.nf \
  -profile test \
  --vcf results/step3/cohort.filtered_snps.vcf.gz
```

## Core parameters (cattle-friendly defaults)

| Parameter | Default | Notes |
|---|---:|---|
| `--min_sample_callrate` | `0.95` | sample keep threshold |
| `--min_snp_callrate` | `0.98` | SNP keep threshold |
| `--min_maf` | `0.01` | low-frequency SNP removal |
| `--enable_hwe` | `false` | HWE OFF by default for selected/structured cattle cohorts |
| `--hwe_p` | `1e-6` | used only when HWE enabled |
| `--ld_prune_window_kb` | `50` | LD prune window |
| `--ld_prune_step` | `5` | LD prune step |
| `--ld_prune_r2` | `0.2` | LD prune r² threshold |
| `--het_outlier_method` | `MAD` | `MAD` or `Z` |
| `--het_outlier_thresh` | `3.5` | robust outlier threshold |
| `--relatedness_method` | `KING` | `KING` or `PLINK` |
| `--duplicate_kinship_thresh` | `0.177` | relatedness/duplicate flag threshold |
| `--pca_k` | `10` | number of PCs |
| `--keep_chroms` | `autosomes` | `autosomes` or `all` |
| `--dry_run` | `false` | generate quick placeholders without heavy processing |

## QC interpretation guide

- **Sample flags**
  - `low_callrate`: sample call rate below threshold.
  - `het_outlier`: heterozygosity outlier (MAD/Z method).
- **Variant flags**
  - `low_callrate`: SNP call rate below threshold.
  - `low_maf`: MAF below threshold.
  - `hwe`: HWE p-value below threshold (only when enabled).
- **Relatedness flags**
  - pairs above `duplicate_kinship_thresh` are flagged as potential duplicates/close relatives.
- **PCA**
  - use PCA plots to inspect structure, breed clusters, and potential batch effects.

## Quality gates

Pipeline fails when:
- final VCF missing or not indexed
- final SNP count equals zero
- report HTML/JSON missing

If QC is too strict, relax:
- `--min_snp_callrate`
- `--min_maf`
- `--hwe_p` (or keep HWE disabled)

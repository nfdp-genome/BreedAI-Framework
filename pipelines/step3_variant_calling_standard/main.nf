nextflow.enable.dsl = 2

params.samplesheet = null
params.outdir = "results/step3"
params.dry_run = false

process DRY_RUN {
  publishDir params.outdir, mode: "copy"
  output:
  path "step3_dry_run.ok"
  script:
  """
  echo "step3 dry run" > step3_dry_run.ok
  """
}

process STEP3_PLACEHOLDER {
  publishDir params.outdir, mode: "copy"
  input:
  path samplesheet
  output:
  path "cohort.filtered_snps.vcf.gz"
  path "cohort.filtered_snps.vcf.gz.tbi"
  path "step3_variant_calling_report.html"
  path "step3_variant_calling_report.json"
  script:
  """
  # Placeholder emits tiny valid bgz-like file markers for wiring tests.
  printf '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n' | bgzip -c > cohort.filtered_snps.vcf.gz
  tabix -f -p vcf cohort.filtered_snps.vcf.gz
  cat > step3_variant_calling_report.html <<'HTML'
  <html><body><h1>Step3 Variant Calling Placeholder</h1></body></html>
  HTML
  cat > step3_variant_calling_report.json <<'JSON'
  {"status":"placeholder","caller":"gatk_gvcf_joint"}
  JSON
  """
}

workflow {
  if (!params.samplesheet) {
    error "Missing required parameter --samplesheet"
  }
  ch = Channel.fromPath(params.samplesheet, checkIfExists: true)
  if (params.dry_run) {
    DRY_RUN()
  } else {
    STEP3_PLACEHOLDER(ch)
  }
}


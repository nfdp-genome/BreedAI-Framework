nextflow.enable.dsl = 2

params_file = "${params.outdir}/run_params.json"

process DRY_RUN {
    publishDir "${params.outdir}", mode: 'copy'
    publishDir "${params.report_dir}", mode: 'copy'
    input:
    val run_params
    output:
    path "dry_run.ok"
    path "run_params.json"
    path "step4_genotype_qc_report.html"
    path "step4_genotype_qc_report.json"
    script:
    """
    echo "STEP4 dry run completed" > dry_run.ok
    cat > run_params.json <<'JSON'
    ${run_params}
    JSON
    cat > step4_genotype_qc_report.json <<'JSON'
    {"summary":{"mode":"dry_run"},"warnings":["No processing executed (dry_run=true)."]}
    JSON
    cat > step4_genotype_qc_report.html <<'HTML'
    <html><body><h1>Step4 Dry Run</h1><p>No processing executed.</p></body></html>
    HTML
    """
}

process TOOL_VERSIONS {
    publishDir "${params.outdir}", mode: 'copy'
    output:
    path "tool_versions.json"
    script:
    """
    python3 - <<'PY'
    import json, subprocess
    def ver(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip().split("\\n")[0]
        except Exception as e:
            return f"NA: {e}"
    out = {
      "bcftools": ver("bcftools --version"),
      "plink2": ver("plink2 --version"),
      "python": ver("python3 --version"),
      "nextflow": "${workflow.nextflow.version}",
    }
    try:
      out["king"] = ver("king --version")
    except Exception:
      out["king"] = "not_available"
    open("tool_versions.json","w").write(json.dumps(out, indent=2))
    PY
    """
}

process NORMALIZE_VCF {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path vcf
    output:
    path "normalized.vcf.gz"
    path "normalized.vcf.gz.tbi"
    script:
    def chromFilter = (params.keep_chroms == 'autosomes') ?
        "-e 'CHROM==\"X\" || CHROM==\"Y\" || CHROM==\"MT\" || CHROM==\"chrX\" || CHROM==\"chrY\" || CHROM==\"chrM\"'" : ""
    """
    set -euo pipefail
    bcftools view -m2 -M2 -v snps ${chromFilter} "$vcf" -Oz -o normalized.vcf.gz
    tabix -f -p vcf normalized.vcf.gz
    """
}

process VCF_TO_PGEN {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path vcf
    output:
    path "cohort.pgen"
    path "cohort.pvar"
    path "cohort.psam"
    script:
    """
    set -euo pipefail
    plink2 --vcf "$vcf" \
      --set-all-var-ids @:#:\\$r:\\$a \
      --new-id-max-allele-len 200 missing \
      --make-pgen \
      --out cohort
    """
}

process SAMPLE_MISSING {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "sample_missing.smiss"
    script:
    """
    plink2 --pfile cohort --missing sample-only --out sample_missing
    """
}

process SAMPLE_HET {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "sample_het.het"
    script:
    """
    plink2 --pfile cohort --het --out sample_het
    """
}

process VARIANT_MISSING {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "variant_missing.vmiss"
    script:
    """
    plink2 --pfile cohort --missing variant-only --out variant_missing
    """
}

process VARIANT_FREQ {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "variant_freq.afreq"
    script:
    """
    plink2 --pfile cohort --freq --out variant_freq
    """
}

process COMPUTE_HWE {
    publishDir "${params.outdir}", mode: 'copy'
    when:
    params.enable_hwe
    input:
    path pgen
    path pvar
    path psam
    output:
    path "variant_hwe.hardy"
    script:
    """
    plink2 --pfile cohort --hardy --out variant_hwe
    """
}

process EMPTY_HWE {
    output:
    path "variant_hwe.hardy"
    script:
    """
    echo -e "CHR\\tSNP\\tA1\\tA2\\tTEST\\tOBS_CT\\tP" > variant_hwe.hardy
    """
}

process BUILD_QC_FILTERS {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path sample_missing
    path sample_het
    path variant_missing
    path variant_freq
    path variant_hwe
    output:
    path "sample_qc.tsv"
    path "variant_qc.tsv"
    path "keep_samples.txt"
    path "keep_snps.txt"
    path "qc_filter_summary.json"
    script:
    def hweFlag = params.enable_hwe ? "--enable_hwe" : ""
    """
    python3 ${projectDir}/pipelines/step4_genotype_qc_grm_ready/report/build_qc_filters.py \
      --sample_missing "$sample_missing" \
      --sample_het "$sample_het" \
      --variant_missing "$variant_missing" \
      --variant_freq "$variant_freq" \
      --variant_hwe "$variant_hwe" \
      --sample_callrate ${params.min_sample_callrate} \
      --snp_callrate ${params.min_snp_callrate} \
      --min_maf ${params.min_maf} \
      ${hweFlag} \
      --hwe_p ${params.hwe_p} \
      --het_method ${params.het_outlier_method} \
      --het_thresh ${params.het_outlier_thresh} \
      --out_prefix qc_filters
    """
}

process APPLY_FILTERS {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    path keep_samples
    path keep_snps
    output:
    path "grm_ready.pgen"
    path "grm_ready.pvar"
    path "grm_ready.psam"
    path "grm_ready.vcf.gz"
    path "grm_ready.vcf.gz.tbi"
    script:
    """
    set -euo pipefail
    plink2 --pfile cohort --keep "$keep_samples" --extract "$keep_snps" --make-pgen --out grm_ready
    plink2 --pfile grm_ready --export vcf bgz --out grm_ready
    tabix -f -p vcf grm_ready.vcf.gz
    """
}

process LD_PRUNE {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "pca_pruned.pgen"
    path "pca_pruned.pvar"
    path "pca_pruned.psam"
    path "pca_pruned.prune.in"
    script:
    """
    plink2 --pfile grm_ready \
      --indep-pairwise ${params.ld_prune_window_kb}kb ${params.ld_prune_step} ${params.ld_prune_r2} \
      --out pca_pruned
    plink2 --pfile grm_ready --extract pca_pruned.prune.in --make-pgen --out pca_pruned
    """
}

process PCA {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "pca_scores.tsv"
    path "pca_loadings.tsv"
    path "pca.eigenval"
    script:
    """
    plink2 --pfile pca_pruned --pca approx ${params.pca_k} --out pca
    python3 - <<'PY'
    import pandas as pd
    import os
    e = pd.read_csv("pca.eigenvec", delim_whitespace=True)
    e.to_csv("pca_scores.tsv", sep="\\t", index=False)
    if os.path.exists("pca.eigenvec.allele"):
      a = pd.read_csv("pca.eigenvec.allele", delim_whitespace=True)
      a.to_csv("pca_loadings.tsv", sep="\\t", index=False)
    else:
      pd.DataFrame(columns=["ID","A1"]).to_csv("pca_loadings.tsv", sep="\\t", index=False)
    PY
    """
}

process RELATEDNESS {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    path pgen
    path pvar
    path psam
    output:
    path "relatedness.tsv"
    script:
    def method = params.relatedness_method.toString().toUpperCase()
    if (method == 'PLINK') {
      """
      plink2 --pfile grm_ready --genome --out relatedness_plink
      python3 - <<'PY'
      import pandas as pd
      g = pd.read_csv("relatedness_plink.genome", delim_whitespace=True)
      out = pd.DataFrame({
        "ID1": g["IID1"],
        "ID2": g["IID2"],
        "kinship": pd.NA,
        "PI_HAT": g["PI_HAT"],
        "source": "PLINK"
      })
      thr = float("${params.duplicate_kinship_thresh}")
      out["flag"] = out["PI_HAT"].fillna(0) >= thr
      out.to_csv("relatedness.tsv", sep="\\t", index=False)
      PY
      """
    } else {
      """
      plink2 --pfile grm_ready --make-king-table --out relatedness_king
      python3 - <<'PY'
      import pandas as pd
      k = pd.read_csv("relatedness_king.kin0", delim_whitespace=True)
      out = pd.DataFrame({
        "ID1": k.get("ID1", k.get("IID1")),
        "ID2": k.get("ID2", k.get("IID2")),
        "kinship": k.get("KINSHIP"),
        "PI_HAT": pd.NA,
        "source": "KING"
      })
      thr = float("${params.duplicate_kinship_thresh}")
      out["flag"] = out["kinship"].fillna(0) >= thr
      out.to_csv("relatedness.tsv", sep="\\t", index=False)
      PY
      """
    }
}

process WRITE_PARAMS {
    publishDir "${params.outdir}", mode: 'copy'
    input:
    val run_params
    output:
    path "run_params.json"
    script:
    """
    cat > run_params.json <<'JSON'
    ${run_params}
    JSON
    """
}

process MAKE_REPORT {
    publishDir "${params.report_dir}", mode: 'copy'
    input:
    path sample_qc
    path variant_qc
    path relatedness
    path pca_scores
    path pca_eigenval
    path qc_summary
    path versions
    path params_json
    path final_vcf
    path final_tbi
    path final_pgen
    path final_pvar
    path final_psam
    val metadata_path
    output:
    path "step4_genotype_qc_report.html"
    path "step4_genotype_qc_report.json"
    path "assets"
    script:
    def metaArg = metadata_path ? "--metadata ${metadata_path}" : ""
    """
    python3 ${projectDir}/pipelines/step4_genotype_qc_grm_ready/report/make_report.py \
      --sample_qc "$sample_qc" \
      --variant_qc "$variant_qc" \
      --relatedness "$relatedness" \
      --pca_scores "$pca_scores" \
      --pca_eigenval "$pca_eigenval" \
      ${metaArg} \
      --qc_summary_json "$qc_summary" \
      --versions_json "$versions" \
      --params_json "$params_json" \
      --final_vcf "$final_vcf" \
      --final_pgen_prefix grm_ready \
      --out_html step4_genotype_qc_report.html \
      --out_json step4_genotype_qc_report.json \
      --assets_dir assets
    """
}

process CHECK_OUTPUTS {
    input:
    path final_vcf
    path final_tbi
    path variant_qc
    path report_html
    path report_json
    output:
    path "quality_gate.ok"
    script:
    """
    set -euo pipefail
    test -s "$final_vcf" || { echo "ERROR: final VCF missing/empty."; exit 1; }
    test -s "$final_tbi" || { echo "ERROR: final VCF index missing."; exit 1; }
    test -s "$report_html" || { echo "ERROR: report HTML missing."; exit 1; }
    test -s "$report_json" || { echo "ERROR: report JSON missing."; exit 1; }
    n=\$(python3 - <<'PY'
import pandas as pd
v = pd.read_csv("${variant_qc}", sep="\\t")
print(int((~v["flagged"]).sum()))
PY
)
    if [ "\$n" -le 0 ]; then
      echo "ERROR: final SNP count is zero. Relax filters (min_snp_callrate/min_maf/hwe)." >&2
      exit 1
    fi
    echo "PASS" > quality_gate.ok
    """
}

workflow {
    def run_params = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(params))

    if (!params.vcf) {
        error "Missing required parameter: --vcf path/to/cohort.filtered_snps.vcf.gz"
    }

    if (params.dry_run) {
        DRY_RUN(run_params)
        return
    }

    ch_vcf = Channel.fromPath(params.vcf, checkIfExists: true)
    metadata_path = params.metadata ? file(params.metadata).toString() : null

    versions = TOOL_VERSIONS()
    params_json = WRITE_PARAMS(run_params)

    normalized = NORMALIZE_VCF(ch_vcf)
    cohort = VCF_TO_PGEN(normalized[0])

    smiss = SAMPLE_MISSING(cohort[0], cohort[1], cohort[2])
    shet = SAMPLE_HET(cohort[0], cohort[1], cohort[2])
    vmiss = VARIANT_MISSING(cohort[0], cohort[1], cohort[2])
    vfreq = VARIANT_FREQ(cohort[0], cohort[1], cohort[2])
    hwe = params.enable_hwe ? COMPUTE_HWE(cohort[0], cohort[1], cohort[2]) : EMPTY_HWE()

    qc = BUILD_QC_FILTERS(smiss, shet, vmiss, vfreq, hwe)
    filtered = APPLY_FILTERS(cohort[0], cohort[1], cohort[2], qc[2], qc[3])
    pruned = LD_PRUNE(filtered[0], filtered[1], filtered[2])
    pca = PCA(pruned[0], pruned[1], pruned[2])
    rel = RELATEDNESS(filtered[0], filtered[1], filtered[2])

    reports = MAKE_REPORT(
        qc[0], qc[1], rel, pca[0], pca[2], qc[4], versions, params_json,
        filtered[3], filtered[4], filtered[0], filtered[1], filtered[2], metadata_path
    )
    CHECK_OUTPUTS(filtered[3], filtered[4], qc[1], reports[0], reports[1])
}

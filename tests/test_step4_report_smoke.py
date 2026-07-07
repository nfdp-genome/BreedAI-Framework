import json
import subprocess
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_step4_report_smoke(tmp_path):
    sample_qc = tmp_path / "sample_qc.tsv"
    variant_qc = tmp_path / "variant_qc.tsv"
    relatedness = tmp_path / "relatedness.tsv"
    pca_scores = tmp_path / "pca_scores.tsv"
    pca_eigenval = tmp_path / "pca.eigenval"
    qc_summary = tmp_path / "qc_filter_summary.json"
    versions = tmp_path / "tool_versions.json"
    params = tmp_path / "run_params.json"
    final_vcf = tmp_path / "grm_ready.vcf.gz"
    final_vcf.write_text("placeholder")
    (tmp_path / "grm_ready.pgen").write_text("placeholder")
    (tmp_path / "grm_ready.pvar").write_text("placeholder")
    (tmp_path / "grm_ready.psam").write_text("placeholder")

    pd.DataFrame(
        [{"sample_id": "s1", "sample_callrate": 0.99, "het_f": 0.01, "flagged": False, "flag_reasons": ""}]
    ).to_csv(sample_qc, sep="\t", index=False)
    pd.DataFrame(
        [{"variant_id": "v1", "maf": 0.1, "variant_callrate": 0.99, "flag_low_callrate": False, "flag_low_maf": False, "flag_hwe": False, "flagged": False}]
    ).to_csv(variant_qc, sep="\t", index=False)
    pd.DataFrame([{"ID1": "s1", "ID2": "s2", "kinship": 0.01, "PI_HAT": 0.02}]).to_csv(
        relatedness, sep="\t", index=False
    )
    pd.DataFrame([{"#IID": "s1", "PC1": 0.1, "PC2": 0.2, "PC3": 0.3}]).to_csv(pca_scores, sep="\t", index=False)
    pca_eigenval.write_text("1.0\n0.5\n0.2\n")
    qc_summary.write_text(json.dumps({"sample_counts": {"raw": 1}, "variant_counts": {"raw": 1}}))
    versions.write_text(json.dumps({"plink2": "x"}))
    params.write_text(json.dumps({"duplicate_kinship_thresh": 0.177, "enable_hwe": False}))

    out_html = tmp_path / "report.html"
    out_json = tmp_path / "report.json"
    assets = tmp_path / "assets"
    cmd = [
        "python3",
        str(REPO_ROOT / "pipelines/step4_genotype_qc_grm_ready/report/make_report.py"),
        "--sample_qc",
        str(sample_qc),
        "--variant_qc",
        str(variant_qc),
        "--relatedness",
        str(relatedness),
        "--pca_scores",
        str(pca_scores),
        "--pca_eigenval",
        str(pca_eigenval),
        "--qc_summary_json",
        str(qc_summary),
        "--versions_json",
        str(versions),
        "--params_json",
        str(params),
        "--final_vcf",
        str(final_vcf),
        "--final_pgen_prefix",
        str(tmp_path / "grm_ready"),
        "--out_html",
        str(out_html),
        "--out_json",
        str(out_json),
        "--assets_dir",
        str(assets),
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    assert out_html.exists()
    assert out_json.exists()


#!/usr/bin/env python3
"""Generate Step4 HTML + JSON QC report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_metadata(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _plot_hist(series: pd.Series, title: str, out_png: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(series.dropna(), bins=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_pca(scores: pd.DataFrame, pcx: int, pcy: int, color_col: str | None, out_png: Path) -> None:
    plt.figure(figsize=(6, 5))
    xcol = f"PC{pcx}"
    ycol = f"PC{pcy}"
    if color_col and color_col in scores.columns:
        for key, grp in scores.groupby(color_col):
            plt.scatter(grp[xcol], grp[ycol], s=14, label=str(key), alpha=0.8)
        plt.legend(fontsize=7, loc="best")
    else:
        plt.scatter(scores[xcol], scores[ycol], s=14, alpha=0.8)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"PCA: {xcol} vs {ycol}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_qc", required=True)
    p.add_argument("--variant_qc", required=True)
    p.add_argument("--relatedness", required=True)
    p.add_argument("--pca_scores", required=True)
    p.add_argument("--pca_eigenval", default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--qc_summary_json", required=True)
    p.add_argument("--versions_json", required=True)
    p.add_argument("--params_json", required=True)
    p.add_argument("--final_vcf", required=True)
    p.add_argument("--final_pgen_prefix", required=True)
    p.add_argument("--out_html", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--assets_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample_qc, sep="\t")
    variant = pd.read_csv(args.variant_qc, sep="\t")
    related = pd.read_csv(args.relatedness, sep="\t")
    scores = pd.read_csv(args.pca_scores, sep="\t")
    meta = _parse_metadata(args.metadata)
    params = json.loads(Path(args.params_json).read_text())
    versions = json.loads(Path(args.versions_json).read_text())
    qc_counts = json.loads(Path(args.qc_summary_json).read_text())
    explained = []
    if args.pca_eigenval and Path(args.pca_eigenval).exists():
        vals = pd.read_csv(args.pca_eigenval, header=None).iloc[:, 0].astype(float).values
        tot = vals.sum()
        if tot > 0:
            explained = (vals / tot).tolist()

    if not meta.empty and "sample_id" in meta.columns:
        scores = scores.merge(meta, left_on="#IID", right_on="sample_id", how="left")

    # Sample plots
    callrate_png = assets_dir / "sample_callrate_hist.png"
    het_png = assets_dir / "sample_het_hist.png"
    _plot_hist(sample["sample_callrate"], "Sample Call Rate", callrate_png)
    _plot_hist(sample["het_f"], "Sample Heterozygosity (F)", het_png)

    # Variant plots
    maf_png = assets_dir / "variant_maf_hist.png"
    vcall_png = assets_dir / "variant_callrate_hist.png"
    _plot_hist(variant["maf"], "Variant MAF", maf_png)
    _plot_hist(variant["variant_callrate"], "Variant Call Rate", vcall_png)

    # PCA plots
    color_col = next((c for c in ["breed", "batch", "herd"] if c in scores.columns), None)
    pca12_png = assets_dir / "pca_pc1_pc2.png"
    pca23_png = assets_dir / "pca_pc2_pc3.png"
    if {"PC1", "PC2"}.issubset(scores.columns):
        _plot_pca(scores, 1, 2, color_col, pca12_png)
    if {"PC2", "PC3"}.issubset(scores.columns):
        _plot_pca(scores, 2, 3, color_col, pca23_png)

    # Heuristic warning: strong batch separation by PC1 ANOVA-like ratio
    warnings = []
    if (color_col == "batch") and ("PC1" in scores.columns):
        grouped = scores.dropna(subset=["batch"]).groupby("batch")["PC1"]
        if grouped.ngroups > 1:
            means = grouped.mean()
            global_var = scores["PC1"].var(ddof=0)
            between_var = means.var(ddof=0)
            if global_var > 0 and between_var / global_var > 0.4:
                warnings.append("Potential strong batch clustering signal on PC1.")

    # Relatedness flags
    rel_flags = []
    kin_thr = float(params.get("duplicate_kinship_thresh", 0.177))
    for _, r in related.iterrows():
        kin = pd.to_numeric(r.get("kinship", np.nan), errors="coerce")
        pi_hat = pd.to_numeric(r.get("PI_HAT", np.nan), errors="coerce")
        score = kin if np.isfinite(kin) else pi_hat
        if np.isfinite(score) and score >= kin_thr:
            rel_flags.append(
                {
                    "id1": r.get("ID1", r.get("IID1", "")),
                    "id2": r.get("ID2", r.get("IID2", "")),
                    "score": float(score),
                    "reason": "related_or_duplicate",
                }
            )
    if rel_flags:
        warnings.append(f"{len(rel_flags)} relatedness pairs flagged above threshold {kin_thr}.")

    flagged_samples = sample[sample["flagged"] == True]
    if len(flagged_samples) > 0:
        warnings.append(f"{len(flagged_samples)} samples flagged by call-rate/heterozygosity thresholds.")

    if Path(args.final_vcf).exists():
        final_vcf_size = Path(args.final_vcf).stat().st_size
    else:
        final_vcf_size = -1
    output_files = [
        args.final_vcf,
        args.final_pgen_prefix + ".pgen",
        args.final_pgen_prefix + ".pvar",
        args.final_pgen_prefix + ".psam",
        args.out_html,
        args.out_json,
    ]
    file_list = []
    for f in output_files:
        p = Path(f)
        file_list.append({"path": f, "size_bytes": p.stat().st_size if p.exists() else -1})

    report_json = {
        "summary": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _safe_git_commit(),
            "tool_versions": versions,
            "parameters": params,
            "sample_counts": qc_counts.get("sample_counts", {}),
            "variant_counts": {
                "raw": int(len(variant)),
                "after_callrate": int((~variant["flag_low_callrate"]).sum()),
                "after_maf": int((~(variant["flag_low_callrate"] | variant["flag_low_maf"])).sum()),
                "after_hwe": int((~variant["flagged"]).sum()),
                "final": int((~variant["flagged"]).sum()),
            },
        },
        "sample_flags": {
            row["sample_id"]: [x for x in str(row["flag_reasons"]).split(",") if x]
            for _, row in flagged_samples.iterrows()
        },
        "variant_counts": {
            "raw": int(len(variant)),
            "after_callrate": int((~variant["flag_low_callrate"]).sum()),
            "after_maf": int((~(variant["flag_low_callrate"] | variant["flag_low_maf"])).sum()),
            "after_hwe": int((~variant["flagged"]).sum()),
            "final": int((~variant["flagged"]).sum()),
        },
        "pca": {
            "scores_path": str(Path(args.pca_scores)),
            "color_by": color_col if color_col else "none",
            "explained_variance_path": str(args.pca_eigenval) if args.pca_eigenval else None,
            "explained_variance": explained,
        },
        "relatedness": {"flagged_pairs": rel_flags},
        "outputs": {
            "final_vcf": args.final_vcf,
            "final_vcf_size_bytes": final_vcf_size,
            "final_pgen_prefix": args.final_pgen_prefix,
            "report_html": args.out_html,
            "report_json": args.out_json,
            "files": file_list,
        },
        "warnings": warnings,
    }
    Path(args.out_json).write_text(json.dumps(report_json, indent=2))

    sample_tbl = sample.head(30).to_html(index=False)
    variant_tbl = variant.head(30).to_html(index=False)
    rel_tbl = related.head(30).to_html(index=False)
    files_tbl = pd.DataFrame(file_list).to_html(index=False)
    warns_html = "".join(f"<li>{w}</li>" for w in warnings) if warnings else "<li>None</li>"
    hwe_warning = ""
    if bool(params.get("enable_hwe", False)):
        hwe_warning = (
            "<div style='padding:8px;border:1px solid #c77;background:#fee;'>"
            "<b>HWE warning:</b> HWE filtering in selected/structured cattle populations may remove true signals."
            "</div>"
        )

    html = f"""
<html><head><title>Step4 Genotype QC Report</title></head>
<body>
<h1>Step 4: Genotype Filtration & QC for GRM Readiness</h1>
<h2>Run Summary</h2>
<pre>{json.dumps(report_json["summary"], indent=2)}</pre>
{hwe_warning}
<h2>Warnings</h2><ul>{warns_html}</ul>

<h2>Sample QC</h2>
<img src="{Path(args.assets_dir).name}/sample_callrate_hist.png" width="450"/>
<img src="{Path(args.assets_dir).name}/sample_het_hist.png" width="450"/>
{sample_tbl}

<h2>Variant QC</h2>
<img src="{Path(args.assets_dir).name}/variant_maf_hist.png" width="450"/>
<img src="{Path(args.assets_dir).name}/variant_callrate_hist.png" width="450"/>
{variant_tbl}

<h2>Population Structure</h2>
<img src="{Path(args.assets_dir).name}/pca_pc1_pc2.png" width="500"/>
<img src="{Path(args.assets_dir).name}/pca_pc2_pc3.png" width="500"/>

<h2>Relatedness</h2>
{rel_tbl}

<h2>Outputs</h2>
<ul>
<li>GRM-ready VCF: {args.final_vcf}</li>
<li>GRM-ready PLINK2 prefix: {args.final_pgen_prefix}</li>
<li>JSON report: {args.out_json}</li>
</ul>
{files_tbl}
</body></html>
"""
    Path(args.out_html).write_text(html)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build sample/snp keep lists and QC tables for Step4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mad_flags(values: pd.Series, thresh: float) -> pd.Series:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return pd.Series(False, index=values.index)
    score = np.abs(values - med) / mad
    return score > thresh


def _z_flags(values: pd.Series, thresh: float) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0:
        return pd.Series(False, index=values.index)
    z = (values - values.mean()) / std
    return z.abs() > thresh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_missing", required=True)
    p.add_argument("--sample_het", required=True)
    p.add_argument("--variant_missing", required=True)
    p.add_argument("--variant_freq", required=True)
    p.add_argument("--variant_hwe", default=None)
    p.add_argument("--sample_callrate", type=float, required=True)
    p.add_argument("--snp_callrate", type=float, required=True)
    p.add_argument("--min_maf", type=float, required=True)
    p.add_argument("--enable_hwe", action="store_true")
    p.add_argument("--hwe_p", type=float, default=1e-6)
    p.add_argument("--het_method", choices=["MAD", "Z"], default="MAD")
    p.add_argument("--het_thresh", type=float, default=3.5)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    smiss = pd.read_csv(args.sample_missing, delim_whitespace=True)
    het = pd.read_csv(args.sample_het, delim_whitespace=True)
    sample = smiss.merge(het[["#IID", "F"]], on="#IID", how="left")
    sample = sample.rename(columns={"#IID": "sample_id", "F_MISS": "sample_missingness", "F": "het_f"})
    sample["sample_callrate"] = 1.0 - sample["sample_missingness"]

    sample["flag_low_callrate"] = sample["sample_callrate"] < args.sample_callrate
    if args.het_method == "MAD":
        sample["flag_het_outlier"] = _mad_flags(sample["het_f"], args.het_thresh)
    else:
        sample["flag_het_outlier"] = _z_flags(sample["het_f"], args.het_thresh)
    sample["flagged"] = sample["flag_low_callrate"] | sample["flag_het_outlier"]
    sample["flag_reasons"] = sample.apply(
        lambda r: ",".join(
            [x for x in [
                "low_callrate" if r["flag_low_callrate"] else "",
                "het_outlier" if r["flag_het_outlier"] else "",
            ] if x]
        ),
        axis=1,
    )
    keep_sample = sample.loc[~sample["flagged"], ["sample_id"]]

    vmiss = pd.read_csv(args.variant_missing, delim_whitespace=True)
    afreq = pd.read_csv(args.variant_freq, delim_whitespace=True)
    var = vmiss.merge(afreq[["ID", "ALT_FREQS"]], on="ID", how="left")
    var = var.rename(columns={"ID": "variant_id", "F_MISS": "variant_missingness", "ALT_FREQS": "maf"})
    var["maf"] = pd.to_numeric(var["maf"], errors="coerce")
    var["variant_callrate"] = 1.0 - var["variant_missingness"]

    var["flag_low_callrate"] = var["variant_callrate"] < args.snp_callrate
    var["flag_low_maf"] = var["maf"] < args.min_maf
    var["flag_hwe"] = False
    if args.enable_hwe and args.variant_hwe and Path(args.variant_hwe).exists():
        hwe = pd.read_csv(args.variant_hwe, delim_whitespace=True)
        hwe = hwe.rename(columns={"ID": "variant_id", "P": "hwe_p"})
        var = var.merge(hwe[["variant_id", "hwe_p"]], on="variant_id", how="left")
        var["flag_hwe"] = var["hwe_p"] < args.hwe_p
    var["flagged"] = var["flag_low_callrate"] | var["flag_low_maf"] | var["flag_hwe"]
    var["flag_reasons"] = var.apply(
        lambda r: ",".join(
            [x for x in [
                "low_callrate" if r["flag_low_callrate"] else "",
                "low_maf" if r["flag_low_maf"] else "",
                "hwe" if r["flag_hwe"] else "",
            ] if x]
        ),
        axis=1,
    )
    keep_var = var.loc[~var["flagged"], ["variant_id"]]

    sample_tsv = out_prefix.with_name("sample_qc.tsv")
    variant_tsv = out_prefix.with_name("variant_qc.tsv")
    keep_samples_txt = out_prefix.with_name("keep_samples.txt")
    keep_snps_txt = out_prefix.with_name("keep_snps.txt")
    summary_json = out_prefix.with_name("qc_filter_summary.json")

    sample.to_csv(sample_tsv, sep="\t", index=False)
    var.to_csv(variant_tsv, sep="\t", index=False)
    keep_sample.to_csv(keep_samples_txt, sep="\t", index=False, header=False)
    keep_var.to_csv(keep_snps_txt, sep="\t", index=False, header=False)

    summary = {
        "sample_counts": {
            "raw": int(len(sample)),
            "kept": int(len(keep_sample)),
            "flagged": int(sample["flagged"].sum()),
        },
        "variant_counts": {
            "raw": int(len(var)),
            "after_callrate_maf_hwe": int(len(keep_var)),
            "flagged": int(var["flagged"].sum()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

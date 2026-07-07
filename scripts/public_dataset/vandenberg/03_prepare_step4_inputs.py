#!/usr/bin/env python3
"""Convert Vandenberg raw text format into BreedAI + Step4-ready inputs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--genotypes_txt", required=True, help="Path to Genotypes_26503SNPs.txt")
    p.add_argument("--phenotypes_txt", required=True, help="Path to Phenotypes_replicate_X.txt")
    p.add_argument("--id_breed_txt", default=None, help="Optional ID_Breed.txt")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--write_vcf", action="store_true", help="Write VCF + bgzip/tabix if available")
    p.add_argument("--chrom", default="1", help="Chromosome label for synthetic SNP map")
    return p.parse_args()


def load_genotypes(path: Path) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    geno_strings: list[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Unexpected genotype line format: {line[:120]}")
            sid, gstr = parts
            sample_ids.append(sid)
            geno_strings.append(gstr)

    if not geno_strings:
        raise ValueError("No genotype rows found.")
    n_snps = len(geno_strings[0])
    if any(len(x) != n_snps for x in geno_strings):
        raise ValueError("Inconsistent SNP-string lengths across samples.")

    # Convert "21001..." strings to 2D uint8 matrix.
    x = np.vstack([np.fromiter((ord(c) - 48 for c in s), dtype=np.uint8, count=n_snps) for s in geno_strings])
    return sample_ids, x


def load_phenotypes(path: Path) -> pd.DataFrame:
    ph = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if ph.shape[1] < 2:
        raise ValueError("Phenotype file must contain sample id + >=1 trait columns.")
    ph = ph.rename(columns={0: "Animal_ID"})
    trait_cols = {i: f"Trait_{i}" for i in range(1, ph.shape[1])}
    ph = ph.rename(columns=trait_cols)
    ph["Animal_ID"] = ph["Animal_ID"].astype(str)
    return ph


def load_breed(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["sample_id", "breed"])
    b = pd.read_csv(path, sep=r"\s+", header=None, names=["sample_id", "breed"], engine="python")
    b["sample_id"] = b["sample_id"].astype(str)
    return b


def write_vcf(path: Path, sample_ids: list[str], x: np.ndarray, chrom: str = "1") -> None:
    gt_map = {
        0: "0/0",
        1: "0/1",
        2: "1/1",
        3: "./.",  # just in case
        4: "./.",
        5: "./.",
        6: "./.",
        7: "./.",
        8: "./.",
        9: "./.",
    }
    with path.open("w", newline="") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + sample_ids
        f.write("\t".join(header) + "\n")
        n_snps = x.shape[1]
        for j in range(n_snps):
            snp_id = f"SNP_{j+1}"
            pos = str(j + 1)
            row = [chrom, pos, snp_id, "A", "C", ".", "PASS", ".", "GT"]
            row.extend(gt_map.get(int(v), "./.") for v in x[:, j])
            f.write("\t".join(row) + "\n")


def bgzip_and_index(vcf_plain: Path) -> Path:
    vcfgz = vcf_plain.with_suffix(vcf_plain.suffix + ".gz")
    subprocess.check_call(f'bgzip -f -c "{vcf_plain}" > "{vcfgz}"', shell=True)
    subprocess.check_call(f'tabix -f -p vcf "{vcfgz}"', shell=True)
    return vcfgz


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_ids, x = load_genotypes(Path(args.genotypes_txt))
    ph = load_phenotypes(Path(args.phenotypes_txt))
    breed = load_breed(Path(args.id_breed_txt) if args.id_breed_txt else None)

    geno_cols = [f"SNP_{i+1}" for i in range(x.shape[1])]
    geno_df = pd.DataFrame(x, index=sample_ids, columns=geno_cols)
    geno_df.index.name = "Animal_ID"
    geno_csv = outdir / "Geno.csv"
    geno_df.to_csv(geno_csv)

    # Align phenotype order to genotype order; keep only common ids.
    ph = ph[ph["Animal_ID"].isin(sample_ids)].copy()
    ph = ph.set_index("Animal_ID").reindex(sample_ids).reset_index()
    pheno_csv = outdir / "Pheno.csv"
    ph.to_csv(pheno_csv, index=False)

    # Metadata for coloring/stratification.
    meta = pd.DataFrame({"sample_id": sample_ids, "animal_id": sample_ids})
    if not breed.empty:
        meta = meta.merge(breed, on="sample_id", how="left")
    meta_csv = outdir / "metadata.csv"
    meta.to_csv(meta_csv, index=False)

    if args.write_vcf:
        step4_dir = outdir / "step4_input"
        step4_dir.mkdir(parents=True, exist_ok=True)
        vcf_plain = step4_dir / "cohort.filtered_snps.vcf"
        write_vcf(vcf_plain, sample_ids, x, chrom=args.chrom)
        try:
            vcfgz = bgzip_and_index(vcf_plain)
            print(f"VCF ready: {vcfgz}")
        except Exception as e:
            print(f"WARNING: bgzip/tabix failed ({e}). Plain VCF kept at {vcf_plain}")

    print(f"Wrote: {geno_csv}")
    print(f"Wrote: {pheno_csv}")
    print(f"Wrote: {meta_csv}")
    print(f"Samples: {len(sample_ids)}, SNPs: {x.shape[1]}, Traits: {ph.shape[1]-1}")


if __name__ == "__main__":
    main()


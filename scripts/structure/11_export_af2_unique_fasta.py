#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export unique WT sequences that need AF2/ColabFold prediction into FASTA format.

This script is intended to run after:
10_build_af2_fill_plan.py

Typical input:
data_processed/sa_prot_structures/linear_ratio1_af2_unique_sequences.csv

Typical output:
data_processed/sa_prot_structures/af2_unique_wt_sequences.fasta
data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv

Example
-------
python scripts/structure/11_export_af2_unique_fasta.py \
  --unique-sequences data_processed/sa_prot_structures/linear_ratio1_af2_unique_sequences.csv \
  --out-fasta data_processed/sa_prot_structures/af2_unique_wt_sequences.fasta \
  --out-manifest data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv

Then run ColabFold:
colabfold_batch \
  data_processed/sa_prot_structures/af2_unique_wt_sequences.fasta \
  raw_data/af2_predictions/linear_ratio1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unique-sequences",
        required=True,
        help="CSV from 10_build_af2_fill_plan.py, usually linear_ratio1_af2_unique_sequences.csv",
    )
    p.add_argument(
        "--out-fasta",
        required=True,
        help="Output FASTA path for AF2/ColabFold prediction",
    )
    p.add_argument(
        "--out-manifest",
        default=None,
        help="Optional manifest CSV path. If omitted, generated from out-fasta name.",
    )
    p.add_argument(
        "--header-prefix",
        default="af2seq",
        help="Prefix used in FASTA headers",
    )
    p.add_argument(
        "--wrap",
        type=int,
        default=80,
        help="FASTA line width",
    )
    return p.parse_args()


def clean_sequence(seq: object) -> str:
    if pd.isna(seq):
        return ""
    seq = str(seq).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    return seq


def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def safe_header_text(x: object, max_len: int = 80) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.:=,+-]", "_", s)
    return s[:max_len]


def main():
    args = parse_args()

    in_csv = Path(args.unique_sequences)
    out_fasta = Path(args.out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    if args.out_manifest is None:
        out_manifest = out_fasta.with_suffix(".manifest.csv")
    else:
        out_manifest = Path(args.out_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    required = {"seq_md5", "selected_wt_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Deduplicate again defensively.
    df["selected_wt_sequence"] = df["selected_wt_sequence"].map(clean_sequence)
    df = df[df["selected_wt_sequence"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["seq_md5", "selected_wt_sequence"]).copy()

    fasta_records = []
    manifest_rows = []

    for idx, row in df.reset_index(drop=True).iterrows():
        seq = row["selected_wt_sequence"]
        seq_md5 = str(row["seq_md5"])

        invalid = sorted(set(seq) - ALLOWED_AA)
        has_invalid = len(invalid) > 0

        n_assays = row.get("n_assays", "")
        assays = safe_header_text(row.get("assays", ""), max_len=100)
        missing_pos = safe_header_text(row.get("missing_dms_fasta_positions_union", ""), max_len=100)

        # Header deliberately starts with a short stable ID.
        # ColabFold output file names are derived from this header, so avoid overly long names.
        seq_id = f"{args.header_prefix}_{idx + 1:04d}_{seq_md5[:10]}"
        header = f">{seq_id} md5={seq_md5} n_assays={n_assays} missing_pos={missing_pos} assays={assays}"

        fasta_records.append(header)
        fasta_records.append(wrap_fasta(seq, args.wrap))

        manifest = row.to_dict()
        manifest.update({
            "af2_seq_id": seq_id,
            "sequence_length": len(seq),
            "has_invalid_aa": has_invalid,
            "invalid_aa": ",".join(invalid),
            "recommended_colabfold_input": str(out_fasta),
            "expected_output_dir": "raw_data/af2_predictions/linear_ratio1",
        })
        manifest_rows.append(manifest)

    out_fasta.write_text("\n".join(fasta_records) + "\n", encoding="utf-8")
    pd.DataFrame(manifest_rows).to_csv(out_manifest, index=False)

    print(f"[OK] Wrote FASTA: {out_fasta}")
    print(f"[OK] Wrote manifest: {out_manifest}")
    print(f"[INFO] Number of unique sequences: {len(manifest_rows)}")

    if any(r["has_invalid_aa"] for r in manifest_rows):
        bad = [r for r in manifest_rows if r["has_invalid_aa"]]
        print(f"[WARN] {len(bad)} sequences contain non-standard amino-acid letters.")
        print("       Check the manifest columns has_invalid_aa and invalid_aa before running ColabFold.")

    print("\nNext command:")
    print(f"colabfold_batch {out_fasta} raw_data/af2_predictions/linear_ratio1")


if __name__ == "__main__":
    main()

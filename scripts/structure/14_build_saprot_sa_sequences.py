#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15_build_saprot_sa_sequences.py

Read Foldseek 3Di outputs, filter valid 3Di result files, and build SaProt
structure-aware sequences.

This script includes the Foldseek-output validation step:
- scans foldseek_3di directory
- writes foldseek_3di_summary.csv
- filters valid files with:
    n_columns >= 3
    aa_len > 0
    3di_len > 0
    aa_len == 3di_len
- writes foldseek_3di_valid_summary.csv
- writes foldseek_3di_ignored_files.csv

Then it converts:
    WT amino-acid sequence + Foldseek 3Di sequence
into SaProt tokens:
    A + d -> Ad
    C + p -> Cp
    residue-only -> A#

Typical command
---------------
python scripts/structure/15_build_saprot_sa_sequences.py \
  --manifest data_processed/saprot_inputs/saprot_input_manifest.csv \
  --foldseek-dir data_processed/saprot_inputs/foldseek_3di \
  --mask-positions data_processed/saprot_inputs/saprot_mask_positions.tsv \
  --out-dir data_processed/saprot_inputs

Outputs
-------
foldseek_3di_summary.csv
foldseek_3di_valid_summary.csv
foldseek_3di_ignored_files.csv
saprot_sa_sequences.fasta
saprot_sa_sequences.tsv
saprot_token_by_residue.csv
saprot_build_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from Bio.Align import PairwiseAligner
except Exception:
    PairwiseAligner = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="data_processed/saprot_inputs/saprot_input_manifest.csv")
    p.add_argument("--foldseek-dir", required=True,
                   help="data_processed/saprot_inputs/foldseek_3di")
    p.add_argument("--mask-positions", default=None,
                   help="saprot_mask_positions.tsv. Positions here are forced to '#'.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory, usually data_processed/saprot_inputs")
    p.add_argument("--lower-3di", action="store_true", default=True,
                   help="Convert Foldseek 3Di characters to lowercase. Default: True.")
    p.add_argument("--no-lower-3di", action="store_false", dest="lower_3di",
                   help="Do not lowercase Foldseek 3Di characters.")
    p.add_argument("--write-spaced", action="store_true",
                   help="Also write space-separated token sequence column.")
    return p.parse_args()


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def scan_foldseek_outputs(foldseek_dir: str, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = Path(foldseek_dir)
    if not fold_dir.exists():
        raise FileNotFoundError(f"Foldseek directory not found: {foldseek_dir}")

    rows = []
    for f in sorted(fold_dir.iterdir()):
        if not f.is_file():
            continue

        try:
            text = f.read_text(errors="ignore").strip()
            if not text:
                rows.append({
                    "file": str(f),
                    "name": f.name,
                    "n_columns": 0,
                    "desc_head": "",
                    "aa_len": 0,
                    "3di_len": 0,
                    "length_match": False,
                    "aa_head": "",
                    "3di_head": "",
                    "parse_status": "empty_file",
                })
                continue

            line = text.splitlines()[0]
            parts = line.split("\t")
            desc = parts[0] if len(parts) > 0 else ""
            aa = parts[1] if len(parts) > 1 else ""
            st = parts[2] if len(parts) > 2 else ""

            rows.append({
                "file": str(f),
                "name": f.name,
                "n_columns": len(parts),
                "desc_head": desc[:80],
                "aa_len": len(aa),
                "3di_len": len(st),
                "length_match": len(aa) == len(st),
                "aa_head": aa[:30],
                "3di_head": st[:30],
                "parse_status": "ok",
            })
        except Exception as e:
            rows.append({
                "file": str(f),
                "name": f.name,
                "n_columns": -1,
                "desc_head": "",
                "aa_len": 0,
                "3di_len": 0,
                "length_match": False,
                "aa_head": "",
                "3di_head": "",
                "parse_status": f"error: {e}",
            })

    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out / "foldseek_3di_summary.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )

    valid = df[
        (df["n_columns"] >= 3) &
        (df["aa_len"] > 0) &
        (df["3di_len"] > 0) &
        (df["length_match"] == True)
    ].copy()

    ignored = df[~df.index.isin(valid.index)].copy()

    valid.to_csv(out / "foldseek_3di_valid_summary.csv", index=False)
    ignored.to_csv(
        out / "foldseek_3di_ignored_files.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )

    return df, valid


def read_valid_foldseek_file(path: str, lower_3di: bool = True) -> Tuple[str, str, str]:
    text = Path(path).read_text(errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty Foldseek output file: {path}")

    line = text.splitlines()[0]
    parts = line.split("\t")
    if len(parts) < 3:
        raise ValueError(f"Foldseek output has <3 columns: {path}")

    desc = parts[0]
    aa = clean_seq(parts[1]).upper()
    st = clean_seq(parts[2])
    if lower_3di:
        st = st.lower()

    if len(aa) != len(st):
        raise ValueError(f"AA/3Di length mismatch in {path}: {len(aa)} vs {len(st)}")

    return desc, aa, st


def infer_seq_id_from_name(name: str) -> Optional[str]:
    m = re.search(r"(saprot_\d{4}_[A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return None


def make_foldseek_lookup(valid: pd.DataFrame) -> Dict[str, str]:
    lookup = {}
    for _, r in valid.iterrows():
        seq_id = infer_seq_id_from_name(str(r["name"]))
        if not seq_id:
            continue
        path = str(r["file"])
        if seq_id in lookup:
            old = lookup[seq_id]
            old_size = os.path.getsize(old) if os.path.exists(old) else -1
            new_size = os.path.getsize(path) if os.path.exists(path) else -1
            if new_size > old_size:
                lookup[seq_id] = path
        else:
            lookup[seq_id] = path
    return lookup


def global_align_fold_to_wt(fold_aa: str, wt_seq: str) -> Tuple[str, str]:
    if fold_aa == wt_seq:
        return fold_aa, wt_seq

    if PairwiseAligner is None:
        raise RuntimeError("Bio.Align.PairwiseAligner is required for non-identical sequence alignment.")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    aln = aligner.align(fold_aa, wt_seq)[0]

    target_blocks, query_blocks = aln.aligned
    target = fold_aa
    query = wt_seq

    out_t = []
    out_q = []
    t_pos = 0
    q_pos = 0

    for (t_start, t_end), (q_start, q_end) in zip(target_blocks, query_blocks):
        if t_start > t_pos:
            out_t.append(target[t_pos:t_start])
            out_q.append("-" * (t_start - t_pos))
        if q_start > q_pos:
            out_t.append("-" * (q_start - q_pos))
            out_q.append(query[q_pos:q_start])

        out_t.append(target[t_start:t_end])
        out_q.append(query[q_start:q_end])

        t_pos = t_end
        q_pos = q_end

    if t_pos < len(target):
        out_t.append(target[t_pos:])
        out_q.append("-" * (len(target) - t_pos))
    if q_pos < len(query):
        out_t.append("-" * (len(query) - q_pos))
        out_q.append(query[q_pos:])

    return "".join(out_t), "".join(out_q)


def build_tokens_for_seq(seq_id: str, wt_seq: str, fold_aa: str, fold_3di: str, mask_positions: set[int]):
    wt_seq = clean_seq(wt_seq).upper()
    fold_aa = clean_seq(fold_aa).upper()
    fold_3di = clean_seq(fold_3di)

    aligned_fold, aligned_wt = global_align_fold_to_wt(fold_aa, wt_seq)

    rows = []
    wt_i = 0
    fold_i = 0
    n_match = 0
    n_mismatch = 0
    n_missing = 0
    n_masked = 0
    n_assigned = 0

    for fa, wa in zip(aligned_fold, aligned_wt):
        if fa != "-":
            fold_i += 1
            st = fold_3di[fold_i - 1]
        else:
            st = "#"

        if wa == "-":
            continue

        wt_i += 1
        wt_aa = wa.upper()

        token_source = "foldseek"
        final_st = st

        if fa == "-":
            final_st = "#"
            token_source = "missing_in_structure"
            n_missing += 1
        elif fa.upper() != wt_aa:
            final_st = "#"
            token_source = "aa_mismatch"
            n_mismatch += 1
        else:
            n_match += 1
            n_assigned += 1

        if wt_i in mask_positions:
            final_st = "#"
            token_source = "forced_mask_residue_only"
            n_masked += 1

        sa_token = wt_aa + final_st

        rows.append({
            "seq_id": seq_id,
            "wt_local_pos": wt_i,
            "wt_aa": wt_aa,
            "foldseek_aa": fa if fa != "-" else "",
            "structure_token": final_st,
            "sa_token": sa_token,
            "token_source": token_source,
        })

    if len(rows) != len(wt_seq):
        raise ValueError(f"{seq_id}: built {len(rows)} tokens but WT length is {len(wt_seq)}")

    summary = {
        "seq_id": seq_id,
        "wt_len": len(wt_seq),
        "foldseek_aa_len": len(fold_aa),
        "foldseek_3di_len": len(fold_3di),
        "n_match_positions": n_match,
        "n_assigned_structure_tokens": n_assigned,
        "n_missing_structure_positions": n_missing,
        "n_aa_mismatch_positions": n_mismatch,
        "n_forced_mask_positions": n_masked,
        "n_hash_tokens_final": sum(1 for r in rows if r["structure_token"] == "#"),
        "sa_token_len": len(rows),
        "status": "ok",
    }

    return rows, summary


def load_mask_positions(path: Optional[str]) -> Dict[str, set[int]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    df = pd.read_csv(p, sep="\t")
    if "seq_id" not in df.columns or "wt_local_pos" not in df.columns:
        return {}

    out = {}
    for seq_id, g in df.groupby("seq_id"):
        vals = set(pd.to_numeric(g["wt_local_pos"], errors="coerce").dropna().astype(int).tolist())
        out[str(seq_id)] = vals
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if "seq_id" not in manifest.columns or "selected_wt_sequence" not in manifest.columns:
        raise ValueError("Manifest must contain seq_id and selected_wt_sequence columns.")

    all_summary, valid_summary = scan_foldseek_outputs(args.foldseek_dir, args.out_dir)
    lookup = make_foldseek_lookup(valid_summary)
    mask_lookup = load_mask_positions(args.mask_positions)

    seq_rows = []
    token_rows_all = []
    build_summaries = []
    errors = []

    uniq = manifest[["seq_id", "selected_wt_sequence"]].drop_duplicates().copy()

    for _, r in uniq.iterrows():
        seq_id = str(r["seq_id"])
        wt_seq = clean_seq(r["selected_wt_sequence"]).upper()

        try:
            if seq_id not in lookup:
                raise FileNotFoundError(f"No valid Foldseek 3Di file found for seq_id={seq_id}")

            fold_file = lookup[seq_id]
            desc, fold_aa, fold_3di = read_valid_foldseek_file(fold_file, lower_3di=args.lower_3di)

            mask_pos = mask_lookup.get(seq_id, set())
            token_rows, summary = build_tokens_for_seq(
                seq_id=seq_id,
                wt_seq=wt_seq,
                fold_aa=fold_aa,
                fold_3di=fold_3di,
                mask_positions=mask_pos,
            )

            token_rows_all.extend(token_rows)

            sa_sequence = "".join(x["sa_token"] for x in token_rows)
            spaced = " ".join(x["sa_token"] for x in token_rows)

            row = {
                "seq_id": seq_id,
                "wt_length": len(wt_seq),
                "foldseek_file": fold_file,
                "foldseek_desc_head": desc[:80],
                "foldseek_aa_len": len(fold_aa),
                "foldseek_3di_len": len(fold_3di),
                "n_hash_tokens_final": summary["n_hash_tokens_final"],
                "hash_token_ratio": summary["n_hash_tokens_final"] / max(len(wt_seq), 1),
                "sa_sequence_compact": sa_sequence,
            }
            if args.write_spaced:
                row["sa_sequence_spaced"] = spaced
            seq_rows.append(row)

            build_summaries.append(summary)

        except Exception as e:
            errors.append({
                "seq_id": seq_id,
                "wt_length": len(wt_seq),
                "error": str(e),
            })
            build_summaries.append({
                "seq_id": seq_id,
                "wt_len": len(wt_seq),
                "status": "error",
                "error": str(e),
            })

    seq_df = pd.DataFrame(seq_rows)
    token_df = pd.DataFrame(token_rows_all)
    build_df = pd.DataFrame(build_summaries)

    seq_df.to_csv(out_dir / "saprot_sa_sequences.tsv", sep="\t", index=False)
    token_df.to_csv(out_dir / "saprot_token_by_residue.csv", index=False)
    build_df.to_csv(out_dir / "saprot_build_summary.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "saprot_build_errors.csv", index=False)

    with open(out_dir / "saprot_sa_sequences.fasta", "w") as f:
        for _, r in seq_df.iterrows():
            f.write(f">{r['seq_id']}\n")
            f.write(wrap_fasta(str(r["sa_sequence_compact"])) + "\n")

    with open(out_dir / "saprot_wt_sequences_used.fasta", "w") as f:
        for _, r in uniq.iterrows():
            f.write(f">{r['seq_id']}\n")
            f.write(wrap_fasta(clean_seq(r["selected_wt_sequence"]).upper()) + "\n")

    print("[OK] Wrote", out_dir / "foldseek_3di_summary.csv")
    print("[OK] Wrote", out_dir / "foldseek_3di_valid_summary.csv")
    print("[OK] Wrote", out_dir / "foldseek_3di_ignored_files.csv")
    print("[OK] Wrote", out_dir / "saprot_sa_sequences.fasta")
    print("[OK] Wrote", out_dir / "saprot_sa_sequences.tsv")
    print("[OK] Wrote", out_dir / "saprot_token_by_residue.csv")
    print("[OK] Wrote", out_dir / "saprot_build_summary.csv")

    if errors:
        print("[WARN] Some sequences failed. See", out_dir / "saprot_build_errors.csv")

    print("\nFoldseek files:")
    print("all files:", len(all_summary))
    print("valid 3Di files:", len(valid_summary))
    print("ignored sidecar/invalid files:", len(all_summary) - len(valid_summary))

    print("\nBuild status:")
    print(build_df["status"].value_counts(dropna=False))

    if len(build_df) > 0 and "n_hash_tokens_final" in build_df.columns:
        print("\nHash-token summary:")
        cols = ["seq_id", "wt_len", "foldseek_aa_len", "n_hash_tokens_final", "status"]
        print(build_df[[c for c in cols if c in build_df.columns]].head(30))


if __name__ == "__main__":
    main()

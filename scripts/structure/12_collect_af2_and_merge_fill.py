
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect ColabFold/AF2 outputs and merge AF2 per-residue pLDDT back to the
PDB-to-FASTA residue mapping table.

This script is intended to run after:
09_map_linear_ratio1_to_structure.py
10_build_af2_fill_plan.py
11_export_af2_unique_fasta.py
colabfold_batch ...

Typical command
---------------
python scripts/structure/12_collect_af2_and_merge_fill.py \
  --residue-mappings data_processed/sa_prot_structures/linear_ratio1_residue_mappings.csv \
  --fill-sites data_processed/sa_prot_structures/linear_ratio1_af2_fill_plan_sites.csv \
  --manifest data_processed/sa_prot_structures/af2_unique_wt_sequences_short_manifest.csv \
  --af2-dir raw_data/af2_predictions/linear_ratio1 \
  --out-dir data_processed/sa_prot_structures \
  --plddt-threshold 70

Outputs
-------
1) af2_prediction_summary.csv
   One row per AF2 sequence. Records rank_001 PDB/JSON paths and global confidence.

2) af2_residue_plddt.csv
   One row per predicted residue. Extracted mainly from rank_001 PDB B-factors.

3) final_structure_source_by_residue.csv
   Original residue mapping table plus AF2 pLDDT and final_source:
   - PDB: experimental structure has coordinates
   - AF2_fill: PDB missing, AF2 residue pLDDT >= threshold
   - residue_only: PDB missing and AF2 unavailable/low-confidence

Important coordinate convention
-------------------------------
AF2 sequence position is aligned to wt_local_pos, not necessarily fasta_pos.

If start_pos = 330:
    wt_local_pos = 1  -> fasta_pos = 330
    wt_local_pos = 15 -> fasta_pos = 344

ColabFold predicts the selected WT sequence itself, so AF2 residue index 1 maps to
wt_local_pos 1.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--residue-mappings", required=True,
                   help="linear_ratio1_residue_mappings.csv from step 09")
    p.add_argument("--fill-sites", required=True,
                   help="linear_ratio1_af2_fill_plan_sites.csv from step 10")
    p.add_argument("--manifest", required=True,
                   help="FASTA manifest from step 11, containing af2_seq_id and seq_md5")
    p.add_argument("--af2-dir", required=True,
                   help="ColabFold output directory")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plddt-threshold", type=float, default=70.0)
    return p.parse_args()


def find_first(patterns: List[str]) -> Optional[str]:
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits))
    return hits[0] if hits else None


def locate_rank1_files(af2_dir: str, af2_seq_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    ColabFold output examples:
    af2seq_0005_d8708e4776_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_000.pdb
    af2seq_0005_d8708e4776_scores_rank_001_alphafold2_ptm_model_5_seed_000.json
    """
    af2_dir = str(af2_dir)

    pdb_patterns = [
        os.path.join(af2_dir, f"{af2_seq_id}*unrelaxed_rank_001*.pdb"),
        os.path.join(af2_dir, f"{af2_seq_id}*relaxed_rank_001*.pdb"),
        os.path.join(af2_dir, "**", f"{af2_seq_id}*unrelaxed_rank_001*.pdb"),
        os.path.join(af2_dir, "**", f"{af2_seq_id}*relaxed_rank_001*.pdb"),
    ]

    json_patterns = [
        os.path.join(af2_dir, f"{af2_seq_id}*scores_rank_001*.json"),
        os.path.join(af2_dir, "**", f"{af2_seq_id}*scores_rank_001*.json"),
    ]

    return find_first(pdb_patterns), find_first(json_patterns)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_score_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    out = {}
    if isinstance(data, dict):
        for k in ["ptm", "pTM", "iptm", "ranking_confidence", "mean_plddt"]:
            if k in data:
                val = data[k]
                if isinstance(val, list):
                    continue
                out[k] = safe_float(val)

        # ColabFold often stores pLDDT as a list.
        plddt = None
        for key in ["plddt", "plddts"]:
            if key in data and isinstance(data[key], list):
                plddt = data[key]
                break
        if plddt:
            vals = [safe_float(v) for v in plddt]
            vals = [v for v in vals if v is not None]
            if vals:
                out["json_mean_plddt"] = sum(vals) / len(vals)
                out["json_n_plddt"] = len(vals)
    return out


def parse_pdb_residue_plddt(pdb_path: str, af2_seq_id: str) -> pd.DataFrame:
    """
    Extract per-residue pLDDT from PDB B-factor columns.
    Prefer CA atom if available. If CA is absent, use the first atom of the residue.
    """
    residues = {}
    order = []

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname3 = line[17:20].strip().upper()
            chain_id = line[21].strip() or "A"
            resseq_raw = line[22:26].strip()
            icode = line[26].strip()
            b_raw = line[60:66].strip()

            try:
                resseq = int(resseq_raw)
            except Exception:
                continue

            try:
                bfactor = float(b_raw)
            except Exception:
                continue

            key = (chain_id, resseq, icode)
            if key not in residues:
                residues[key] = {
                    "af2_seq_id": af2_seq_id,
                    "af2_chain_id": chain_id,
                    "af2_pdb_resseq": resseq,
                    "af2_icode": icode,
                    "af2_resname3": resname3,
                    "af2_aa": AA3_TO_1.get(resname3, "X"),
                    "plddt": bfactor,
                    "plddt_atom": atom_name,
                }
                order.append(key)
            elif atom_name == "CA":
                residues[key]["plddt"] = bfactor
                residues[key]["plddt_atom"] = atom_name

    rows = []
    for i, key in enumerate(order, start=1):
        rec = residues[key]
        rec["af2_res_index"] = i
        rec["af2_pdb_path"] = pdb_path
        rows.append(rec)

    return pd.DataFrame(rows)


def classify_source(row, threshold: float) -> str:
    if int(row.get("has_pdb_coord", 0)) == 1:
        return "PDB"

    plddt = row.get("af2_plddt")
    if pd.notna(plddt) and float(plddt) >= threshold:
        return "AF2_fill"

    return "residue_only"


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residue_map = pd.read_csv(args.residue_mappings)
    fill_sites = pd.read_csv(args.fill_sites)
    manifest = pd.read_csv(args.manifest)

    required_manifest = {"af2_seq_id", "seq_md5"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Map each DMS/chain entry to seq_md5 using fill plan, then to af2_seq_id using manifest.
    key_cols = [c for c in ["DMS_name", "pdb_id", "pdb_chain_id"] if c in fill_sites.columns and c in residue_map.columns]
    if not key_cols:
        key_cols = ["DMS_name"]

    dms_seq = fill_sites[key_cols + ["seq_md5"]].drop_duplicates().copy()
    dms_seq = dms_seq.merge(
        manifest[["seq_md5", "af2_seq_id", "sequence_length"]].drop_duplicates(),
        on="seq_md5",
        how="left",
    )

    # Collect AF2 rank_001 files and residue pLDDT.
    summary_rows = []
    residue_plddt_tables = []

    for _, row in manifest.drop_duplicates(subset=["af2_seq_id"]).iterrows():
        af2_seq_id = str(row["af2_seq_id"])
        pdb_path, json_path = locate_rank1_files(args.af2_dir, af2_seq_id)

        score_info = read_score_json(json_path)
        pdb_mean_plddt = None
        n_residues_from_pdb = 0

        if pdb_path:
            rdf = parse_pdb_residue_plddt(pdb_path, af2_seq_id)
            n_residues_from_pdb = len(rdf)
            if len(rdf):
                pdb_mean_plddt = float(rdf["plddt"].mean())
                residue_plddt_tables.append(rdf)

        summary = {
            "af2_seq_id": af2_seq_id,
            "seq_md5": row.get("seq_md5"),
            "sequence_length_manifest": row.get("sequence_length"),
            "rank1_pdb_path": pdb_path,
            "rank1_score_json_path": json_path,
            "n_residues_from_pdb": n_residues_from_pdb,
            "pdb_mean_plddt": pdb_mean_plddt,
            "json_mean_plddt": score_info.get("json_mean_plddt"),
            "json_n_plddt": score_info.get("json_n_plddt"),
            "ptm": score_info.get("ptm") or score_info.get("pTM"),
            "iptm": score_info.get("iptm"),
            "ranking_confidence": score_info.get("ranking_confidence"),
            "has_rank1_pdb": pdb_path is not None,
            "has_rank1_json": json_path is not None,
        }
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "af2_prediction_summary.csv", index=False)

    if residue_plddt_tables:
        af2_res = pd.concat(residue_plddt_tables, ignore_index=True)
    else:
        af2_res = pd.DataFrame(columns=[
            "af2_seq_id", "af2_res_index", "af2_aa", "plddt", "af2_pdb_path"
        ])

    af2_res = af2_res.rename(columns={"plddt": "af2_plddt"})
    af2_res.to_csv(out_dir / "af2_residue_plddt.csv", index=False)

    # Merge AF2 ID back to all residue mapping rows.
    merged = residue_map.merge(dms_seq, on=key_cols, how="left")

    # AF2 residue index corresponds to wt_local_pos.
    if "wt_local_pos" not in merged.columns:
        raise ValueError("residue_mappings must contain wt_local_pos.")

    merged["af2_res_index"] = pd.to_numeric(merged["wt_local_pos"], errors="coerce").astype("Int64")

    af2_for_merge = af2_res[[
        "af2_seq_id", "af2_res_index", "af2_aa", "af2_plddt",
        "af2_chain_id", "af2_pdb_resseq", "af2_icode", "af2_pdb_path"
    ]].copy()

    final_df = merged.merge(
        af2_for_merge,
        on=["af2_seq_id", "af2_res_index"],
        how="left",
    )

    final_df["plddt_threshold"] = args.plddt_threshold
    final_df["final_source"] = final_df.apply(lambda r: classify_source(r, args.plddt_threshold), axis=1)

    # Useful diagnostic flags.
    final_df["af2_available_for_position"] = final_df["af2_plddt"].notna().astype(int)
    final_df["af2_pass_plddt"] = (
        final_df["af2_plddt"].notna() &
        (final_df["af2_plddt"].astype(float) >= args.plddt_threshold)
    ).astype(int)

    final_df.to_csv(out_dir / "final_structure_source_by_residue.csv", index=False)

    # DMS-site-level summary.
    if "is_dms_site" in final_df.columns:
        dms_sites = final_df[final_df["is_dms_site"] == 1].copy()
        if len(dms_sites):
            dms_summary = (
                dms_sites.groupby(["DMS_name", "final_source"], dropna=False)
                .size()
                .reset_index(name="n_dms_sites")
                .pivot_table(index="DMS_name", columns="final_source", values="n_dms_sites", fill_value=0)
                .reset_index()
            )
            dms_summary.columns.name = None
            dms_summary.to_csv(out_dir / "final_structure_source_dms_site_summary.csv", index=False)

    print("[OK] Wrote:", out_dir / "af2_prediction_summary.csv")
    print("[OK] Wrote:", out_dir / "af2_residue_plddt.csv")
    print("[OK] Wrote:", out_dir / "final_structure_source_by_residue.csv")
    if (out_dir / "final_structure_source_dms_site_summary.csv").exists():
        print("[OK] Wrote:", out_dir / "final_structure_source_dms_site_summary.csv")

    print("\nQuick counts:")
    print(final_df["final_source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

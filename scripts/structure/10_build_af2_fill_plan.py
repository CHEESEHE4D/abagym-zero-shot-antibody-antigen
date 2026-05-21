
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an AF2 fill plan after PDB/mmCIF -> WT FASTA residue mapping.

This script reads:
1) linear_ratio1_residue_mappings.csv   (site-level output from step 09)
2) linear_ratio1_mapping_summary.csv    (assay-level output from step 09)
3) dms_reference_with_pdb.csv           (reference metadata)

It then answers four practical questions:
- Which assays already have full PDB coverage on all DMS sites?
- Which assays still miss some DMS sites in the experimental structure?
- Exactly which WT FASTA positions need AF2 supplementation?
- Which WT sequences are duplicated across assays, so AF2 can be downloaded/predicted once and reused?

Outputs
-------
1) linear_ratio1_af2_fill_plan_sites.csv
   Site-level list of DMS positions that are missing in PDB and should be checked in AF2.

2) linear_ratio1_af2_fill_plan_assays.csv
   Assay-level summary of AF2 need.

3) linear_ratio1_no_af2_needed_assays.csv
   Assays whose DMS sites are already fully covered by experimental structure.

4) linear_ratio1_af2_unique_sequences.csv
   Unique WT sequences among assays needing AF2, for deduplicated AF2 download/prediction.

Typical usage
-------------
python scripts/structure/10_build_af2_fill_plan.py \
  --residue-mappings data_processed/sa_prot_structures/linear_ratio1_residue_mappings.csv \
  --mapping-summary data_processed/sa_prot_structures/linear_ratio1_mapping_summary.csv \
  --dms-ref data_processed/dms_reference_with_pdb.csv \
  --out-dir data_processed/sa_prot_structures
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--residue-mappings", required=True,
                   help="Path to linear_ratio1_residue_mappings.csv")
    p.add_argument("--mapping-summary", required=True,
                   help="Path to linear_ratio1_mapping_summary.csv")
    p.add_argument("--dms-ref", required=True,
                   help="Path to dms_reference_with_pdb.csv")
    p.add_argument("--out-dir", required=True,
                   help="Output directory")
    return p.parse_args()


def extract_start_pos(notes: object) -> int:
    if pd.isna(notes):
        return 1
    m = re.search(r"start_pos=(\d+)", str(notes))
    return int(m.group(1)) if m else 1


def choose_wt_sequence(row: pd.Series) -> str:
    best = str(row.get("best_wt_sequence", "")).strip()
    if best and best.lower() != "nan":
        return best
    return str(row.get("wt_sequence", "")).strip()


def md5_of_seq(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()[:12]


def safe_join_unique(values, sep=";"):
    vals = []
    for x in values:
        if pd.isna(x):
            continue
        s = str(x).strip()
        if not s or s.lower() == "nan":
            continue
        vals.append(s)
    return sep.join(sorted(set(vals)))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residue = pd.read_csv(args.residue_mappings)
    summary = pd.read_csv(args.mapping_summary)
    ref = pd.read_csv(args.dms_ref)

    # Basic validation
    required_residue_cols = {
        "DMS_name", "pdb_id", "pdb_chain_id", "wt_local_pos", "fasta_pos",
        "wt_aa", "is_dms_site", "has_pdb_coord", "mapping_state"
    }
    missing_cols = required_residue_cols - set(residue.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in residue mappings: {sorted(missing_cols)}"
        )

    # Keep one reference row per (DMS_name, pdb_chain_id) if possible.
    # If duplicates still exist, keep the first one deterministically.
    ref = ref.copy()
    ref["start_pos"] = ref["notes"].apply(extract_start_pos)
    ref["selected_wt_sequence"] = ref.apply(choose_wt_sequence, axis=1)
    ref["wt_length_from_ref"] = ref["selected_wt_sequence"].str.len()

    ref_key_cols = [
        c for c in [
            "DMS_name", "pdb_id", "pdb_chain_id", "abagym_chain_label", "chains_raw",
            "source", "status", "notes", "start_pos",
            "selected_wt_sequence", "wt_length_from_ref",
            "matched_sites", "unmatched_sites", "pdb_file_status"
        ] if c in ref.columns
    ]
    ref_dedup = (
        ref[ref_key_cols]
        .sort_values([c for c in ["DMS_name", "pdb_chain_id", "pdb_id"] if c in ref.columns])
        .drop_duplicates(subset=[c for c in ["DMS_name", "pdb_chain_id"] if c in ref.columns], keep="first")
    )

    # Merge residue mappings with metadata
    join_keys = [c for c in ["DMS_name", "pdb_id", "pdb_chain_id"] if c in residue.columns and c in ref_dedup.columns]
    residue_full = residue.merge(ref_dedup, on=join_keys, how="left")

    # ---- 1) Site-level AF2 fill plan ----
    # DMS positions that exist in WT FASTA but have no coordinate in experimental structure
    fill_sites = residue_full[
        (residue_full["is_dms_site"] == 1) &
        (residue_full["has_pdb_coord"] == 0)
    ].copy()

    fill_sites["selected_wt_sequence"] = fill_sites["selected_wt_sequence"].fillna("")
    fill_sites["wt_length_from_mapping"] = (
        fill_sites.groupby(["DMS_name", "pdb_chain_id"])["wt_local_pos"].transform("max")
    )
    fill_sites["seq_md5"] = fill_sites["selected_wt_sequence"].apply(
        lambda s: md5_of_seq(s) if s else ""
    )
    fill_sites["af2_action"] = "check_or_download_af2_then_fill_if_confident"
    fill_sites["af2_rule"] = "use AF2 only if pLDDT passes threshold; otherwise keep residue_only"
    fill_sites["why_need_af2"] = "DMS site missing in experimental structure"
    fill_sites["site_label"] = fill_sites["wt_aa"].astype(str) + fill_sites["fasta_pos"].astype(int).astype(str)

    site_cols = [
        c for c in [
            "DMS_name", "pdb_id", "pdb_chain_id", "abagym_chain_label", "chains_raw",
            "start_pos", "wt_local_pos", "fasta_pos", "wt_aa", "site_label",
            "mapping_state", "why_need_af2",
            "selected_wt_sequence", "wt_length_from_ref", "wt_length_from_mapping", "seq_md5",
            "matched_sites", "unmatched_sites",
            "af2_action", "af2_rule"
        ] if c in fill_sites.columns
    ]
    fill_sites = fill_sites[site_cols].sort_values(["DMS_name", "fasta_pos", "pdb_chain_id"])

    # ---- 2) Assay-level AF2 need summary ----
    # Start from mapping summary because it already has useful PDB coverage counts.
    summary_full = summary.merge(ref_dedup, on=[c for c in ["DMS_name", "pdb_id", "pdb_chain_id"] if c in summary.columns and c in ref_dedup.columns], how="left")

    need_counts = (
        fill_sites.groupby([c for c in ["DMS_name", "pdb_id", "pdb_chain_id"] if c in fill_sites.columns], dropna=False)
        .agg(
            n_dms_sites_need_af2=("fasta_pos", "nunique"),
            missing_dms_fasta_positions=("fasta_pos", lambda s: ",".join(map(str, sorted(set(map(int, s)))))),
            missing_dms_labels=("site_label", lambda s: ",".join(sorted(set(map(str, s)))))
        )
        .reset_index()
    )

    assay_plan = summary_full.merge(
        need_counts,
        on=[c for c in ["DMS_name", "pdb_id", "pdb_chain_id"] if c in summary_full.columns and c in need_counts.columns],
        how="left"
    )

    assay_plan["n_dms_sites_need_af2"] = assay_plan["n_dms_sites_need_af2"].fillna(0).astype(int)
    assay_plan["missing_dms_fasta_positions"] = assay_plan["missing_dms_fasta_positions"].fillna("")
    assay_plan["missing_dms_labels"] = assay_plan["missing_dms_labels"].fillna("")
    assay_plan["selected_wt_sequence"] = assay_plan["selected_wt_sequence"].fillna("")
    assay_plan["seq_md5"] = assay_plan["selected_wt_sequence"].apply(
        lambda s: md5_of_seq(s) if s else ""
    )

    assay_plan["af2_need"] = assay_plan["n_dms_sites_need_af2"].apply(
        lambda n: "yes" if n > 0 else "no"
    )
    assay_plan["recommended_next_step"] = assay_plan["af2_need"].map({
        "yes": "retrieve_or_predict_af2_for_this_WT_sequence_and_check_only_missing_DMS_sites",
        "no": "no_af2_needed_for_dms_sites"
    })

    assay_cols = [
        c for c in [
            "DMS_name", "pdb_id", "pdb_chain_id", "abagym_chain_label", "chains_raw",
            "structure_file",
            "wt_length", "atom_length", "start_pos",
            "n_positions_total", "n_with_pdb_coord", "n_missing_in_structure", "n_mismatch_coord",
            "n_dms_sites", "n_dms_sites_with_coord", "dms_site_coord_coverage", "full_seq_coord_coverage",
            "n_dms_sites_need_af2", "missing_dms_fasta_positions", "missing_dms_labels",
            "selected_wt_sequence", "wt_length_from_ref", "seq_md5",
            "af2_need", "recommended_next_step"
        ] if c in assay_plan.columns
    ]
    assay_plan = assay_plan[assay_cols].sort_values(["af2_need", "DMS_name", "pdb_chain_id"], ascending=[False, True, True])

    # ---- 3) No-AF2-needed subset ----
    no_af2 = assay_plan[assay_plan["af2_need"] == "no"].copy()

    # ---- 4) Unique sequence table for deduplicated AF2 retrieval/prediction ----
    need_af2 = assay_plan[assay_plan["af2_need"] == "yes"].copy()

    if len(need_af2) > 0:
        uniq_seq = (
            need_af2.groupby("seq_md5", dropna=False)
            .agg(
                wt_length=("selected_wt_sequence", lambda s: len(next((x for x in s if isinstance(x, str) and x), ""))),
                selected_wt_sequence=("selected_wt_sequence", lambda s: next((x for x in s if isinstance(x, str) and x), "")),
                n_assays=("DMS_name", "nunique"),
                assays=("DMS_name", safe_join_unique),
                pdb_ids=("pdb_id", safe_join_unique),
                pdb_chain_ids=("pdb_chain_id", safe_join_unique),
                missing_dms_fasta_positions_union=("missing_dms_fasta_positions", safe_join_unique),
                missing_dms_labels_union=("missing_dms_labels", safe_join_unique),
            )
            .reset_index()
            .sort_values(["n_assays", "wt_length"], ascending=[False, False])
        )
        uniq_seq["af2_reuse_note"] = "same WT sequence can reuse one AF2 structure across listed assays"
    else:
        uniq_seq = pd.DataFrame(columns=[
            "seq_md5", "wt_length", "selected_wt_sequence", "n_assays", "assays",
            "pdb_ids", "pdb_chain_ids", "missing_dms_fasta_positions_union",
            "missing_dms_labels_union", "af2_reuse_note"
        ])

    # Write files
    fill_sites.to_csv(out_dir / "linear_ratio1_af2_fill_plan_sites.csv", index=False)
    assay_plan.to_csv(out_dir / "linear_ratio1_af2_fill_plan_assays.csv", index=False)
    no_af2.to_csv(out_dir / "linear_ratio1_no_af2_needed_assays.csv", index=False)
    uniq_seq.to_csv(out_dir / "linear_ratio1_af2_unique_sequences.csv", index=False)

    # Small console summary
    print(f"[OK] site-level AF2 fill plan written: {out_dir / 'linear_ratio1_af2_fill_plan_sites.csv'}")
    print(f"[OK] assay-level AF2 plan written:     {out_dir / 'linear_ratio1_af2_fill_plan_assays.csv'}")
    print(f"[OK] no-AF2-needed table written:      {out_dir / 'linear_ratio1_no_af2_needed_assays.csv'}")
    print(f"[OK] unique WT sequences written:      {out_dir / 'linear_ratio1_af2_unique_sequences.csv'}")
    print()
    print(f"Assays needing AF2: {need_af2['DMS_name'].nunique()}")
    print(f"Unique WT sequences needing AF2: {uniq_seq['seq_md5'].nunique() if len(uniq_seq) else 0}")
    print(f"Missing DMS sites to check in AF2: {fill_sites.shape[0]}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map WT sequence positions to PDB/mmCIF residue coordinates for assays with:
- mapping_method == linear
- mapping_ratio == 1
- status == ok

Inputs
------
1) dms_reference_with_pdb.csv
2) site_mapping_summary.csv
3) structure directory containing .pdb / .cif / .mmcif files

Main idea
---------
For each assay:
- read WT sequence and selected pdb_chain_id
- extract ATOM-sequence from the structure chain
- globally align ATOM-sequence to WT sequence
- build a per-position table on the WT sequence coordinate system:
    wt_local_pos, fasta_pos, wt_aa, has_pdb_coord, pdb_resseq, icode, ...
- positions present in WT but absent in ATOM alignment are treated as
  "missing_in_structure" and can later be filled by AF2

Example
-------
python scripts/structure/08_map_linear_ratio1_to_structure.py \
  --dms-ref data_processed/dms_reference_with_pdb.csv \
  --site-summary data_processed/site_mapping_summary.csv \
  --struct-dir raw_data/pdb_structs \
  --out-dir data_processed/sa_prot_structures

Outputs
-------
1) linear_ratio1_residue_mappings.csv
2) linear_ratio1_mapping_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import pairwise2
from Bio.PDB import MMCIFParser, PDBParser, Polypeptide


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # common selenium-methionine fallback
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dms-ref", required=True, help="Path to dms_reference_with_pdb.csv")
    p.add_argument("--site-summary", required=True, help="Path to site_mapping_summary.csv")
    p.add_argument("--struct-dir", required=True, help="Directory with pdb/cif/mmcif files")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--gap-open", type=float, default=-10.0)
    p.add_argument("--gap-extend", type=float, default=-0.5)
    return p.parse_args()


def extract_start_pos(notes: str) -> int:
    if pd.isna(notes):
        return 1
    m = re.search(r"start_pos=(\d+)", str(notes))
    return int(m.group(1)) if m else 1


def parse_site_list(x: object) -> set[int]:
    if pd.isna(x) or str(x).strip() == "":
        return set()
    vals = []
    for s in str(x).split(","):
        s = s.strip()
        if not s:
            continue
        try:
            vals.append(int(s))
        except ValueError:
            pass
    return set(vals)


def locate_structure_file(struct_dir: str, pdb_id_value: str) -> Optional[str]:
    """
    Try multiple patterns.
    Priority:
    1) exact stem startswith full pdb_id_value
    2) fallback to trailing 4-char pdb code
    """
    struct_dir = str(struct_dir)
    candidates = []

    for ext in ("*.cif", "*.mmcif", "*.pdb", "*.ent"):
        candidates.extend(glob.glob(os.path.join(struct_dir, f"{pdb_id_value}{ext[1:]}")))
        candidates.extend(glob.glob(os.path.join(struct_dir, f"{pdb_id_value}*{ext[1:]}")))

    if candidates:
        return sorted(set(candidates))[0]

    # fallback: last token after underscore, if it looks like a 4-char pdb code
    tail = str(pdb_id_value).split("_")[-1]
    if re.fullmatch(r"[0-9A-Za-z]{4}", tail):
        for ext in ("*.cif", "*.mmcif", "*.pdb", "*.ent"):
            candidates.extend(glob.glob(os.path.join(struct_dir, f"*{tail}*{ext[1:]}")))
        if candidates:
            return sorted(set(candidates))[0]

    return None


def load_structure(struct_path: str):
    ext = Path(struct_path).suffix.lower()
    if ext in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("x", struct_path)


def residue_to_aa(residue) -> Optional[str]:
    hetflag, resseq, icode = residue.id
    if hetflag.strip() not in {"", "W"}:
        return None
    resname = residue.get_resname().upper()
    return AA3_TO_1.get(resname)


def extract_chain_atom_residues(struct_path: str, chain_id: str) -> List[Dict]:
    ext = Path(struct_path).suffix.lower()

    # 先尝试 auth chain
    if ext in {".cif", ".mmcif"}:
        parsers = [
            ("auth", MMCIFParser(QUIET=True, auth_chains=True)),
            ("label", MMCIFParser(QUIET=True, auth_chains=False)),
        ]
    else:
        parsers = [("pdb", PDBParser(QUIET=True))]

    last_available = None

    for mode, parser in parsers:
        structure = parser.get_structure("x", struct_path)
        model = next(structure.get_models())

        chains = {c.id: c for c in model.get_chains()}
        last_available = list(chains.keys())

        if chain_id in chains:
            chain = chains[chain_id]
            rows = []
            chain_local_pos = 0
            for residue in chain.get_residues():
                aa = residue_to_aa(residue)
                if aa is None:
                    continue
                chain_local_pos += 1
                hetflag, resseq, icode = residue.id
                rows.append({
                    "chain_local_pos": chain_local_pos,
                    "atom_aa": aa,
                    "pdb_resseq": int(resseq),
                    "icode": "" if icode == " " else str(icode),
                    "resname3": residue.get_resname().upper(),
                    "chain_id_mode": mode,
                })
            return rows

    raise ValueError(
        f"Chain '{chain_id}' not found in {os.path.basename(struct_path)}. "
        f"Available chains (last tried mode): {last_available}"
    )

def global_align(atom_seq: str, wt_seq: str, gap_open: float, gap_extend: float):
    aligns = pairwise2.align.globalms(atom_seq, wt_seq, 2, -1, gap_open, gap_extend)
    if not aligns:
        raise ValueError("No alignment produced.")
    # best alignment
    return aligns[0]


def build_mapping_for_one(
    row: pd.Series,
    gap_open: float,
    gap_extend: float,
    struct_dir: str,
) -> Tuple[pd.DataFrame, Dict]:
    dms_name = row["DMS_name"]
    pdb_id = row["pdb_id"]
    pdb_chain_id = row["pdb_chain_id"]
    wt_seq = str(row["best_wt_sequence"] if pd.notna(row.get("best_wt_sequence")) and str(row.get("best_wt_sequence")).strip() else row["wt_sequence"]).strip()
    start_pos = extract_start_pos(row.get("notes", ""))
    dms_sites = parse_site_list(row.get("matched_sites"))

    struct_path = locate_structure_file(struct_dir, pdb_id)
    if struct_path is None:
        raise FileNotFoundError(f"No structure file found for pdb_id={pdb_id}")

    atom_rows = extract_chain_atom_residues(struct_path, pdb_chain_id)
    atom_seq = "".join(r["atom_aa"] for r in atom_rows)

    print(f"[DEBUG] {dms_name} | pdb={pdb_id} | chain={pdb_chain_id} | atom_len={len(atom_seq)} | wt_len={len(wt_seq)}")
    print(f"[DEBUG] atom_head={atom_seq[:60]}")
    print(f"[DEBUG] wt_head={wt_seq[:60]}")

    if len(atom_seq) == 0:
        raise ValueError(
            f"Empty atom sequence after residue filtering. "
            f"Chain exists but no recognized standard residues were extracted."
        )

    aln = global_align(atom_seq, wt_seq, gap_open, gap_extend)
    atom_aln, wt_aln, score, begin, end = aln

    records = []
    atom_i = 0
    wt_i = 0
    matched_aa = 0
    mismatch_aa = 0
    missing_in_structure = 0

    for a, w in zip(atom_aln, wt_aln):
        atom_info = None
        if a != "-":
            atom_i += 1
            atom_info = atom_rows[atom_i - 1]
        if w != "-":
            wt_i += 1
            fasta_pos = start_pos + wt_i - 1

            rec = {
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "structure_file": os.path.basename(struct_path),
                "pdb_chain_id": pdb_chain_id,
                "start_pos": start_pos,
                "wt_local_pos": wt_i,
                "fasta_pos": fasta_pos,
                "wt_aa": w,
                "is_dms_site": int(fasta_pos in dms_sites),
                "has_pdb_coord": 0,
                "mapping_state": "missing_in_structure",
                "atom_chain_local_pos": pd.NA,
                "atom_aa": pd.NA,
                "pdb_resseq": pd.NA,
                "icode": pd.NA,
                "resname3": pd.NA,
            }

            if a == "-":
                missing_in_structure += 1
            else:
                rec["has_pdb_coord"] = 1
                rec["atom_chain_local_pos"] = atom_info["chain_local_pos"]
                rec["atom_aa"] = atom_info["atom_aa"]
                rec["pdb_resseq"] = atom_info["pdb_resseq"]
                rec["icode"] = atom_info["icode"]
                rec["resname3"] = atom_info["resname3"]

                if a == w:
                    rec["mapping_state"] = "matched_coord"
                    matched_aa += 1
                else:
                    rec["mapping_state"] = "mismatch_coord"
                    mismatch_aa += 1

            records.append(rec)

    out_df = pd.DataFrame(records)

    summary = {
        "DMS_name": dms_name,
        "pdb_id": pdb_id,
        "pdb_chain_id": pdb_chain_id,
        "structure_file": os.path.basename(struct_path),
        "wt_length": len(wt_seq),
        "atom_length": len(atom_seq),
        "start_pos": start_pos,
        "n_positions_total": len(out_df),
        "n_with_pdb_coord": int((out_df["has_pdb_coord"] == 1).sum()),
        "n_missing_in_structure": int((out_df["mapping_state"] == "missing_in_structure").sum()),
        "n_mismatch_coord": int((out_df["mapping_state"] == "mismatch_coord").sum()),
        "n_dms_sites": int(out_df["is_dms_site"].sum()),
        "n_dms_sites_with_coord": int(((out_df["is_dms_site"] == 1) & (out_df["has_pdb_coord"] == 1)).sum()),
        "dms_site_coord_coverage": round(
            ((out_df["is_dms_site"] == 1) & (out_df["has_pdb_coord"] == 1)).sum() / max(int(out_df["is_dms_site"].sum()), 1), 4
        ),
        "full_seq_coord_coverage": round((out_df["has_pdb_coord"] == 1).sum() / max(len(out_df), 1), 4),
    }
    return out_df, summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dms_ref = pd.read_csv(args.dms_ref)
    site_sum = pd.read_csv(args.site_summary)

    keep = site_sum[
        (site_sum["mapping_method"] == "linear") &
        (site_sum["mapping_ratio"] == 1) &
        (site_sum["status"] == "ok")
    ][["DMS_name"]].drop_duplicates()

    df = dms_ref.merge(keep, on="DMS_name", how="inner").copy()
    df = df[(df["status"] == "ok") & (df["pdb_file_status"].isin(["exists", "downloaded (cif)"]))].copy()

    all_maps = []
    all_summaries = []
    errors = []

    for _, row in df.iterrows():
        try:
            mapping_df, summary = build_mapping_for_one(
                row=row,
                gap_open=args.gap_open,
                gap_extend=args.gap_extend,
                struct_dir=args.struct_dir,
            )
            all_maps.append(mapping_df)
            all_summaries.append(summary)
            print(f"[OK] {row['DMS_name']}: {summary['n_with_pdb_coord']}/{summary['wt_length']} WT positions have coordinates")
        except Exception as e:
            errors.append({
                "DMS_name": row["DMS_name"],
                "pdb_id": row["pdb_id"],
                "pdb_chain_id": row["pdb_chain_id"],
                "error": str(e),
            })
            print(f"[ERR] {row['DMS_name']}: {e}")

    if all_maps:
        pd.concat(all_maps, ignore_index=True).to_csv(out_dir / "linear_ratio1_residue_mappings.csv", index=False)
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(out_dir / "linear_ratio1_mapping_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "linear_ratio1_mapping_errors.csv", index=False)

    print("\nDone.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

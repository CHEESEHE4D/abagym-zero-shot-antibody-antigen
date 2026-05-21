#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_run_saprot_zero_shot.py

Run SaProt zero-shot mutational effect prediction using prepared structure-aware
sequences from step 15.

This script expects that you have already generated:
- saprot_sa_sequences.tsv
- saprot_input_manifest.csv
- final_structure_source_by_residue.csv

It loads SaProt's official mutation model:
    from model.saprot.saprot_foldseek_mutation_model import SaprotFoldseekMutationModel

and calls:
    model.predict_mut(sa_sequence, mut_info)

where:
    sa_sequence = compact structure-aware SaProt sequence, e.g. "MdKc..."
    mut_info    = mutation in WT-local coordinate, e.g. "V3A" or "V3A:Q4M"

Typical command
---------------
python scripts/structure/16_run_saprot_zero_shot.py \
  --saprot-repo /public/home/huangwenle/tools/SaProt \
  --model-dir /public/home/huangwenle/models/SaProt_650M_AF2 \
  --variant-glob "data_processed/esm_inputs/*.csv" \
  --sa-sequences data_processed/saprot_inputs/saprot_sa_sequences.tsv \
  --manifest data_processed/saprot_inputs/saprot_input_manifest.csv \
  --position-map data_processed/sa_prot_structures/final_structure_source_by_residue.csv \
  --out-dir data_processed/saprot_scores \
  --device cuda

If your input variants are in one file:
python scripts/structure/16_run_saprot_zero_shot.py \
  --saprot-repo /path/to/SaProt \
  --model-dir /path/to/SaProt_650M_AF2 \
  --variant-file data_processed/abagym_clean.csv \
  --sa-sequences data_processed/saprot_inputs/saprot_sa_sequences.tsv \
  --manifest data_processed/saprot_inputs/saprot_input_manifest.csv \
  --position-map data_processed/sa_prot_structures/final_structure_source_by_residue.csv \
  --out-dir data_processed/saprot_scores \
  --device cuda

Important coordinate convention
-------------------------------
SaProt's predict_mut uses the position within the SaProt input sequence:
    wt_local_pos = 1, 2, 3, ...

Your DMS mutation may be written using the reference FASTA coordinate:
    fasta_pos = start_pos + wt_local_pos - 1

Therefore this script uses final_structure_source_by_residue.csv to convert:
    DMS_name + fasta_pos -> wt_local_pos

If a variant file already uses local WT positions, pass:
    --input-position-system local

Otherwise default is:
    --input-position-system fasta

Outputs
-------
1) One scored CSV per input file:
   <out-dir>/<input_stem>_saprot_scores.csv

2) Combined scored file:
   <out-dir>/saprot_scores_all.csv

3) Error file:
   <out-dir>/saprot_score_errors.csv

4) Run summary:
   <out-dir>/saprot_score_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


EXPERIMENTAL_CANDIDATES = [
    "experimental_score", "DMS_score", "dms_score", "fitness",
    "score", "target", "bind", "binding", "phenotype"
]

MUTATION_CANDIDATES = [
    "mutation", "mutant", "variant", "substitution", "mut_info"
]

POSITION_CANDIDATES = [
    "fasta_pos", "site_raw", "site", "position", "pos", "mut_pos", "residue_index"
]

WT_CANDIDATES = [
    "wt_aa", "wildtype", "wild_type", "wild", "WT", "wt"
]

MUT_AA_CANDIDATES = [
    "mut_aa", "mutant_aa", "mutation_aa", "mut", "Mut"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--saprot-repo", required=True,
                   help="Path to cloned westlake-repl/SaProt repository.")
    p.add_argument("--model-dir", required=True,
                   help="Directory path of SaProt model, e.g. SaProt_650M_AF2. Not the .pt file.")
    p.add_argument("--variant-file", action="append", default=[],
                   help="One variant CSV file. Can be used multiple times.")
    p.add_argument("--variant-glob", action="append", default=[],
                   help="Glob pattern for variant CSV files. Can be used multiple times.")
    p.add_argument("--sa-sequences", required=True,
                   help="saprot_sa_sequences.tsv from step 15.")
    p.add_argument("--manifest", required=True,
                   help="saprot_input_manifest.csv from step 13.")
    p.add_argument("--position-map", required=True,
                   help="final_structure_source_by_residue.csv from step 12.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--input-position-system", default="fasta", choices=["fasta", "local"],
                   help="Whether mutation positions in input files are fasta_pos or wt_local_pos.")
    p.add_argument("--dms-col", default="DMS_name")
    p.add_argument("--mutation-col", default="auto")
    p.add_argument("--pos-col", default="auto")
    p.add_argument("--wt-col", default="auto")
    p.add_argument("--mut-col", default="auto")
    p.add_argument("--experimental-col", default="auto")
    p.add_argument("--batch-log-every", type=int, default=500)
    p.add_argument("--allow-aa-mismatch", action="store_true",
                   help="If set, do not fail when input WT AA differs from mapped WT AA; still use mapped WT AA.")
    return p.parse_args()


def setup_saprot(repo_path: str):
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"SaProt repo not found: {repo}")
    sys.path.insert(0, str(repo))

    try:
        from model.saprot.saprot_foldseek_mutation_model import SaprotFoldseekMutationModel
    except Exception as e:
        raise ImportError(
            "Failed to import SaprotFoldseekMutationModel. "
            "Check --saprot-repo points to the cloned SaProt repository and dependencies are installed."
        ) from e

    return SaprotFoldseekMutationModel


def load_model(SaprotFoldseekMutationModel, model_dir: str, device: str):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = "cpu"

    config = {
        "foldseek_path": None,
        "config_path": model_dir,
        "load_pretrained": True,
    }
    model = SaprotFoldseekMutationModel(**config)
    model.eval()
    model.to(device)
    return model, device


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def pick_col(df: pd.DataFrame, requested: str, candidates: List[str], allow_none=False) -> Optional[str]:
    if requested != "auto":
        if requested not in df.columns:
            if allow_none:
                return None
            raise ValueError(f"Requested column '{requested}' not found. Columns={list(df.columns)}")
        return requested

    for c in candidates:
        if c in df.columns:
            return c
    if allow_none:
        return None
    raise ValueError(f"Could not infer column from candidates={candidates}. Columns={list(df.columns)}")


def infer_exp_col(df: pd.DataFrame, requested: str) -> Optional[str]:
    if requested != "auto":
        return requested if requested in df.columns else None
    for c in EXPERIMENTAL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def parse_single_mutation_string(s: str) -> List[Tuple[str, int, str]]:
    """
    Parses mutation strings:
    A123V
    A123V:Q124M
    A123V,Q124M
    A123V;Q124M
    """
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    chunks = re.split(r"[:;,]", s)
    muts = []
    for ch in chunks:
        ch = ch.strip()
        m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", ch)
        if not m:
            # Try patterns like A_123_V or A123->V
            m = re.match(r"^([A-Za-z])[_\-\s]*(\d+)[_\-\s>]*([A-Za-z])$", ch)
        if not m:
            raise ValueError(f"Cannot parse mutation chunk: '{ch}' from '{s}'")
        wt, pos, mut = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        muts.append((wt, pos, mut))
    return muts


def build_mutations_from_columns(row: pd.Series, wt_col: str, pos_col: str, mut_col: str) -> List[Tuple[str, int, str]]:
    wt = str(row[wt_col]).strip().upper()
    mut = str(row[mut_col]).strip().upper()
    pos = int(float(row[pos_col]))
    if len(wt) != 1 or len(mut) != 1:
        raise ValueError(f"Bad wt/mut aa columns: wt={wt}, mut={mut}, pos={pos}")
    return [(wt, pos, mut)]


def get_input_files(variant_files: List[str], variant_globs: List[str]) -> List[str]:
    files = []
    for f in variant_files:
        files.append(f)
    for pat in variant_globs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No variant input files provided/matched.")
    return files


def to_float(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        try:
            return float(x.detach().cpu().item())
        except Exception:
            return np.nan


def build_maps(sa_sequences_path: str, manifest_path: str, position_map_path: str):
    sa = pd.read_csv(sa_sequences_path, sep="\t")
    manifest = pd.read_csv(manifest_path)
    posmap = pd.read_csv(position_map_path)

    # seq_id -> compact SaProt sequence
    seq_lookup = dict(zip(sa["seq_id"].astype(str), sa["sa_sequence_compact"].astype(str)))

    # DMS_name/pdb/chain -> seq_id
    manifest_keys = ["DMS_name", "pdb_id", "pdb_chain_id", "seq_id", "sequence_length"]
    manifest_keys = [c for c in manifest_keys if c in manifest.columns]
    manifest_small = manifest[manifest_keys].drop_duplicates()

    # map DMS + fasta_pos -> local pos and seq candidates.
    needed = ["DMS_name", "pdb_id", "pdb_chain_id", "fasta_pos", "wt_local_pos", "wt_aa", "final_source", "af2_plddt"]
    needed = [c for c in needed if c in posmap.columns]
    label = posmap[needed].drop_duplicates().copy()
    label = label.merge(
        manifest_small[["DMS_name", "pdb_id", "pdb_chain_id", "seq_id"]].drop_duplicates(),
        on=["DMS_name", "pdb_id", "pdb_chain_id"],
        how="left",
    )

    return seq_lookup, manifest_small, label


def resolve_one_mutation(
    dms_name: str,
    mut: Tuple[str, int, str],
    label: pd.DataFrame,
    input_position_system: str,
    allow_aa_mismatch: bool,
    row_context: pd.Series,
) -> Tuple[str, str, Dict]:
    """
    Returns:
        seq_id, local_mut_string, diagnostic
    """
    input_wt, input_pos, input_mut = mut

    sub = label[label["DMS_name"].astype(str) == str(dms_name)].copy()

    # Use optional pdb_id / pdb_chain_id if available in variant row.
    if "pdb_id" in row_context.index and pd.notna(row_context["pdb_id"]):
        sub = sub[sub["pdb_id"].astype(str) == str(row_context["pdb_id"])]
    if "pdb_chain_id" in row_context.index and pd.notna(row_context["pdb_chain_id"]):
        sub = sub[sub["pdb_chain_id"].astype(str) == str(row_context["pdb_chain_id"])]

    if input_position_system == "fasta":
        sub = sub[pd.to_numeric(sub["fasta_pos"], errors="coerce") == input_pos]
    else:
        sub = sub[pd.to_numeric(sub["wt_local_pos"], errors="coerce") == input_pos]

    if len(sub) == 0:
        raise ValueError(f"No position mapping for DMS_name={dms_name}, pos={input_pos}, system={input_position_system}")

    # If multiple rows remain but same seq_id/local position, it is okay.
    candidates = sub[["seq_id", "wt_local_pos", "wt_aa"]].drop_duplicates()
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous mapping for DMS_name={dms_name}, pos={input_pos}: "
            f"{candidates.to_dict(orient='records')[:5]}"
        )

    c = candidates.iloc[0]
    seq_id = str(c["seq_id"])
    local_pos = int(c["wt_local_pos"])
    mapped_wt = str(c["wt_aa"]).upper()

    if input_wt != mapped_wt:
        msg = f"WT AA mismatch: input={input_wt}, mapped={mapped_wt}, DMS={dms_name}, pos={input_pos}"
        if not allow_aa_mismatch:
            raise ValueError(msg)

    local_mut_info = f"{mapped_wt}{local_pos}{input_mut}"
    diag = {
        "input_wt_aa": input_wt,
        "mapped_wt_aa": mapped_wt,
        "input_pos": input_pos,
        "wt_local_pos": local_pos,
        "mut_aa": input_mut,
    }
    return seq_id, local_mut_info, diag


def resolve_variant_to_saprot_mut(
    row: pd.Series,
    dms_name: str,
    raw_muts: List[Tuple[str, int, str]],
    label: pd.DataFrame,
    input_position_system: str,
    allow_aa_mismatch: bool,
) -> Tuple[str, str, List[Dict]]:
    seq_id = None
    local_infos = []
    diags = []

    for m in raw_muts:
        this_seq_id, local_info, diag = resolve_one_mutation(
            dms_name=dms_name,
            mut=m,
            label=label,
            input_position_system=input_position_system,
            allow_aa_mismatch=allow_aa_mismatch,
            row_context=row,
        )
        if seq_id is None:
            seq_id = this_seq_id
        elif seq_id != this_seq_id:
            raise ValueError(f"Combinatorial mutation maps to different seq_ids: {seq_id} vs {this_seq_id}")
        local_infos.append(local_info)
        diags.append(diag)

    return seq_id, ":".join(local_infos), diags


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    SaprotFoldseekMutationModel = setup_saprot(args.saprot_repo)
    model, device = load_model(SaprotFoldseekMutationModel, args.model_dir, args.device)

    seq_lookup, manifest_small, label = build_maps(
        args.sa_sequences,
        args.manifest,
        args.position_map,
    )

    files = get_input_files(args.variant_file, args.variant_glob)

    all_out = []
    errors = []
    summaries = []

    # Delay torch import until model is loaded.
    import torch

    for fp in files:
        print(f"[INFO] Scoring file: {fp}")
        df = pd.read_csv(fp)

        if args.dms_col not in df.columns:
            # Infer from filename if DMS_name is absent.
            dms_from_file = Path(fp).stem
            dms_from_file = re.sub(r"_(esm1v|esm2|saprot|scores?|predictions?|zero_shot|input).*?$", "", dms_from_file, flags=re.I)
            df[args.dms_col] = dms_from_file

        exp_col = infer_exp_col(df, args.experimental_col)

        mut_col = pick_col(df, args.mutation_col, MUTATION_CANDIDATES, allow_none=True)
        if mut_col is None:
            pos_col = pick_col(df, args.pos_col, POSITION_CANDIDATES, allow_none=False)
            wt_col = pick_col(df, args.wt_col, WT_CANDIDATES, allow_none=False)
            mutaa_col = pick_col(df, args.mut_col, MUT_AA_CANDIDATES, allow_none=False)
        else:
            pos_col = wt_col = mutaa_col = None

        scores = []
        local_infos = []
        seq_ids = []
        statuses = []

        for i, row in df.iterrows():
            try:
                dms_name = str(row[args.dms_col])

                if mut_col is not None:
                    raw_muts = parse_single_mutation_string(str(row[mut_col]))
                else:
                    raw_muts = build_mutations_from_columns(row, wt_col, pos_col, mutaa_col)

                if not raw_muts:
                    raise ValueError("No mutation parsed")

                seq_id, local_mut_info, diags = resolve_variant_to_saprot_mut(
                    row=row,
                    dms_name=dms_name,
                    raw_muts=raw_muts,
                    label=label,
                    input_position_system=args.input_position_system,
                    allow_aa_mismatch=args.allow_aa_mismatch,
                )

                if seq_id not in seq_lookup:
                    raise ValueError(f"No SaProt sequence for seq_id={seq_id}")

                sa_seq = seq_lookup[seq_id]

                with torch.no_grad():
                    value = model.predict_mut(sa_seq, local_mut_info)

                score = to_float(value)
                scores.append(score)
                local_infos.append(local_mut_info)
                seq_ids.append(seq_id)
                statuses.append("ok")

            except Exception as e:
                scores.append(np.nan)
                local_infos.append("")
                seq_ids.append("")
                statuses.append("error")
                err = {
                    "source_file": fp,
                    "row_index": i,
                    "error": str(e),
                }
                if args.dms_col in df.columns:
                    err["DMS_name"] = row.get(args.dms_col)
                if mut_col is not None:
                    err["mutation"] = row.get(mut_col)
                errors.append(err)

            if (i + 1) % args.batch_log_every == 0:
                print(f"[INFO] {Path(fp).name}: scored {i + 1}/{len(df)} rows")

        out = df.copy()
        out["saprot_score"] = scores
        out["saprot_mut_info_local"] = local_infos
        out["saprot_seq_id"] = seq_ids
        out["saprot_status"] = statuses

        if exp_col is not None and exp_col in out.columns:
            out["experimental_score"] = pd.to_numeric(out[exp_col], errors="coerce")

        stem = Path(fp).stem
        out_path = out_dir / f"{stem}_saprot_scores.csv"
        out.to_csv(out_path, index=False)

        all_out.append(out.assign(source_file=fp))

        summaries.append({
            "source_file": fp,
            "n_rows": len(out),
            "n_ok": int((out["saprot_status"] == "ok").sum()),
            "n_error": int((out["saprot_status"] == "error").sum()),
            "out_path": str(out_path),
            "mutation_col": mut_col,
            "pos_col": pos_col,
            "wt_col": wt_col,
            "mut_col": mutaa_col,
            "experimental_col": exp_col,
        })

        print(f"[OK] Wrote {out_path}")

    if all_out:
        pd.concat(all_out, ignore_index=True).to_csv(out_dir / "saprot_scores_all.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "saprot_score_errors.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "saprot_score_summary.csv", index=False)

    print("[OK] Wrote", out_dir / "saprot_scores_all.csv")
    print("[OK] Wrote", out_dir / "saprot_score_summary.csv")
    if errors:
        print("[WARN] Errors written to", out_dir / "saprot_score_errors.csv")

    print("\nSummary:")
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()

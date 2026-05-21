#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_standardize_model_scores.py

Standardize ESM-1v, ESM-2 and SaProt zero-shot score files into one long table.

Typical command:
python scripts/analysis/16_standardize_model_scores.py \
  --esm2-glob "data_processed/esm_inputs/*_esm2_t33_650M.csv" \
  --esm1v-glob "data_processed/esm_scores/*_esm1v_1to5.csv" \
  --saprot-file data_processed/saprot_scores_v2/saprot_scores_ok.csv \
  --out-dir data_processed/model_scores_standardized

Outputs:
  esm2_standardized.csv
  esm1v_standardized.csv
  saprot_standardized.csv
  all_model_scores_long.csv
  model_score_coverage_summary.csv
  score_direction_by_assay.csv
  score_direction_summary.csv
  standardize_warnings.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


COMMON_COLS = [
    "DMS_name", "pdb_id", "chains_raw", "site_raw", "wildtype", "mutation",
    "mut_name", "DMS_score", "MinMax_normalized", "Rank_quartile",
    "closest_inter", "abagym_chain_label", "pdb_chain_id",
    "seq_index", "seq_pos_1", "esm_mut", "wt_sequence", "mutant_sequence"
]

EXCLUDE_FOR_ESM1V = {
    "Unnamed: 0", "index", "site_raw", "DMS_score", "MinMax_normalized",
    "Rank_quartile", "closest_inter", "seq_index", "seq_pos_1",
    "experimental_score", "saprot_score", "esm2_t33_650M_score"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--esm2-glob", default="data_processed/esm_inputs/*.csv")
    p.add_argument("--esm1v-glob", default="data_processed/esm_scores/*.csv")
    p.add_argument("--saprot-file", default="data_processed/saprot_scores_v2/saprot_scores_ok.csv")
    p.add_argument("--out-dir", default="data_processed/model_scores_standardized")
    p.add_argument("--experimental-col", default="DMS_score")
    p.add_argument("--esm2-score-col", default="esm2_t33_650M_score")
    p.add_argument("--saprot-score-col", default="saprot_score")
    p.add_argument("--esm1v-score-cols", default="auto",
                   help="Comma-separated ESM-1v score columns, or auto.")
    p.add_argument("--min-n", type=int, default=10)
    return p.parse_args()


def infer_dms_from_filename(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"_(esm1v|esm2|saprot|scores?|predictions?|zero_shot|input).*?$", "", stem, flags=re.I)
    return stem


def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def ensure_dms_name(df: pd.DataFrame, path: str) -> pd.DataFrame:
    if "DMS_name" not in df.columns:
        df = df.copy()
        df["DMS_name"] = infer_dms_from_filename(path)
    return df


def standardize_one(df, path, model, score_col, status_col=None, ok_value="ok", experimental_col="DMS_score"):
    df = ensure_dms_name(df, path).copy()

    if score_col not in df.columns:
        raise ValueError(f"{model}: score column '{score_col}' not found in {path}. Columns={list(df.columns)}")
    if experimental_col not in df.columns:
        raise ValueError(f"{model}: experimental column '{experimental_col}' not found in {path}. Columns={list(df.columns)}")

    status = "ok"
    if status_col and status_col in df.columns:
        mask = df[status_col].astype(str).str.lower() == ok_value.lower()
        df = df[mask].copy()
        status = ok_value

    keep = existing_cols(df, COMMON_COLS)
    out = df[keep].copy()
    out["model"] = model
    out["model_score"] = pd.to_numeric(df[score_col], errors="coerce")
    out["status"] = status
    out["source_file"] = path
    out["score_col"] = score_col

    if "DMS_score" not in out.columns and experimental_col in df.columns:
        out["DMS_score"] = pd.to_numeric(df[experimental_col], errors="coerce")

    out = out.dropna(subset=["model_score", "DMS_score"]).copy()
    return out


def infer_esm1v_score_cols(df: pd.DataFrame) -> List[str]:
    preferred = []
    for c in df.columns:
        lc = str(c).lower()
        if c in EXCLUDE_FOR_ESM1V:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if str(c).startswith("/") or "esm1v" in lc or "ur90" in lc or "ur50" in lc:
            preferred.append(c)
    if preferred:
        return preferred

    numeric = []
    for c in df.columns:
        if c in EXCLUDE_FOR_ESM1V or c in COMMON_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    if len(numeric) >= 5:
        return numeric[-5:]
    return numeric


def load_many_csv(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
    return [f for f in files if os.path.isfile(f)]


def load_esm2(args, warnings):
    files = load_many_csv(args.esm2_glob)
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if args.esm2_score_col not in df.columns:
                continue
            out = standardize_one(
                df=df,
                path=f,
                model="ESM2",
                score_col=args.esm2_score_col,
                status_col="esm2_status",
                experimental_col=args.experimental_col,
            )
            parts.append(out)
        except Exception as e:
            warnings.append({"model": "ESM2", "file": f, "warning": str(e)})
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_saprot(args, warnings):
    f = args.saprot_file
    if not os.path.exists(f):
        warnings.append({"model": "SaProt", "file": f, "warning": "file not found"})
        return pd.DataFrame()
    try:
        df = pd.read_csv(f)
        status_col = "saprot_status" if "saprot_status" in df.columns else None
        return standardize_one(
            df=df,
            path=f,
            model="SaProt",
            score_col=args.saprot_score_col,
            status_col=status_col,
            experimental_col=args.experimental_col,
        )
    except Exception as e:
        warnings.append({"model": "SaProt", "file": f, "warning": str(e)})
        return pd.DataFrame()


def load_esm1v(args, warnings):
    files = load_many_csv(args.esm1v_glob)
    parts = []

    specified_cols = None
    if args.esm1v_score_cols != "auto":
        specified_cols = [x.strip() for x in args.esm1v_score_cols.split(",") if x.strip()]

    for f in files:
        try:
            df = pd.read_csv(f)
            df = ensure_dms_name(df, f)

            if args.experimental_col not in df.columns:
                warnings.append({"model": "ESM1v", "file": f, "warning": f"missing {args.experimental_col}"})
                continue

            score_cols = specified_cols if specified_cols is not None else infer_esm1v_score_cols(df)
            score_cols = [c for c in score_cols if c in df.columns]
            if not score_cols:
                warnings.append({"model": "ESM1v", "file": f, "warning": "no ESM-1v score columns inferred"})
                continue

            tmp = df.copy()
            for c in score_cols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp["esm1v_mean_score"] = tmp[score_cols].mean(axis=1)

            keep = existing_cols(tmp, COMMON_COLS)
            out = tmp[keep].copy()
            out["model"] = "ESM1v"
            out["model_score"] = tmp["esm1v_mean_score"]
            out["status"] = "ok"
            out["source_file"] = f
            out["score_col"] = "mean(" + "|".join(score_cols) + ")"
            out["esm1v_n_models"] = tmp[score_cols].notna().sum(axis=1)
            out = out.dropna(subset=["model_score", "DMS_score"]).copy()
            parts.append(out)
        except Exception as e:
            warnings.append({"model": "ESM1v", "file": f, "warning": str(e)})
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def coverage_summary(all_long):
    rows = []
    for model, g in all_long.groupby("model"):
        rows.append({
            "model": model,
            "n_rows": len(g),
            "n_assays": g["DMS_name"].nunique(),
            "n_source_files": g["source_file"].nunique(),
            "n_nonnull_scores": g["model_score"].notna().sum(),
            "mean_model_score": g["model_score"].mean(),
            "std_model_score": g["model_score"].std(),
        })
    return pd.DataFrame(rows)


def compute_direction(all_long, min_n=10):
    rows = []
    for (model, dms), g in all_long.groupby(["model", "DMS_name"], dropna=False):
        gg = g[["DMS_score", "model_score"]].dropna()
        n = len(gg)
        rho = np.nan
        pearson = np.nan
        if n >= min_n and gg["DMS_score"].nunique() > 1 and gg["model_score"].nunique() > 1:
            rho = gg["DMS_score"].corr(gg["model_score"], method="spearman")
            pearson = gg["DMS_score"].corr(gg["model_score"], method="pearson")
        rows.append({"model": model, "DMS_name": dms, "n": n, "spearman": rho, "pearson": pearson})
    return pd.DataFrame(rows)


def direction_summary(direction):
    if direction.empty:
        return pd.DataFrame()
    return (
        direction.groupby("model")
        .agg(
            n_assays=("DMS_name", "nunique"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            n_positive_spearman=("spearman", lambda x: int((x > 0).sum())),
            n_negative_spearman=("spearman", lambda x: int((x < 0).sum())),
            mean_pearson=("pearson", "mean"),
            median_pearson=("pearson", "median"),
        )
        .reset_index()
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    print("[INFO] Loading ESM-2...")
    esm2 = load_esm2(args, warnings)
    print("[INFO] ESM-2 rows:", len(esm2))

    print("[INFO] Loading ESM-1v...")
    esm1v = load_esm1v(args, warnings)
    print("[INFO] ESM-1v rows:", len(esm1v))

    print("[INFO] Loading SaProt...")
    saprot = load_saprot(args, warnings)
    print("[INFO] SaProt rows:", len(saprot))

    if not esm2.empty:
        esm2.to_csv(out_dir / "esm2_standardized.csv", index=False)
    if not esm1v.empty:
        esm1v.to_csv(out_dir / "esm1v_standardized.csv", index=False)
    if not saprot.empty:
        saprot.to_csv(out_dir / "saprot_standardized.csv", index=False)

    parts = [x for x in [esm1v, esm2, saprot] if not x.empty]
    all_long = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    all_long.to_csv(out_dir / "all_model_scores_long.csv", index=False)

    cov = coverage_summary(all_long) if not all_long.empty else pd.DataFrame()
    cov.to_csv(out_dir / "model_score_coverage_summary.csv", index=False)

    direction = compute_direction(all_long, min_n=args.min_n) if not all_long.empty else pd.DataFrame()
    direction.to_csv(out_dir / "score_direction_by_assay.csv", index=False)

    ds = direction_summary(direction)
    ds.to_csv(out_dir / "score_direction_summary.csv", index=False)

    pd.DataFrame(warnings).to_csv(out_dir / "standardize_warnings.csv", index=False)

    print("\n[OK] Wrote standardized outputs to:", out_dir)
    print("\nCoverage:")
    print(cov.to_string(index=False) if not cov.empty else "No rows")

    print("\nDirection summary:")
    print(ds.to_string(index=False) if not ds.empty else "No rows")

    if warnings:
        print("\nWarnings:")
        print(pd.DataFrame(warnings).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

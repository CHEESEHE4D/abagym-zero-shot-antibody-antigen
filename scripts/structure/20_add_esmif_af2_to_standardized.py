#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    20_add_esmif_af2_to_standardized.py

这个脚本做什么：
    将第 19 步得到的 ESM-IF-AF2 merged 结果，整理成与现有三个模型
    esm1v_standardized.csv / esm2_standardized.csv / saprot_standardized.csv
    完全一致的 standardized 表结构，并追加到 all_model_scores_long.csv。

本版新增：
    同时更新：
        data_processed/model_scores_standardized/score_direction_by_assay.csv
        data_processed/model_scores_standardized/score_direction_summary.csv

为什么要更新 score_direction_by_assay.csv：
    你现有的 score_direction_by_assay.csv 记录了每个模型在每个 assay 上与 DMS_score 的
    Spearman / Pearson 相关性，列为：
        model, DMS_name, n, spearman, pearson

    加入 ESM-IF-AF2 后，应当把 ESM-IF-AF2 对应的 assay-level 相关性也追加进去，
    这样后续方向判断、模型比较和评价图都能直接读取统一文件。

读取：
    1. data_processed/esm_if_af2_scores/esmif_af2_scores_merged.csv
    2. data_processed/model_scores_standardized/all_model_scores_long.csv
    3. data_processed/model_scores_standardized/score_direction_by_assay.csv
    4. data_processed/model_scores_standardized/score_direction_summary.csv

输出 / 更新：
    1. data_processed/model_scores_standardized/esmif_af2_standardized.csv
    2. data_processed/model_scores_standardized/all_model_scores_long.csv
    3. data_processed/model_scores_standardized/score_direction_by_assay.csv
    4. data_processed/model_scores_standardized/score_direction_summary.csv
    5. data_processed/model_scores_standardized/esmif_af2_append_report.csv

使用示例：
    python scripts/structure/20_add_esmif_af2_to_standardized.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None


STANDARDIZED_COLS = [
    "DMS_name",
    "pdb_id",
    "chains_raw",
    "site_raw",
    "wildtype",
    "mutation",
    "DMS_score",
    "abagym_chain_label",
    "pdb_chain_id",
    "esm_mut",
    "wt_sequence",
    "mutant_sequence",
    "model",
    "model_score",
    "status",
    "source_file",
    "score_col",
]

LONG_COLS = [
    "DMS_name",
    "pdb_id",
    "chains_raw",
    "site_raw",
    "wildtype",
    "mutation",
    "DMS_score",
    "abagym_chain_label",
    "pdb_chain_id",
    "model",
    "model_score",
    "status",
    "source_file",
    "score_col",
]

SCORE_DIRECTION_BY_ASSAY_COLS = [
    "model",
    "DMS_name",
    "n",
    "spearman",
    "pearson",
]


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = Path(str(path) + f".bak_{ts}")
    shutil.copy2(path, bak)
    return bak


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols].copy()


def safe_corr(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(sub)
    if n < 3:
        return None, None, n
    if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return None, None, n

    if spearmanr is not None:
        sp = spearmanr(sub["x"], sub["y"]).correlation
    else:
        sp = sub["x"].rank().corr(sub["y"].rank())

    if pearsonr is not None:
        try:
            pr = pearsonr(sub["x"], sub["y"])[0]
        except Exception:
            pr = None
    else:
        pr = sub["x"].corr(sub["y"])

    return sp, pr, n


def build_esmif_standardized(merged: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if score_col not in merged.columns:
        raise ValueError(f"Score column not found in ESM-IF merged table: {score_col}")

    work = merged.copy()

    if "esmif_af2_merge_status" in work.columns:
        work = work[work["esmif_af2_merge_status"].astype(str) == "ok"].copy()

    work["model"] = "ESM-IF-AF2"
    work["model_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["status"] = work["model_score"].notna().map(lambda x: "ok" if x else "missing_score")
    work["source_file"] = work["esmif_af2_score_csv"] if "esmif_af2_score_csv" in work.columns else ""
    work["score_col"] = score_col

    std = ensure_cols(work, STANDARDIZED_COLS)

    for c in ["site_raw", "DMS_score", "model_score"]:
        if c in std.columns:
            std[c] = pd.to_numeric(std[c], errors="coerce")

    return std


def remove_model_rows(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    return df[df["model"].astype(str) != model_name].copy()


def compute_score_direction_by_assay(std: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = std[std["status"].astype(str) == "ok"].copy()
    ok["DMS_score"] = pd.to_numeric(ok["DMS_score"], errors="coerce")
    ok["model_score"] = pd.to_numeric(ok["model_score"], errors="coerce")

    for dms_name, g in ok.groupby("DMS_name", dropna=False):
        sp, pr, n = safe_corr(g["DMS_score"], g["model_score"])
        rows.append({
            "model": "ESM-IF-AF2",
            "DMS_name": dms_name,
            "n": n,
            "spearman": sp,
            "pearson": pr,
        })

    return pd.DataFrame(rows, columns=SCORE_DIRECTION_BY_ASSAY_COLS)


def build_score_direction_summary(score_by_assay: pd.DataFrame) -> pd.DataFrame:
    if score_by_assay.empty:
        return pd.DataFrame(columns=[
            "model", "n_assays", "n_assays_with_spearman",
            "spearman_mean", "spearman_median",
            "pearson_mean", "pearson_median",
            "n_total"
        ])

    return (
        score_by_assay
        .groupby("model", dropna=False)
        .agg(
            n_assays=("DMS_name", "nunique"),
            n_assays_with_spearman=("spearman", lambda x: x.notna().sum()),
            spearman_mean=("spearman", "mean"),
            spearman_median=("spearman", "median"),
            pearson_mean=("pearson", "mean"),
            pearson_median=("pearson", "median"),
            n_total=("n", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add ESM-IF-AF2 to existing standardized model score tables.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录。")
    parser.add_argument(
        "--esmif-merged-csv",
        type=str,
        default="data_processed/esm_if_af2_scores/esmif_af2_scores_merged.csv",
        help="第 19 步输出的 ESM-IF-AF2 merged 表。",
    )
    parser.add_argument(
        "--standardized-dir",
        type=str,
        default="data_processed/model_scores_standardized",
        help="标准化模型分数目录。",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default="esmif_af2_neg_delta_ll",
        help="作为 model_score 的 ESM-IF-AF2 分数列。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份将被更新的 CSV。默认会自动备份。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    esmif_merged_csv = (project_root / args.esmif_merged_csv).resolve()
    standardized_dir = (project_root / args.standardized_dir).resolve()
    standardized_dir.mkdir(parents=True, exist_ok=True)

    esmif_std_csv = standardized_dir / "esmif_af2_standardized.csv"
    all_long_csv = standardized_dir / "all_model_scores_long.csv"
    score_by_assay_csv = standardized_dir / "score_direction_by_assay.csv"
    score_summary_csv = standardized_dir / "score_direction_summary.csv"
    report_csv = standardized_dir / "esmif_af2_append_report.csv"

    if not esmif_merged_csv.exists():
        raise FileNotFoundError(f"ESM-IF-AF2 merged csv not found: {esmif_merged_csv}")

    merged = pd.read_csv(esmif_merged_csv)
    esmif_std = build_esmif_standardized(merged, args.score_col)
    esmif_std.to_csv(esmif_std_csv, index=False)

    # 更新 all_model_scores_long.csv
    esmif_long = ensure_cols(esmif_std, LONG_COLS)

    backups = []

    if all_long_csv.exists():
        old_long = pd.read_csv(all_long_csv)
        old_long = ensure_cols(old_long, LONG_COLS)
        old_long_without_esmif = remove_model_rows(old_long, "ESM-IF-AF2")

        if not args.no_backup:
            bak = backup_file(all_long_csv)
            if bak:
                backups.append(str(bak))

        updated_long = pd.concat([old_long_without_esmif, esmif_long], ignore_index=True)
    else:
        old_long = pd.DataFrame(columns=LONG_COLS)
        old_long_without_esmif = old_long.copy()
        updated_long = esmif_long.copy()

    updated_long.to_csv(all_long_csv, index=False)

    # 更新 score_direction_by_assay.csv
    esmif_by_assay = compute_score_direction_by_assay(esmif_std)

    if score_by_assay_csv.exists():
        old_by_assay = pd.read_csv(score_by_assay_csv)
        old_by_assay = ensure_cols(old_by_assay, SCORE_DIRECTION_BY_ASSAY_COLS)
        old_by_assay_without_esmif = remove_model_rows(old_by_assay, "ESM-IF-AF2")

        if not args.no_backup:
            bak = backup_file(score_by_assay_csv)
            if bak:
                backups.append(str(bak))

        updated_by_assay = pd.concat([old_by_assay_without_esmif, esmif_by_assay], ignore_index=True)
    else:
        old_by_assay = pd.DataFrame(columns=SCORE_DIRECTION_BY_ASSAY_COLS)
        old_by_assay_without_esmif = old_by_assay.copy()
        updated_by_assay = esmif_by_assay.copy()

    updated_by_assay.to_csv(score_by_assay_csv, index=False)

    # 更新 score_direction_summary.csv
    updated_summary = build_score_direction_summary(updated_by_assay)

    if score_summary_csv.exists() and not args.no_backup:
        bak = backup_file(score_summary_csv)
        if bak:
            backups.append(str(bak))

    updated_summary.to_csv(score_summary_csv, index=False)

    report_rows = [
        {
            "item": "esmif_merged_input",
            "path": str(esmif_merged_csv),
            "rows": len(merged),
            "note": "",
        },
        {
            "item": "esmif_standardized_output",
            "path": str(esmif_std_csv),
            "rows": len(esmif_std),
            "note": f"score_col={args.score_col}",
        },
        {
            "item": "all_model_scores_long_updated",
            "path": str(all_long_csv),
            "rows": len(updated_long),
            "note": "",
        },
        {
            "item": "score_direction_by_assay_updated",
            "path": str(score_by_assay_csv),
            "rows": len(updated_by_assay),
            "note": "",
        },
        {
            "item": "score_direction_summary_updated",
            "path": str(score_summary_csv),
            "rows": len(updated_summary),
            "note": "",
        },
        {
            "item": "backups",
            "path": ";".join(backups),
            "rows": len(backups),
            "note": "",
        },
    ]

    report = pd.DataFrame(report_rows)
    report.to_csv(report_csv, index=False)

    print(f"[OK] ESM-IF-AF2 standardized table written to: {esmif_std_csv}")
    print(f"[OK] Updated all_model_scores_long.csv written to: {all_long_csv}")
    print(f"[OK] Updated score_direction_by_assay.csv written to: {score_by_assay_csv}")
    print(f"[OK] Updated score_direction_summary.csv written to: {score_summary_csv}")
    print(f"[OK] Append report written to: {report_csv}")

    print("[INFO] Rows:")
    print(f"  ESM-IF merged input: {len(merged)}")
    print(f"  ESM-IF standardized: {len(esmif_std)}")
    print(f"  all_model_scores_long updated: {len(updated_long)}")
    print(f"  score_direction_by_assay updated: {len(updated_by_assay)}")
    print(f"  score_direction_summary updated: {len(updated_summary)}")

    print("[INFO] Models in updated all_model_scores_long:")
    if "model" in updated_long.columns:
        print(updated_long["model"].value_counts(dropna=False).to_string())

    print("[INFO] Models in updated score_direction_by_assay:")
    if "model" in updated_by_assay.columns:
        print(updated_by_assay["model"].value_counts(dropna=False).to_string())

    print("[INFO] ESM-IF-AF2 assay-level direction preview:")
    print(esmif_by_assay.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

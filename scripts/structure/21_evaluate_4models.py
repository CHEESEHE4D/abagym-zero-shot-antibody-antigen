#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    21_evaluate_4models.py

这个脚本做什么：
    基于已经更新后的：
        data_processed/model_scores_standardized/all_model_scores_long.csv

    重新计算加入 ESM-IF-AF2 后的四模型评价结果，并输出到：
        data_processed/evaluation_4models/

本版新增：
    在原有 Spearman / Pearson 评价基础上，加入 Top-K 指标：
        topk_by_assay_available.csv
        topk_by_model_available.csv
        topk_by_virus_available.csv
        topk_by_assay_common.csv
        topk_by_model_common.csv
        topk_by_virus_common.csv

评价方式：
    1. available set：
       每个模型在自己有预测分数的所有突变上分别评估。

    2. common set：
       只保留所有模型都有预测分数的相同突变集合，再比较模型表现。

Top-K 定义：
    在每个 assay 内，按实验 DMS_score 取 top K 突变作为 true_topK；
    按模型 model_score 取 top K 突变作为 pred_topK；
    计算二者重叠程度：
        topk_overlap
        topk_recall = overlap / K
        topk_precision = overlap / K
        topk_jaccard = overlap / union

默认 K：
    使用比例型 K：top 5%、top 10%、top 20%
    即 K = max(1, round(n * top_fraction))

重要前提：
    默认假设 DMS_score 越大表示实验效应越强；
    model_score 越大表示模型预测效应越强。
    如果某些 assay 的 DMS_score 方向尚未统一，则 Top-K 的解释需要谨慎。

读取：
    data_processed/model_scores_standardized/all_model_scores_long.csv

输出：
    data_processed/evaluation_4models/
        all_model_scores_long.csv
        model_score_coverage_summary.csv

        metrics_by_assay_available.csv
        metrics_by_model_available.csv
        metrics_by_virus_available.csv

        metrics_by_assay_common.csv
        metrics_by_model_common.csv
        metrics_by_virus_common.csv

        topk_by_assay_available.csv
        topk_by_model_available.csv
        topk_by_virus_available.csv

        topk_by_assay_common.csv
        topk_by_model_common.csv
        topk_by_virus_common.csv

        common_subset_mutation_keys.csv
        evaluation_summary.txt

使用示例：
    python scripts/structure/21_evaluate_4models.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym

如果只想计算 top10%：
    python scripts/structure/21_evaluate_4models.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym \
        --top-fracs 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def derive_virus(dms_name: str) -> str:
    dms_name = clean_text(dms_name)
    if dms_name.startswith("COVID-19"):
        return "COVID-19"
    if dms_name.startswith("HIV"):
        return "HIV"
    if dms_name.startswith("Influenza"):
        return "Influenza"
    return dms_name.split("_")[0] if dms_name else ""


def make_mutation_key(df: pd.DataFrame) -> pd.Series:
    """
    当前 all_model_scores_long 通常没有 esm_mut / mutant_sequence，
    因此使用 DMS_name + pdb_chain_id + site_raw + wildtype + mutation 构造突变唯一键。
    """
    key_cols = ["DMS_name", "pdb_chain_id", "site_raw", "wildtype", "mutation"]
    parts = []
    for c in key_cols:
        if c in df.columns:
            parts.append(df[c].astype(str).fillna("").str.strip())
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    return key


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
        sp = sub["x"].rank().corr(sub["y"].rank(), method="pearson")

    if pearsonr is not None:
        try:
            pr = pearsonr(sub["x"], sub["y"])[0]
        except Exception:
            pr = None
    else:
        pr = sub["x"].corr(sub["y"], method="pearson")

    return sp, pr, n


def evaluate_by_assay(df: pd.DataFrame, target_col: str, score_col: str, evaluation_set: str) -> pd.DataFrame:
    rows = []
    for (model, dms_name), g in df.groupby(["model", "DMS_name"], dropna=False):
        sp, pr, n = safe_corr(g[target_col], g[score_col])
        rows.append({
            "evaluation_set": evaluation_set,
            "model": model,
            "DMS_name": dms_name,
            "virus": derive_virus(dms_name),
            "n": n,
            "spearman": sp,
            "pearson": pr,
            "n_total_rows": len(g),
            "n_score_nonnull": int(g[score_col].notna().sum()),
            "n_target_nonnull": int(g[target_col].notna().sum()),
        })
    return pd.DataFrame(rows)


def summarize_by_model(metrics_by_assay: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_assay.empty:
        return pd.DataFrame()
    return (
        metrics_by_assay
        .groupby(["evaluation_set", "model"], dropna=False)
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


def summarize_by_virus(metrics_by_assay: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_assay.empty:
        return pd.DataFrame()
    return (
        metrics_by_assay
        .groupby(["evaluation_set", "virus", "model"], dropna=False)
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


def coverage_summary(df: pd.DataFrame, target_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model", dropna=False):
        rows.append({
            "model": model,
            "n_rows": len(g),
            "n_assays": g["DMS_name"].nunique(),
            "n_mutation_keys": g["mutation_key"].nunique(),
            "n_score_nonnull": int(g[score_col].notna().sum()),
            "n_target_nonnull": int(g[target_col].notna().sum()),
            "n_usable_rows": int(g[[target_col, score_col]].dropna().shape[0]),
        })
    return pd.DataFrame(rows)


def compute_topk_by_assay(
    df: pd.DataFrame,
    target_col: str,
    score_col: str,
    evaluation_set: str,
    top_fracs: List[float],
    target_descending: bool = True,
    score_descending: bool = True,
) -> pd.DataFrame:
    rows = []

    for (model, dms_name), g in df.groupby(["model", "DMS_name"], dropna=False):
        sub = g[["mutation_key", target_col, score_col]].dropna().drop_duplicates("mutation_key").copy()
        n = len(sub)

        for frac in top_fracs:
            if n < 1:
                rows.append({
                    "evaluation_set": evaluation_set,
                    "model": model,
                    "DMS_name": dms_name,
                    "virus": derive_virus(dms_name),
                    "top_frac": frac,
                    "k": 0,
                    "n": n,
                    "topk_overlap": 0,
                    "topk_recall": None,
                    "topk_precision": None,
                    "topk_jaccard": None,
                })
                continue

            k = max(1, int(round(n * frac)))
            k = min(k, n)

            true_top = set(
                sub.sort_values(target_col, ascending=not target_descending)
                .head(k)["mutation_key"]
                .tolist()
            )
            pred_top = set(
                sub.sort_values(score_col, ascending=not score_descending)
                .head(k)["mutation_key"]
                .tolist()
            )

            overlap = len(true_top & pred_top)
            union = len(true_top | pred_top)

            rows.append({
                "evaluation_set": evaluation_set,
                "model": model,
                "DMS_name": dms_name,
                "virus": derive_virus(dms_name),
                "top_frac": frac,
                "k": k,
                "n": n,
                "topk_overlap": overlap,
                "topk_recall": overlap / k if k else None,
                "topk_precision": overlap / k if k else None,
                "topk_jaccard": overlap / union if union else None,
            })

    return pd.DataFrame(rows)


def summarize_topk_by_model(topk_by_assay: pd.DataFrame) -> pd.DataFrame:
    if topk_by_assay.empty:
        return pd.DataFrame()

    return (
        topk_by_assay
        .groupby(["evaluation_set", "model", "top_frac"], dropna=False)
        .agg(
            n_assays=("DMS_name", "nunique"),
            topk_recall_mean=("topk_recall", "mean"),
            topk_recall_median=("topk_recall", "median"),
            topk_precision_mean=("topk_precision", "mean"),
            topk_jaccard_mean=("topk_jaccard", "mean"),
            n_total=("n", "sum"),
            k_mean=("k", "mean"),
        )
        .reset_index()
    )


def summarize_topk_by_virus(topk_by_assay: pd.DataFrame) -> pd.DataFrame:
    if topk_by_assay.empty:
        return pd.DataFrame()

    return (
        topk_by_assay
        .groupby(["evaluation_set", "virus", "model", "top_frac"], dropna=False)
        .agg(
            n_assays=("DMS_name", "nunique"),
            topk_recall_mean=("topk_recall", "mean"),
            topk_recall_median=("topk_recall", "median"),
            topk_precision_mean=("topk_precision", "mean"),
            topk_jaccard_mean=("topk_jaccard", "mean"),
            n_total=("n", "sum"),
            k_mean=("k", "mean"),
        )
        .reset_index()
    )


def parse_top_fracs(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("--top-fracs cannot be empty")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4 models with correlation and Top-K metrics.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录。")
    parser.add_argument(
        "--long-csv",
        type=str,
        default="data_processed/model_scores_standardized/all_model_scores_long.csv",
        help="标准化长表。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_processed/evaluation_4models",
        help="输出目录。",
    )
    parser.add_argument("--target-col", type=str, default="DMS_score", help="实验真值列。")
    parser.add_argument("--score-col", type=str, default="model_score", help="模型分数列。")
    parser.add_argument(
        "--top-fracs",
        type=str,
        default="0.05,0.10,0.20",
        help="Top-K 比例，逗号分隔。默认 0.05,0.10,0.20。",
    )
    parser.add_argument(
        "--target-ascending",
        action="store_true",
        help="若实验分数越小表示越强效，使用该参数。默认实验分数越大越强效。",
    )
    parser.add_argument(
        "--score-ascending",
        action="store_true",
        help="若模型分数越小表示预测越强效，使用该参数。默认模型分数越大越强效。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    long_csv = (project_root / args.long_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_fracs = parse_top_fracs(args.top_fracs)

    if not long_csv.exists():
        raise FileNotFoundError(f"Long table not found: {long_csv}")

    df = pd.read_csv(long_csv)

    required = {"DMS_name", "model", args.target_col, args.score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"all_model_scores_long.csv missing required columns: {sorted(missing)}")

    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    df["virus"] = df["DMS_name"].map(derive_virus)
    df["mutation_key"] = make_mutation_key(df)

    if "status" in df.columns:
        eval_df = df[df["status"].astype(str) == "ok"].copy()
    else:
        eval_df = df.copy()

    available_df = eval_df.dropna(subset=[args.target_col, args.score_col]).copy()

    cov = coverage_summary(eval_df, args.target_col, args.score_col)

    # Correlation: available
    metrics_by_assay_available = evaluate_by_assay(
        available_df,
        target_col=args.target_col,
        score_col=args.score_col,
        evaluation_set="available",
    )
    metrics_by_model_available = summarize_by_model(metrics_by_assay_available)
    metrics_by_virus_available = summarize_by_virus(metrics_by_assay_available)

    # Common set
    models = sorted(available_df["model"].dropna().astype(str).unique().tolist())
    n_models = len(models)

    key_model_count = available_df.groupby("mutation_key")["model"].nunique()
    common_keys = set(key_model_count[key_model_count == n_models].index)
    common_df = available_df[available_df["mutation_key"].isin(common_keys)].copy()

    # Correlation: common
    metrics_by_assay_common = evaluate_by_assay(
        common_df,
        target_col=args.target_col,
        score_col=args.score_col,
        evaluation_set="common",
    )
    metrics_by_model_common = summarize_by_model(metrics_by_assay_common)
    metrics_by_virus_common = summarize_by_virus(metrics_by_assay_common)

    # Top-K: available
    topk_by_assay_available = compute_topk_by_assay(
        available_df,
        target_col=args.target_col,
        score_col=args.score_col,
        evaluation_set="available",
        top_fracs=top_fracs,
        target_descending=not args.target_ascending,
        score_descending=not args.score_ascending,
    )
    topk_by_model_available = summarize_topk_by_model(topk_by_assay_available)
    topk_by_virus_available = summarize_topk_by_virus(topk_by_assay_available)

    # Top-K: common
    topk_by_assay_common = compute_topk_by_assay(
        common_df,
        target_col=args.target_col,
        score_col=args.score_col,
        evaluation_set="common",
        top_fracs=top_fracs,
        target_descending=not args.target_ascending,
        score_descending=not args.score_ascending,
    )
    topk_by_model_common = summarize_topk_by_model(topk_by_assay_common)
    topk_by_virus_common = summarize_topk_by_virus(topk_by_assay_common)

    # Common mutation keys
    common_cols = ["mutation_key", "DMS_name", "pdb_chain_id", "site_raw", "wildtype", "mutation"]
    for c in common_cols:
        if c not in common_df.columns:
            common_df[c] = pd.NA

    common_key_table = (
        common_df[common_cols]
        .drop_duplicates()
        .sort_values(["DMS_name", "pdb_chain_id", "site_raw", "wildtype", "mutation"], kind="stable")
    )

    # Outputs
    df.to_csv(out_dir / "all_model_scores_long.csv", index=False)
    cov.to_csv(out_dir / "model_score_coverage_summary.csv", index=False)

    metrics_by_assay_available.to_csv(out_dir / "metrics_by_assay_available.csv", index=False)
    metrics_by_model_available.to_csv(out_dir / "metrics_by_model_available.csv", index=False)
    metrics_by_virus_available.to_csv(out_dir / "metrics_by_virus_available.csv", index=False)

    metrics_by_assay_common.to_csv(out_dir / "metrics_by_assay_common.csv", index=False)
    metrics_by_model_common.to_csv(out_dir / "metrics_by_model_common.csv", index=False)
    metrics_by_virus_common.to_csv(out_dir / "metrics_by_virus_common.csv", index=False)

    topk_by_assay_available.to_csv(out_dir / "topk_by_assay_available.csv", index=False)
    topk_by_model_available.to_csv(out_dir / "topk_by_model_available.csv", index=False)
    topk_by_virus_available.to_csv(out_dir / "topk_by_virus_available.csv", index=False)

    topk_by_assay_common.to_csv(out_dir / "topk_by_assay_common.csv", index=False)
    topk_by_model_common.to_csv(out_dir / "topk_by_model_common.csv", index=False)
    topk_by_virus_common.to_csv(out_dir / "topk_by_virus_common.csv", index=False)

    common_key_table.to_csv(out_dir / "common_subset_mutation_keys.csv", index=False)

    with (out_dir / "evaluation_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Evaluation summary after adding ESM-IF-AF2, including Top-K metrics\n")
        f.write("=" * 80 + "\n")
        f.write(f"input_long_csv: {long_csv}\n")
        f.write(f"target_col: {args.target_col}\n")
        f.write(f"score_col: {args.score_col}\n")
        f.write(f"top_fracs: {top_fracs}\n")
        f.write(f"target_direction: {'ascending' if args.target_ascending else 'descending'}\n")
        f.write(f"score_direction: {'ascending' if args.score_ascending else 'descending'}\n")
        f.write(f"n_rows_all: {len(df)}\n")
        f.write(f"n_rows_status_ok_or_all: {len(eval_df)}\n")
        f.write(f"n_rows_available_nonnull: {len(available_df)}\n")
        f.write(f"n_models: {n_models}\n")
        f.write(f"models: {', '.join(models)}\n")
        f.write(f"n_mutation_keys_available: {available_df['mutation_key'].nunique()}\n")
        f.write(f"n_common_mutation_keys_all_models: {len(common_keys)}\n\n")

        f.write("[Available set: correlation metrics by model]\n")
        f.write(metrics_by_model_available.to_string(index=False))
        f.write("\n\n[Common set: correlation metrics by model]\n")
        f.write(metrics_by_model_common.to_string(index=False) if not metrics_by_model_common.empty else "No common subset.\n")

        f.write("\n\n[Available set: Top-K metrics by model]\n")
        f.write(topk_by_model_available.to_string(index=False))
        f.write("\n\n[Common set: Top-K metrics by model]\n")
        f.write(topk_by_model_common.to_string(index=False) if not topk_by_model_common.empty else "No common subset.\n")

    print(f"[OK] Evaluation outputs written to: {out_dir}")
    print(f"[INFO] Models: {models}")
    print(f"[INFO] n_rows_available_nonnull: {len(available_df)}")
    print(f"[INFO] n_common_mutation_keys_all_models: {len(common_keys)}")
    print(f"[INFO] Top fractions: {top_fracs}")

    print("\n[INFO] Available correlation metrics by model:")
    print(metrics_by_model_available.to_string(index=False))

    print("\n[INFO] Common correlation metrics by model:")
    print(metrics_by_model_common.to_string(index=False) if not metrics_by_model_common.empty else "No common subset.")

    print("\n[INFO] Available Top-K metrics by model:")
    print(topk_by_model_available.to_string(index=False))

    print("\n[INFO] Common Top-K metrics by model:")
    print(topk_by_model_common.to_string(index=False) if not topk_by_model_common.empty else "No common subset.")


if __name__ == "__main__":
    main()

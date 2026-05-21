#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    19_merge_esmif_af2_scores.py

这个脚本做什么：
    将 ESM-IF-AF2 的打分结果合并回原始 esm_inputs/*.csv 变体表，
    并计算每个 mutant 相对于 WT 的 log-likelihood 差值。

本版为什么要改：
    之前第 18 步曾经因为输出文件命名不唯一导致同一 DMS_name 下不同 input_filename
    互相覆盖。后来新版第 18 步改成了唯一命名：

        {input_filename_stem}__{DMS_name}__chain_{pdb_chain_id}__esmif_{chain_used_for_run}__{mode}.csv

    但已有的一部分旧结果仍可能保留旧命名：

        {DMS_name}__{chain_used_for_run}_esmif_{mode}.csv

    因此本脚本不再依赖 esm_if_run_manifest_single.csv，而是：
        1. 读取第 17 步的 fasta manifest 和第 16 步的 AF2 input manifest
        2. 对每个 assay/chain 生成“预期唯一命名结果文件”
        3. 若唯一命名文件存在，优先使用
        4. 若不存在，再回退寻找旧命名文件
        5. 将找到的 score CSV 合并回对应 esm_inputs/*.csv

这个脚本读取什么数据：
    1. data_processed/esm_if_af2_inputs/esm_if_fasta_manifest.csv
       - 来自第 17 步
       - 提供 input_filename、DMS_name、pdb_chain_id、fasta_path 等

    2. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv
       - 来自第 16 步
       - 提供 can_run_esm_if、esm_if_chain_id、af2_seq_id、af2_structure_path 等

    3. data_processed/esm_if_af2_scores/single/*.csv
       - 来自第 18 步
       - ESM-IF 官方输出，通常包含：
           * seqid
           * log_likelihood

    4. data_processed/esm_inputs/*.csv
       - 原始变体输入表
       - 用于合并 DMS_score、mut_names、esm_mut、wt_sequence、mutant_sequence 等信息

这个脚本输出什么：
    1. data_processed/esm_if_af2_scores/esmif_af2_scores_merged.csv
       - 每个突变一行
       - 在原始 esm_inputs 字段基础上新增：
           * esmif_af2_wt_ll
           * esmif_af2_mut_ll
           * esmif_af2_delta_ll
           * esmif_af2_neg_delta_ll
           * esmif_af2_merge_status
           * esmif_af2_score_csv
           * esmif_af2_score_filename_type

    2. data_processed/esm_if_af2_scores/esmif_af2_scores_merge_summary.csv
       - 每个 assay/chain 一行的合并统计
       - 记录实际使用的是 unique_name / old_name / missing

分数定义：
    esmif_af2_delta_ll = log_likelihood(mutant | AF2 backbone)
                         - log_likelihood(WT | AF2 backbone)

    esmif_af2_neg_delta_ll = -esmif_af2_delta_ll

论文写作时可以怎么描述：
    ESM-IF 输出每条 WT 或 mutant 序列在给定 AF2 backbone 条件下的 log-likelihood。
    本研究以 mutant 与 WT 的 log-likelihood 差值作为 ESM-IF 零样本突变分数。
    为保证不同 assay/chain 的结果不因文件名重复而覆盖，合并阶段以 input_filename、
    DMS_name 与链标识共同确定唯一结果文件。

使用示例：
    python scripts/structure/19_merge_esmif_af2_scores.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def safe_name(text: str) -> str:
    text = str(text).strip()
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in text)


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def first_nonempty(row: pd.Series, candidates: List[str]) -> str:
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return ""


def parse_seqid(seqid: str) -> Dict[str, str]:
    """
    解析第 17 步生成的 FASTA header。

    WT 示例：
        WT|COVID-19_2021a_AZD1061|chain_Q|esmif_A|af2esmif_0008_d409965e69

    mutant 示例：
        mut_000001|AA344E|A15E
    """
    seqid = clean_text(seqid)
    parts = seqid.split("|")

    out = {
        "record_type": "unknown",
        "mut_index": "",
        "mut_names": "",
        "esm_mut": "",
    }

    if not parts:
        return out

    if parts[0] == "WT":
        out["record_type"] = "WT"
        return out

    if parts[0].startswith("mut_"):
        out["record_type"] = "mutant"
        out["mut_index"] = parts[0]
        if len(parts) >= 2:
            out["mut_names"] = parts[1]
        if len(parts) >= 3:
            out["esm_mut"] = parts[2]
        return out

    return out


def load_score_file(score_csv: Path):
    score_df = pd.read_csv(score_csv)

    required = {"seqid", "log_likelihood"}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"{score_csv} missing columns: {sorted(missing)}")

    parsed = score_df["seqid"].map(parse_seqid).apply(pd.Series)
    score_df = pd.concat([score_df, parsed], axis=1)

    wt_rows = score_df[score_df["record_type"] == "WT"].copy()
    if wt_rows.empty:
        return score_df, None, "missing_wt_score"

    wt_ll = float(wt_rows["log_likelihood"].iloc[0])
    status = "ok" if len(wt_rows) == 1 else "multiple_wt_scores_take_first"
    return score_df, wt_ll, status


def build_score_maps(score_df: pd.DataFrame):
    mutants = score_df[score_df["record_type"] == "mutant"].copy()

    by_pair: Dict[Tuple[str, str], float] = {}
    by_esm_mut: Dict[str, float] = {}
    by_mut_names: Dict[str, float] = {}

    for _, r in mutants.iterrows():
        mut_names = clean_text(r.get("mut_names", ""))
        esm_mut = clean_text(r.get("esm_mut", ""))
        ll = float(r["log_likelihood"])

        if mut_names or esm_mut:
            by_pair[(mut_names, esm_mut)] = ll
        if esm_mut and esm_mut not in by_esm_mut:
            by_esm_mut[esm_mut] = ll
        if mut_names and mut_names not in by_mut_names:
            by_mut_names[mut_names] = ll

    return by_pair, by_esm_mut, by_mut_names


def expected_unique_score_name(input_filename: str, dms_name: str, pdb_chain_id: str, esm_if_chain_id: str, mode: str) -> str:
    input_stem = safe_name(Path(input_filename).stem)
    return (
        f"{input_stem}__"
        f"{safe_name(dms_name)}__"
        f"chain_{safe_name(pdb_chain_id)}__"
        f"esmif_{safe_name(esm_if_chain_id)}__"
        f"{mode}.csv"
    )


def expected_old_score_name(dms_name: str, esm_if_chain_id: str, mode: str) -> str:
    return f"{safe_name(dms_name)}__{safe_name(esm_if_chain_id)}_esmif_{mode}.csv"


def choose_score_file(score_dir: Path, input_filename: str, dms_name: str, pdb_chain_id: str, esm_if_chain_id: str, mode: str):
    """
    优先唯一命名文件；找不到再回退旧命名文件。
    返回：score_path, filename_type, expected_unique, expected_old
    """
    unique_name = expected_unique_score_name(input_filename, dms_name, pdb_chain_id, esm_if_chain_id, mode)
    old_name = expected_old_score_name(dms_name, esm_if_chain_id, mode)

    unique_path = score_dir / unique_name
    old_path = score_dir / old_name

    if unique_path.exists() and unique_path.stat().st_size > 0:
        return unique_path, "unique_name", unique_name, old_name

    if old_path.exists() and old_path.stat().st_size > 0:
        return old_path, "old_name", unique_name, old_name

    return None, "missing", unique_name, old_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ESM-IF-AF2 scores by scanning score directory and manifests.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录。")
    parser.add_argument(
        "--fasta-manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_fasta_manifest.csv",
        help="第 17 步 FASTA manifest。",
    )
    parser.add_argument(
        "--input-manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv",
        help="第 16 步 AF2 input manifest。",
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default="data_processed/esm_if_af2_scores/single",
        help="第 18 步输出的 score CSV 目录。",
    )
    parser.add_argument(
        "--esm-input-dir",
        type=str,
        default="data_processed/esm_inputs",
        help="原始 esm_inputs 目录。",
    )
    parser.add_argument("--mode", type=str, default="single", help="ESM-IF 模式，默认 single。")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="data_processed/esm_if_af2_scores/esmif_af2_scores_merged.csv",
        help="合并后的逐突变结果表。",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data_processed/esm_if_af2_scores/esmif_af2_scores_merge_summary.csv",
        help="合并统计表。",
    )
    parser.add_argument(
        "--allow-old-name",
        action="store_true",
        default=True,
        help="允许回退使用旧命名文件。默认允许。",
    )
    parser.add_argument(
        "--no-old-name",
        action="store_true",
        help="禁止使用旧命名文件，只接受唯一命名文件。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    fasta_manifest_csv = (project_root / args.fasta_manifest_csv).resolve()
    input_manifest_csv = (project_root / args.input_manifest_csv).resolve()
    score_dir = (project_root / args.score_dir).resolve()
    esm_input_dir = (project_root / args.esm_input_dir).resolve()
    out_csv = (project_root / args.out_csv).resolve()
    summary_csv = (project_root / args.summary_csv).resolve()
    mode = args.mode

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not fasta_manifest_csv.exists():
        raise FileNotFoundError(f"FASTA manifest not found: {fasta_manifest_csv}")
    if not input_manifest_csv.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest_csv}")
    if not score_dir.exists():
        raise FileNotFoundError(f"Score dir not found: {score_dir}")
    if not esm_input_dir.exists():
        raise FileNotFoundError(f"ESM input dir not found: {esm_input_dir}")

    fasta_manifest = pd.read_csv(fasta_manifest_csv)
    input_manifest = pd.read_csv(input_manifest_csv)

    required_fasta_cols = {"input_filename", "DMS_name", "pdb_chain_id", "fasta_path"}
    missing_fasta = required_fasta_cols - set(fasta_manifest.columns)
    if missing_fasta:
        raise ValueError(f"FASTA manifest missing columns: {sorted(missing_fasta)}")

    required_input_cols = {"input_filename", "DMS_name", "pdb_chain_id", "can_run_esm_if", "esm_if_chain_id"}
    missing_input = required_input_cols - set(input_manifest.columns)
    if missing_input:
        raise ValueError(f"Input manifest missing columns: {sorted(missing_input)}")

    merged_manifest = fasta_manifest.merge(
        input_manifest[[
            "input_filename",
            "DMS_name",
            "pdb_chain_id",
            "can_run_esm_if",
            "esm_if_chain_id",
            "af2_seq_id",
            "af2_structure_path",
        ]],
        on=["input_filename", "DMS_name", "pdb_chain_id"],
        how="left",
        suffixes=("_fasta", "_manifest"),
    )

    runnable = merged_manifest[merged_manifest["can_run_esm_if"].fillna(0).astype(int) == 1].copy()
    runnable = runnable.sort_values(["DMS_name", "pdb_chain_id", "input_filename"]).reset_index(drop=True)

    merged_tables: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    use_old_name = not args.no_old_name

    for _, row in runnable.iterrows():
        input_filename = clean_text(row["input_filename"])
        dms_name = clean_text(row["DMS_name"])
        pdb_chain_id = clean_text(row["pdb_chain_id"])
        esm_if_chain_id = first_nonempty(row, ["esm_if_chain_id_manifest", "esm_if_chain_id_fasta", "esm_if_chain_id"])
        if not esm_if_chain_id:
            esm_if_chain_id = "A"

        af2_seq_id = first_nonempty(row, ["af2_seq_id_manifest", "af2_seq_id_fasta", "af2_seq_id"])
        af2_structure_path = first_nonempty(row, ["af2_structure_path_manifest", "af2_structure_path_fasta", "af2_structure_path"])

        score_path, filename_type, unique_name, old_name = choose_score_file(
            score_dir=score_dir,
            input_filename=input_filename,
            dms_name=dms_name,
            pdb_chain_id=pdb_chain_id,
            esm_if_chain_id=esm_if_chain_id,
            mode=mode,
        )

        if filename_type == "old_name" and not use_old_name:
            score_path = None
            filename_type = "missing_old_name_disabled"

        esm_input_path = esm_input_dir / input_filename

        base_summary = {
            "input_filename": input_filename,
            "DMS_name": dms_name,
            "pdb_chain_id": pdb_chain_id,
            "esm_if_chain_id": esm_if_chain_id,
            "af2_seq_id": af2_seq_id,
            "af2_structure_path": af2_structure_path,
            "expected_unique_score_name": unique_name,
            "expected_old_score_name": old_name,
            "score_filename_type": filename_type,
            "score_csv": str(score_path) if score_path is not None else "",
        }

        if not esm_input_path.exists():
            summary_rows.append({
                **base_summary,
                "status": "missing_esm_input",
                "n_input_rows": 0,
                "n_scored_mutants": 0,
                "n_merged": 0,
                "n_missing_score": 0,
                "wt_ll_status": "",
                "wt_ll": pd.NA,
            })
            continue

        if score_path is None or not score_path.exists() or score_path.stat().st_size == 0:
            input_df = pd.read_csv(esm_input_path)
            work = input_df.copy()
            work["esmif_af2_wt_ll"] = pd.NA
            work["esmif_af2_mut_ll"] = pd.NA
            work["esmif_af2_delta_ll"] = pd.NA
            work["esmif_af2_neg_delta_ll"] = pd.NA
            work["esmif_af2_score_source"] = ""
            work["esmif_af2_merge_status"] = "missing_score_csv"
            work["esmif_af2_score_csv"] = ""
            work["esmif_af2_score_filename_type"] = filename_type
            work["esmif_af2_expected_unique_score_name"] = unique_name
            work["esmif_af2_expected_old_score_name"] = old_name
            merged_tables.append(work)

            summary_rows.append({
                **base_summary,
                "status": "missing_score_csv",
                "n_input_rows": len(input_df),
                "n_scored_mutants": 0,
                "n_merged": 0,
                "n_missing_score": len(input_df),
                "wt_ll_status": "",
                "wt_ll": pd.NA,
            })
            continue

        input_df = pd.read_csv(esm_input_path)
        score_df, wt_ll, wt_ll_status = load_score_file(score_path)
        by_pair, by_esm_mut, by_mut_names = build_score_maps(score_df)

        work = input_df.copy()
        work["esmif_af2_wt_ll"] = wt_ll
        work["esmif_af2_mut_ll"] = pd.NA
        work["esmif_af2_delta_ll"] = pd.NA
        work["esmif_af2_neg_delta_ll"] = pd.NA
        work["esmif_af2_score_source"] = ""
        work["esmif_af2_merge_status"] = "missing_score"
        work["esmif_af2_score_csv"] = str(score_path)
        work["esmif_af2_score_filename_type"] = filename_type
        work["esmif_af2_expected_unique_score_name"] = unique_name
        work["esmif_af2_expected_old_score_name"] = old_name

        for idx, r in work.iterrows():
            mut_names = clean_text(r.get("mut_names", ""))
            esm_mut = clean_text(r.get("esm_mut", ""))

            mut_ll = None
            source = ""

            if (mut_names, esm_mut) in by_pair:
                mut_ll = by_pair[(mut_names, esm_mut)]
                source = "mut_names+esm_mut"
            elif esm_mut in by_esm_mut:
                mut_ll = by_esm_mut[esm_mut]
                source = "esm_mut"
            elif mut_names in by_mut_names:
                mut_ll = by_mut_names[mut_names]
                source = "mut_names"

            if mut_ll is not None and wt_ll is not None:
                delta = float(mut_ll) - float(wt_ll)
                work.at[idx, "esmif_af2_mut_ll"] = float(mut_ll)
                work.at[idx, "esmif_af2_delta_ll"] = delta
                work.at[idx, "esmif_af2_neg_delta_ll"] = -delta
                work.at[idx, "esmif_af2_score_source"] = source
                work.at[idx, "esmif_af2_merge_status"] = "ok"

        n_input = len(work)
        n_ok = int((work["esmif_af2_merge_status"] == "ok").sum())
        n_missing = n_input - n_ok
        n_scored_mutants = int((score_df["record_type"] == "mutant").sum())

        merged_tables.append(work)
        summary_rows.append({
            **base_summary,
            "status": "ok",
            "n_input_rows": n_input,
            "n_scored_mutants": n_scored_mutants,
            "n_merged": n_ok,
            "n_missing_score": n_missing,
            "wt_ll_status": wt_ll_status,
            "wt_ll": wt_ll,
        })

    merged = pd.concat(merged_tables, ignore_index=True) if merged_tables else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    merged.to_csv(out_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"[OK] Merged ESM-IF-AF2 scores written to: {out_csv}")
    print(f"[OK] Merge summary written to: {summary_csv}")
    print(f"[INFO] Runnable assay/chain rows considered: {len(runnable)}")
    print(f"[INFO] Total merged variant rows: {len(merged)}")
    if not summary.empty:
        print("[INFO] Score filename type counts:")
        print(summary["score_filename_type"].value_counts(dropna=False).to_string())
        print("[INFO] Summary status counts:")
        print(summary["status"].value_counts(dropna=False).to_string())
    if not merged.empty:
        print(f"[INFO] Rows with ESM-IF score: {(merged['esmif_af2_merge_status'] == 'ok').sum()}")
        print(f"[INFO] Rows missing ESM-IF score: {(merged['esmif_af2_merge_status'] != 'ok').sum()}")


if __name__ == "__main__":
    main()

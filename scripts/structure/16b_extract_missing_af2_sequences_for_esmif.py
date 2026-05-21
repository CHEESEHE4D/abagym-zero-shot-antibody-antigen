#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    16b_extract_missing_af2_sequences_for_esmif.py

这个脚本做什么：
    从 ESM-IF-AF2 主线的 excluded manifest 中，提取因为
    af2_unique_manifest_not_found 而被排除的 WT 序列。
    对这些 WT 序列按 seq_md5 去重后，生成用于补充 AF2 / ColabFold 预测的 FASTA
    和对应 manifest。

为什么需要这个脚本：
    ESM-IF-AF2 主线要求每个 assay/chain 都有与 WT 全长序列一一对应的 AF2 结构。
    如果某些 WT 序列没有出现在前期 SaProt 的 AF2 清单中，就会在第 16 步被标记为：
        af2_unique_manifest_not_found
    这些样本不是因为 AF2 质量低被排除，而是因为还没有对应 AF2 预测结果。
    本脚本用于把这些缺失的唯一 WT 序列提取出来，供后续补充 AF2 预测。

这个脚本读取什么数据：
    1. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest_excluded.csv
       - 来自 16_prepare_esmif_af2_inputs.py
       - 包含被排除样本及 exclude_reason
       - 本脚本筛选 exclude_reason 中包含 af2_unique_manifest_not_found 的条目

这个脚本输出什么：
    1. data_processed/esm_if_af2_inputs/af2_missing_wt_sequences.fasta
       - 用于 colabfold_batch 的补充预测输入 FASTA
       - 每条记录是一个唯一 WT 序列
       - header 使用 af2esmif_<编号>_<seq_md5>

    2. data_processed/esm_if_af2_inputs/af2_missing_wt_sequences_manifest.csv
       - 每条唯一 WT 序列对应一行
       - 包含：
           * af2_seq_id
           * seq_md5
           * wt_length
           * selected_wt_sequence
           * n_assays
           * assays
           * input_filenames
           * pdb_ids
           * pdb_chain_ids
           * recommended_colabfold_input
           * expected_output_dir

    3. data_processed/esm_if_af2_inputs/af2_missing_wt_sequences_assay_map.csv
       - assay/chain 到唯一 WT 序列的映射表
       - 用于后续把新 AF2 结果合并回 ESM-IF manifest

论文写作时可以怎么描述：
    对于未能在既有 AF2 结构清单中匹配到的 WT 序列，
    本研究进一步从 ESM-IF 候选样本中提取唯一 WT 序列并补充进行 AF2 预测。
    补充预测仍按唯一序列去重，以避免相同 WT 序列在多个 assay 中重复预测。

使用示例：
    python scripts/structure/16b_extract_missing_af2_sequences_for_esmif.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym

后续 ColabFold 命令示例：
    colabfold_batch \
        data_processed/esm_if_af2_inputs/af2_missing_wt_sequences.fasta \
        raw_data/af2_predictions/esmif_missing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def clean_sequence(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return "".join(str(seq).split()).upper()


def write_fasta(records: List[tuple[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def join_unique(values) -> str:
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in vals:
            vals.append(s)
    return ";".join(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract missing WT sequences for supplemental AF2 prediction.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--excluded-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest_excluded.csv",
        help="第 16 步生成的 excluded manifest。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_processed/esm_if_af2_inputs",
        help="输出目录。",
    )
    parser.add_argument(
        "--expected-output-dir",
        type=str,
        default="raw_data/af2_predictions/esmif_missing",
        help="建议的 ColabFold 输出目录，仅写入 manifest 作为记录。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="af2esmif",
        help="补充 AF2 序列 ID 前缀。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    excluded_csv = (project_root / args.excluded_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_out = out_dir / "af2_missing_wt_sequences.fasta"
    manifest_out = out_dir / "af2_missing_wt_sequences_manifest.csv"
    assay_map_out = out_dir / "af2_missing_wt_sequences_assay_map.csv"

    if not excluded_csv.exists():
        raise FileNotFoundError(f"Excluded manifest not found: {excluded_csv}")

    df = pd.read_csv(excluded_csv)

    required_cols = {"DMS_name", "pdb_id", "pdb_chain_id", "wt_sequence", "sequence_length", "seq_md5", "exclude_reason", "input_filename"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excluded manifest missing required columns: {sorted(missing)}")

    miss = df[
        df["exclude_reason"].astype(str).str.contains("af2_unique_manifest_not_found", na=False)
    ].copy()

    if miss.empty:
        pd.DataFrame().to_csv(manifest_out, index=False)
        pd.DataFrame().to_csv(assay_map_out, index=False)
        fasta_out.write_text("", encoding="utf-8")
        print("[INFO] No missing AF2 WT sequences found.")
        print(f"[OK] Empty FASTA written to: {fasta_out}")
        print(f"[OK] Empty manifest written to: {manifest_out}")
        print(f"[OK] Empty assay map written to: {assay_map_out}")
        return

    miss["wt_sequence_clean"] = miss["wt_sequence"].map(clean_sequence)
    miss = miss[miss["wt_sequence_clean"] != ""].copy()

    # seq_md5 缺失时，用序列本身分组；正常情况下第 16 步已经有 seq_md5
    grouped_rows = []
    assay_map_rows = []
    fasta_records = []

    grouped = miss.groupby("seq_md5", dropna=False)

    idx = 1
    for seq_md5, g in grouped:
        seq_md5 = str(seq_md5).strip()
        wt_seqs = g["wt_sequence_clean"].dropna().unique().tolist()

        if len(wt_seqs) == 0:
            continue

        if len(wt_seqs) > 1:
            # 理论上同一个 seq_md5 不应对应多条不同序列；如果出现，按序列再拆
            subgroups = [(f"{seq_md5}_sub{i+1}", pd.DataFrame({"wt_sequence_clean": [s]}), s) for i, s in enumerate(wt_seqs)]
        else:
            subgroups = [(seq_md5, g, wt_seqs[0])]

        for sub_md5, sub_g, wt_seq in subgroups:
            af2_seq_id = f"{args.prefix}_{idx:04d}_{str(sub_md5)[:10]}"
            idx += 1

            assays = join_unique(g["DMS_name"])
            input_filenames = join_unique(g["input_filename"])
            pdb_ids = join_unique(g["pdb_id"])
            pdb_chain_ids = join_unique(g["pdb_chain_id"])

            fasta_records.append((af2_seq_id, wt_seq))

            grouped_rows.append({
                "af2_seq_id": af2_seq_id,
                "seq_md5": sub_md5,
                "wt_length": len(wt_seq),
                "selected_wt_sequence": wt_seq,
                "n_assays": g["DMS_name"].nunique(),
                "assays": assays,
                "input_filenames": input_filenames,
                "pdb_ids": pdb_ids,
                "pdb_chain_ids": pdb_chain_ids,
                "recommended_colabfold_input": str(fasta_out.relative_to(project_root)),
                "expected_output_dir": args.expected_output_dir,
                "source_note": "extracted_from_esm_if_af2_excluded_manifest_due_to_af2_unique_manifest_not_found",
            })

            for _, r in g.iterrows():
                assay_map_rows.append({
                    "af2_seq_id": af2_seq_id,
                    "seq_md5": sub_md5,
                    "input_filename": r.get("input_filename", ""),
                    "DMS_name": r.get("DMS_name", ""),
                    "pdb_id": r.get("pdb_id", ""),
                    "pdb_chain_id": r.get("pdb_chain_id", ""),
                    "sequence_length": r.get("sequence_length", ""),
                    "exclude_reason": r.get("exclude_reason", ""),
                })

    write_fasta(fasta_records, fasta_out)

    manifest = pd.DataFrame(grouped_rows)
    assay_map = pd.DataFrame(assay_map_rows)

    if not manifest.empty:
        manifest = manifest.sort_values(["af2_seq_id"]).reset_index(drop=True)
    if not assay_map.empty:
        assay_map = assay_map.sort_values(["af2_seq_id", "DMS_name", "pdb_chain_id"]).reset_index(drop=True)

    manifest.to_csv(manifest_out, index=False)
    assay_map.to_csv(assay_map_out, index=False)

    print(f"[OK] Missing WT FASTA written to: {fasta_out}")
    print(f"[OK] Missing WT manifest written to: {manifest_out}")
    print(f"[OK] Missing WT assay map written to: {assay_map_out}")
    print(f"[INFO] Excluded rows with missing AF2: {len(miss)}")
    print(f"[INFO] Unique missing WT sequences: {len(manifest)}")
    print(f"[INFO] Total assays represented: {miss['DMS_name'].nunique()}")
    print(f"[NEXT] Run ColabFold, for example:")
    print(f"colabfold_batch {fasta_out.relative_to(project_root)} {args.expected_output_dir}")


if __name__ == "__main__":
    main()

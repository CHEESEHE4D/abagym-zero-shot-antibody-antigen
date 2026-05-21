#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    17_build_esm_if_fastas_af2.py

这个脚本做什么：
    根据 AF2 主线的 ESM-IF 输入清单（manifest）和 esm_inputs/*.csv，
    为每个可纳入 ESM-IF 分析的 assay/chain 生成全长 FASTA 文件。

为什么这一版不裁剪：
    在 AF2 主线中，结构文件来自 WT 全长序列的一对一预测结果。
    因此 WT / mutant 序列直接使用完整全长即可，不需要实验结构路线中的局部裁剪。

这个脚本读取什么数据：
    1. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv
       - 来自 16_prepare_esmif_af2_inputs.py
       - 提供：
           * input_filename
           * DMS_name
           * pdb_chain_id
           * af2_structure_path
           * esm_if_chain_id
           * can_run_esm_if

    2. data_processed/esm_inputs/*.csv
       - 提供：
           * wt_sequence
           * mutant_sequence
           * mut_names
           * esm_mut

这个脚本输出什么：
    1. data_processed/esm_if_af2_inputs/fasta/*.fasta
       - 每个 assay/chain 一个 FASTA
       - 第一条为 WT，后续为 mutant 全长序列

    2. data_processed/esm_if_af2_inputs/esm_if_fasta_manifest.csv
       - 记录 FASTA 文件与条目数、路径等

    3. data_processed/esm_if_af2_inputs/esm_if_fasta_skipped.csv
       - 记录未生成 FASTA 的条目及原因

论文写作时可以怎么描述：
    在 AF2 主线中，ESM-IF 输入结构来自与 WT 序列一一对应的全长 AF2 预测结果，
    因此 WT 与 mutant 序列均以完整全长形式写入 FASTA，无需再做与实验结构片段的局部裁剪。

使用示例：
    python scripts/structure/17_build_esm_if_fastas_af2.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def sanitize_header(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.|:+\-]", "_", text)
    return text


def clean_sequence(seq: str) -> str:
    seq = "" if pd.isna(seq) else str(seq)
    return re.sub(r"\s+", "", seq).upper()


def write_fasta(records: List[Tuple[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def pick_first_unique_value(series: pd.Series) -> tuple[str, int]:
    vals = series.dropna().astype(str).map(clean_sequence)
    vals = vals[vals != ""]
    uniq = vals.unique().tolist()
    return (uniq[0] if uniq else "", len(uniq))


def build_mutant_header(row: pd.Series, fallback_index: int) -> str:
    mut_names = sanitize_header(row.get("mut_names", ""))
    esm_mut = sanitize_header(row.get("esm_mut", ""))
    wildtype = sanitize_header(row.get("wildtype", ""))
    mutation = sanitize_header(row.get("mutation", ""))

    parts = [f"mut_{fallback_index:06d}"]
    if mut_names:
        parts.append(mut_names)
    if esm_mut:
        parts.append(esm_mut)
    elif wildtype and mutation:
        parts.append(f"{wildtype}_to_{mutation}")
    return "|".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AF2-based ESM-IF FASTA files.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录，默认当前目录。")
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv",
        help="AF2 主线 manifest 路径。",
    )
    parser.add_argument(
        "--esm-input-dir",
        type=str,
        default="data_processed/esm_inputs",
        help="esm_inputs 目录。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_processed/esm_if_af2_inputs",
        help="ESM-IF AF2 输入输出根目录。",
    )
    parser.add_argument(
        "--keep-duplicate-mutants",
        action="store_true",
        help="保留重复 mutant_sequence；默认按全长 mutant_sequence 去重。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_csv = (project_root / args.manifest_csv).resolve()
    esm_input_dir = (project_root / args.esm_input_dir).resolve()
    out_root = (project_root / args.out_dir).resolve()
    fasta_dir = out_root / "fasta"

    out_root.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest csv not found: {manifest_csv}")
    if not esm_input_dir.exists():
        raise FileNotFoundError(f"ESM input dir not found: {esm_input_dir}")

    manifest = pd.read_csv(manifest_csv)

    required_manifest_cols = {
        "input_filename",
        "DMS_name",
        "pdb_chain_id",
        "can_run_esm_if",
        "esm_if_chain_id",
        "af2_structure_path",
    }
    missing = required_manifest_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    runnable = manifest[manifest["can_run_esm_if"].fillna(0).astype(int) == 1].copy()

    fasta_rows = []
    skipped_rows = []

    for _, row in runnable.iterrows():
        input_filename = str(row["input_filename"]).strip()
        dms_name = str(row["DMS_name"]).strip()
        pdb_chain_id = str(row["pdb_chain_id"]).strip()
        esm_if_chain_id = str(row["esm_if_chain_id"]).strip()
        af2_structure_path = str(row["af2_structure_path"]).strip()
        af2_seq_id = str(row.get("af2_seq_id", "")).strip()

        esm_path = esm_input_dir / input_filename
        if not esm_path.exists():
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": "esm_input_file_not_found",
            })
            continue

        df = pd.read_csv(esm_path)
        required_cols = {"wt_sequence", "mutant_sequence"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": f"missing_columns:{','.join(sorted(missing_cols))}",
            })
            continue

        wt_sequence, wt_unique_count = pick_first_unique_value(df["wt_sequence"])

        if not wt_sequence:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": "missing_wt_sequence",
            })
            continue

        if wt_unique_count != 1:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": f"wt_sequence_not_unique:{wt_unique_count}",
            })
            continue

        records: List[Tuple[str, str]] = []
        wt_header = sanitize_header(
            f"WT|{dms_name}|chain_{pdb_chain_id}|esmif_{esm_if_chain_id}|{af2_seq_id}"
        )
        records.append((wt_header, wt_sequence))

        work_df = df.copy()
        work_df["mutant_sequence_clean"] = work_df["mutant_sequence"].map(clean_sequence)
        work_df = work_df[work_df["mutant_sequence_clean"] != ""].copy()

        if work_df.empty:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": "no_valid_mutant_sequence",
            })
            continue

        # 长度保护：ESM-IF 要求 FASTA 中所有序列和结构长度一致
        work_df = work_df[work_df["mutant_sequence_clean"].map(len) == len(wt_sequence)].copy()
        if work_df.empty:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": pdb_chain_id,
                "reason": "no_mutant_sequence_with_wt_length",
            })
            continue

        n_input_mutants = len(work_df)

        if not args.keep_duplicate_mutants:
            work_df = work_df.drop_duplicates(subset=["mutant_sequence_clean"], keep="first").copy()

        n_written_mutants = 0
        seen_headers = set()

        for idx, (_, r) in enumerate(work_df.iterrows(), start=1):
            seq = r["mutant_sequence_clean"]
            header = build_mutant_header(r, idx)
            base_header = header
            suffix_i = 2
            while header in seen_headers:
                header = f"{base_header}|dup{suffix_i}"
                suffix_i += 1
            seen_headers.add(header)
            records.append((header, seq))
            n_written_mutants += 1

        out_name = Path(input_filename).with_suffix(".fasta").name
        out_path = fasta_dir / out_name
        write_fasta(records, out_path)

        fasta_rows.append({
            "input_filename": input_filename,
            "DMS_name": dms_name,
            "pdb_chain_id": pdb_chain_id,
            "esm_if_chain_id": esm_if_chain_id,
            "af2_seq_id": af2_seq_id,
            "af2_structure_path": af2_structure_path,
            "fasta_filename": out_name,
            "fasta_path": str(out_path),
            "wt_header": wt_header,
            "sequence_length": len(wt_sequence),
            "n_total_records_written": len(records),
            "n_mutants_input": n_input_mutants,
            "n_mutants_written": n_written_mutants,
            "deduplicated_mutants": int(not args.keep_duplicate_mutants),
            "crop_applied": 0,
        })

    fasta_manifest = pd.DataFrame(fasta_rows)
    skipped = pd.DataFrame(skipped_rows)

    fasta_manifest_csv = out_root / "esm_if_fasta_manifest.csv"
    skipped_csv = out_root / "esm_if_fasta_skipped.csv"

    if not fasta_manifest.empty:
        fasta_manifest = fasta_manifest.sort_values(["DMS_name", "pdb_chain_id"]).reset_index(drop=True)
    if not skipped.empty:
        skipped = skipped.sort_values(["DMS_name", "pdb_chain_id"]).reset_index(drop=True)

    fasta_manifest.to_csv(fasta_manifest_csv, index=False)
    skipped.to_csv(skipped_csv, index=False)

    print(f"[OK] FASTA manifest written to: {fasta_manifest_csv}")
    print(f"[OK] FASTA skipped report written to: {skipped_csv}")
    print(f"[INFO] FASTA files written: {len(fasta_manifest)}")
    print(f"[INFO] FASTA files skipped: {len(skipped)}")
    if not fasta_manifest.empty:
        print(f"[INFO] Total mutant sequences written: {int(fasta_manifest['n_mutants_written'].sum())}")


if __name__ == "__main__":
    main()

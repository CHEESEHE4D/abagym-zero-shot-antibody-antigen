#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    18_run_esm_if_af2.py

这个脚本做什么：
    在 AF2 主线下调用 ESM 官方 inverse folding 打分脚本 score_log_likelihoods.py，
    对每个 assay/chain 的 WT 与 mutant 全长序列进行结构条件打分。

为什么这一版适合 AF2 主线：
    - 结构文件直接来自 AF2 全长预测结果
    - 结构长度与输入 WT / mutant 全长序列一致
    - 不依赖实验结构链解析、局部裁剪或 resolved / low_confidence 分类

这个脚本读取什么数据：
    1. data_processed/esm_if_af2_inputs/esm_if_fasta_manifest.csv
       - 来自 17_build_esm_if_fastas_af2.py
       - 提供 FASTA 路径

    2. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv
       - 来自 16_prepare_esmif_af2_inputs.py
       - 提供 AF2 结构路径、esm_if_chain_id、是否可运行等信息

    3. ESM 官方脚本 score_log_likelihoods.py
       - 实际执行 inverse folding 条件对数似然打分

这个脚本输出什么：
    1. data_processed/esm_if_af2_scores/<mode>/*.csv
    2. data_processed/esm_if_af2_scores/<mode>/*.log
    3. data_processed/esm_if_af2_scores/esm_if_run_manifest_<mode>.csv
    4. data_processed/esm_if_af2_scores/esm_if_run_skipped_<mode>.csv
    5. data_processed/esm_if_af2_scores/esm_if_run_summary_<mode>.csv

使用示例：
    python scripts/structure/18_run_esm_if_af2.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym \
        --esm-script /public/home/huangwenle/tools/esm/examples/inverse_folding/score_log_likelihoods.py \
        --mode single \
        --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def safe_name(text: str) -> str:
    text = str(text).strip()
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in text)


def first_nonempty(row: pd.Series, candidates: List[str]) -> str:
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AF2-based ESM-IF scoring for assay/chain FASTA files.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录，默认当前目录。")
    parser.add_argument("--esm-script", type=str, required=True, help="ESM 官方 score_log_likelihoods.py 的绝对路径。")
    parser.add_argument(
        "--fasta-manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_fasta_manifest.csv",
        help="AF2 主线 FASTA 清单路径。",
    )
    parser.add_argument(
        "--input-manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv",
        help="AF2 主线输入 manifest 路径。",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data_processed/esm_if_af2_scores",
        help="输出根目录。",
    )
    parser.add_argument("--mode", choices=["single", "multichain"], default="single", help="打分模式：single 或 multichain。")
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="用于调用 ESM 官方脚本的 python 可执行文件，默认当前解释器。",
    )
    parser.add_argument("--overwrite", action="store_true", help="若输出结果已存在，是否覆盖重跑。默认跳过已有结果。")
    parser.add_argument("--dry-run", action="store_true", help="只生成运行清单与命令，不真正执行。")
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 个任务，0 表示不限制。")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    esm_script = Path(args.esm_script).resolve()
    fasta_manifest_csv = (project_root / args.fasta_manifest_csv).resolve()
    input_manifest_csv = (project_root / args.input_manifest_csv).resolve()
    out_root = (project_root / args.out_root).resolve()
    mode = args.mode

    if not esm_script.exists():
        raise FileNotFoundError(f"ESM script not found: {esm_script}")
    if not fasta_manifest_csv.exists():
        raise FileNotFoundError(f"FASTA manifest not found: {fasta_manifest_csv}")
    if not input_manifest_csv.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest_csv}")

    mode_dir = out_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_csv = out_root / f"esm_if_run_manifest_{mode}.csv"
    run_skipped_csv = out_root / f"esm_if_run_skipped_{mode}.csv"
    run_summary_csv = out_root / f"esm_if_run_summary_{mode}.csv"

    fasta_manifest = pd.read_csv(fasta_manifest_csv)
    input_manifest = pd.read_csv(input_manifest_csv)

    required_fasta_cols = {"input_filename", "DMS_name", "pdb_chain_id", "fasta_path"}
    missing_fasta_cols = required_fasta_cols - set(fasta_manifest.columns)
    if missing_fasta_cols:
        raise ValueError(f"FASTA manifest missing columns: {sorted(missing_fasta_cols)}")

    required_input_cols = {"input_filename", "DMS_name", "pdb_chain_id", "af2_structure_path", "can_run_esm_if", "esm_if_chain_id"}
    missing_input_cols = required_input_cols - set(input_manifest.columns)
    if missing_input_cols:
        raise ValueError(f"Input manifest missing columns: {sorted(missing_input_cols)}")

    merged = fasta_manifest.merge(
        input_manifest[[
            "input_filename",
            "DMS_name",
            "pdb_chain_id",
            "af2_structure_path",
            "can_run_esm_if",
            "esm_if_chain_id",
            "af2_seq_id",
        ]],
        on=["input_filename", "DMS_name", "pdb_chain_id"],
        how="left",
        suffixes=("_fasta", "_manifest"),
    )

    merged = merged[merged["can_run_esm_if"].fillna(0).astype(int) == 1].copy()
    merged = merged.sort_values(["DMS_name", "pdb_chain_id"]).reset_index(drop=True)

    if args.limit and args.limit > 0:
        merged = merged.head(args.limit).copy()

    run_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    for _, row in merged.iterrows():
        dms_name = str(row["DMS_name"]).strip()
        original_chain = str(row["pdb_chain_id"]).strip()
        esm_if_chain = first_nonempty(row, ["esm_if_chain_id_manifest", "esm_if_chain_id_fasta", "esm_if_chain_id"])
        chain_for_run = esm_if_chain if esm_if_chain else "A"
        af2_seq_id = first_nonempty(row, ["af2_seq_id_manifest", "af2_seq_id_fasta", "af2_seq_id"])

        fasta_path = Path(str(row["fasta_path"])).resolve()

        structure_path_str = first_nonempty(
            row,
            ["af2_structure_path_manifest", "af2_structure_path_fasta", "af2_structure_path"]
        )
        structure_path = Path(structure_path_str).resolve() if structure_path_str else None

        input_filename = str(row["input_filename"]).strip()

        input_stem = safe_name(Path(input_filename).stem)
        base_name = f"{input_stem}__{safe_name(dms_name)}__chain_{safe_name(original_chain)}__esmif_{safe_name(chain_for_run)}__{mode}"

        out_csv = mode_dir / f"{base_name}.csv"
        out_log = mode_dir / f"{base_name}.log"

        if structure_path is None or not structure_path.exists():
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": original_chain,
                "esm_if_chain_id": esm_if_chain,
                "af2_seq_id": af2_seq_id,
                "chain_used_for_run": chain_for_run,
                "mode": mode,
                "reason": "missing_structure_path",
                "fasta_path": str(fasta_path),
                "structure_path": "" if structure_path is None else str(structure_path),
                "out_csv": str(out_csv),
            })
            continue

        if not fasta_path.exists():
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": original_chain,
                "esm_if_chain_id": esm_if_chain,
                "af2_seq_id": af2_seq_id,
                "chain_used_for_run": chain_for_run,
                "mode": mode,
                "reason": "missing_fasta_path",
                "fasta_path": str(fasta_path),
                "structure_path": str(structure_path),
                "out_csv": str(out_csv),
            })
            continue

        if out_csv.exists() and not args.overwrite:
            skipped_rows.append({
                "input_filename": input_filename,
                "DMS_name": dms_name,
                "pdb_chain_id": original_chain,
                "esm_if_chain_id": esm_if_chain,
                "af2_seq_id": af2_seq_id,
                "chain_used_for_run": chain_for_run,
                "mode": mode,
                "reason": "output_exists",
                "fasta_path": str(fasta_path),
                "structure_path": str(structure_path),
                "out_csv": str(out_csv),
            })
            continue

        cmd = [
            args.python_bin,
            str(esm_script),
            str(structure_path),
            str(fasta_path),
            "--chain",
            chain_for_run,
            "--outpath",
            str(out_csv),
        ]
        if mode == "multichain":
            cmd.append("--multichain-backbone")

        cmd_str = quote_cmd(cmd)
        status = "dry_run"
        returncode = None
        error_message = ""

        if args.dry_run:
            out_log.write_text(cmd_str + "\n", encoding="utf-8")
        else:
            with out_log.open("w", encoding="utf-8") as log_f:
                log_f.write("[COMMAND]\n")
                log_f.write(cmd_str + "\n\n")
                log_f.flush()
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
                returncode = proc.returncode

            if returncode == 0 and out_csv.exists() and out_csv.stat().st_size > 0:
                status = "ok"
            else:
                status = "failed"
                error_message = f"returncode={returncode}; out_exists={out_csv.exists()}; out_size={out_csv.stat().st_size if out_csv.exists() else 0}"

        run_rows.append({
            "input_filename": input_filename,
            "DMS_name": dms_name,
            "pdb_chain_id": original_chain,
            "esm_if_chain_id": esm_if_chain,
            "af2_seq_id": af2_seq_id,
            "chain_used_for_run": chain_for_run,
            "mode": mode,
            "fasta_path": str(fasta_path),
            "structure_path": str(structure_path),
            "out_csv": str(out_csv),
            "out_log": str(out_log),
            "command": cmd_str,
            "status": status,
            "returncode": returncode,
            "error_message": error_message,
        })

    run_manifest = pd.DataFrame(run_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if not run_manifest.empty:
        run_manifest = run_manifest.sort_values(["DMS_name", "chain_used_for_run"]).reset_index(drop=True)
    if not skipped_df.empty:
        skipped_df = skipped_df.sort_values(["DMS_name", "chain_used_for_run"]).reset_index(drop=True)

    run_manifest.to_csv(run_manifest_csv, index=False)
    skipped_df.to_csv(run_skipped_csv, index=False)

    summary = pd.DataFrame([{
        "mode": mode,
        "n_tasks_considered": len(merged),
        "n_run_rows": len(run_manifest),
        "n_ok": int((run_manifest["status"] == "ok").sum()) if not run_manifest.empty else 0,
        "n_failed": int((run_manifest["status"] == "failed").sum()) if not run_manifest.empty else 0,
        "n_dry_run": int((run_manifest["status"] == "dry_run").sum()) if not run_manifest.empty else 0,
        "n_skipped": len(skipped_df),
        "overwrite": int(args.overwrite),
        "dry_run": int(args.dry_run),
    }])
    summary.to_csv(run_summary_csv, index=False)

    print(f"[OK] Run manifest written to: {run_manifest_csv}")
    print(f"[OK] Skipped report written to: {run_skipped_csv}")
    print(f"[OK] Summary written to: {run_summary_csv}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Tasks considered: {len(merged)}")
    print(f"[INFO] Run rows: {len(run_manifest)}")
    if not run_manifest.empty:
        print(f"[INFO] OK: {(run_manifest['status'] == 'ok').sum()}")
        print(f"[INFO] Failed: {(run_manifest['status'] == 'failed').sum()}")
        print(f"[INFO] Dry-run: {(run_manifest['status'] == 'dry_run').sum()}")
    print(f"[INFO] Skipped: {len(skipped_df)}")


if __name__ == "__main__":
    main()

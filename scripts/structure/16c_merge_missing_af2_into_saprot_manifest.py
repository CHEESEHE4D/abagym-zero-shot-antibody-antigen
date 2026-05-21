#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    16c_merge_missing_af2_into_saprot_manifest.py

这个脚本做什么：
    将 ESM-IF-AF2 主线中补充预测的 AF2 结构结果，直接合并回原来的 SaProt AF2 清单，
    使后续 16_prepare_esmif_af2_inputs.py 可以继续读取默认的 AF2 manifest 和 summary。

为什么需要这个脚本：
    第 16 步中出现 af2_unique_manifest_not_found，表示某些 WT 序列没有在既有
    data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv 中登记。
    这些序列已经通过 16b_extract_missing_af2_sequences_for_esmif.py 提取，并完成补充 AF2 预测。
    本脚本用于：
        1. 把补充 WT 序列登记追加进旧的 af2_unique_wt_sequences_manifest.csv
        2. 扫描补充 AF2 输出目录，提取 rank1 PDB / JSON / pLDDT / 残基数
        3. 把补充预测结果追加进旧的 af2_prediction_summary.csv
        4. 自动备份旧表，避免误覆盖

这个脚本读取什么数据：
    1. data_processed/esm_if_af2_inputs/af2_missing_wt_sequences_manifest.csv
       - 来自 16b_extract_missing_af2_sequences_for_esmif.py
       - 包含补充预测的唯一 WT 序列及 af2_seq_id

    2. raw_data/af2_predictions/esmif_missing/
       - ColabFold 对补充 WT 序列的输出目录
       - 每条序列应包含 rank1 PDB 和 score JSON

    3. data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv
       - 旧的 AF2 唯一 WT 序列清单，将被追加更新

    4. data_processed/sa_prot_structures/af2_prediction_summary.csv
       - 旧的 AF2 预测结果汇总表，将被追加更新

这个脚本输出什么：
    1. 更新后的：
       data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv

    2. 更新后的：
       data_processed/sa_prot_structures/af2_prediction_summary.csv

    3. 自动备份：
       data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv.bak_时间戳
       data_processed/sa_prot_structures/af2_prediction_summary.csv.bak_时间戳

    4. 补充合并日志：
       data_processed/esm_if_af2_inputs/af2_missing_merge_report.csv

论文写作时可以怎么描述：
    对于既有 AF2 清单中缺失的 WT 序列，本文进一步提取唯一 WT 序列并补充进行 AF2 预测。
    补充预测完成后，将新生成的 rank1 AF2 结构及其 pLDDT、残基数等质量信息合并回统一
    AF2 结构清单中，供 ESM-IF 主分析使用。

使用示例：
    python scripts/structure/16c_merge_missing_af2_into_saprot_manifest.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym

后续步骤：
    合并完成后，重新运行：
    python scripts/structure/16_prepare_esmif_af2_inputs.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym \
        --min-mean-plddt 70
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def relpath_or_abs(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def bool_like(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def find_rank1_files(output_dir: Path, af2_seq_id: str) -> Tuple[Path | None, Path | None]:
    """
    根据 af2_seq_id 在 ColabFold 输出目录中寻找 rank1 PDB 和 score JSON。
    兼容常见命名：
        <id>_unrelaxed_rank_001_*.pdb
        <id>_relaxed_rank_001_*.pdb
        <id>_scores_rank_001_*.json
    """
    pdb_patterns = [
        f"{af2_seq_id}*unrelaxed_rank_001*.pdb",
        f"{af2_seq_id}*relaxed_rank_001*.pdb",
        f"{af2_seq_id}*rank_001*.pdb",
        f"{af2_seq_id}*.pdb",
    ]
    json_patterns = [
        f"{af2_seq_id}*scores_rank_001*.json",
        f"{af2_seq_id}*rank_001*.json",
        f"{af2_seq_id}*.json",
    ]

    pdb_path = None
    json_path = None

    for pat in pdb_patterns:
        hits = sorted(output_dir.glob(pat))
        if hits:
            pdb_path = hits[0]
            break

    for pat in json_patterns:
        hits = sorted(output_dir.glob(pat))
        # 排除可能的 settings/log json；优先 score
        hits = [h for h in hits if "scores" in h.name or "rank_001" in h.name]
        if hits:
            json_path = hits[0]
            break

    return pdb_path, json_path


def parse_pdb_residues_and_plddt(pdb_path: Path) -> Tuple[int, float | None]:
    """
    从 AF2 PDB 中估算残基数和平均 pLDDT。
    ColabFold/AF2 通常将 pLDDT 写在 B-factor 列。
    优先使用 CA 原子统计残基和 pLDDT；若没有 CA，则退回到唯一残基统计。
    """
    if pdb_path is None or not pdb_path.exists():
        return 0, None

    ca_bfactors: List[float] = []
    residues = set()

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()

            residues.add((chain_id, resseq, icode, resname))

            if atom_name == "CA":
                try:
                    b = float(line[60:66].strip())
                    ca_bfactors.append(b)
                except Exception:
                    pass

    if ca_bfactors:
        return len(ca_bfactors), sum(ca_bfactors) / len(ca_bfactors)

    return len(residues), None


def parse_score_json(json_path: Path | None) -> Tuple[float | None, int, float | None, float | None, float | None]:
    """
    读取 ColabFold score JSON，返回：
        json_mean_plddt, json_n_plddt, ptm, iptm, ranking_confidence
    """
    if json_path is None or not json_path.exists():
        return None, 0, None, None, None

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    plddt = data.get("plddt", None)
    json_mean_plddt = None
    json_n_plddt = 0

    if isinstance(plddt, list) and len(plddt) > 0:
        vals = [float(x) for x in plddt]
        json_mean_plddt = sum(vals) / len(vals)
        json_n_plddt = len(vals)

    ptm = data.get("ptm", None)
    iptm = data.get("iptm", None)
    ranking_confidence = data.get("ranking_confidence", None)

    def to_float_or_none(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    return (
        json_mean_plddt,
        json_n_plddt,
        to_float_or_none(ptm),
        to_float_or_none(iptm),
        to_float_or_none(ranking_confidence),
    )


def backup_file(path: Path, timestamp: str) -> Path:
    bak = Path(str(path) + f".bak_{timestamp}")
    shutil.copy2(path, bak)
    return bak


def align_row_to_columns(row: Dict, columns: List[str]) -> Dict:
    return {col: row.get(col, "") for col in columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge supplemental ESM-IF AF2 predictions into SaProt AF2 manifests.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--missing-manifest-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/af2_missing_wt_sequences_manifest.csv",
        help="16b 输出的补充 WT 序列 manifest。",
    )
    parser.add_argument(
        "--af2-output-dir",
        type=str,
        default="raw_data/af2_predictions/esmif_missing",
        help="补充 AF2 / ColabFold 输出目录。",
    )
    parser.add_argument(
        "--old-unique-manifest-csv",
        type=str,
        default="data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv",
        help="旧的 SaProt AF2 唯一 WT 清单，将被追加更新。",
    )
    parser.add_argument(
        "--old-prediction-summary-csv",
        type=str,
        default="data_processed/sa_prot_structures/af2_prediction_summary.csv",
        help="旧的 SaProt AF2 预测结果汇总，将被追加更新。",
    )
    parser.add_argument(
        "--merge-report-csv",
        type=str,
        default="data_processed/esm_if_af2_inputs/af2_missing_merge_report.csv",
        help="补充合并报告输出路径。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份旧表。默认会自动备份。",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    missing_manifest_csv = (project_root / args.missing_manifest_csv).resolve()
    af2_output_dir = (project_root / args.af2_output_dir).resolve()
    old_unique_manifest_csv = (project_root / args.old_unique_manifest_csv).resolve()
    old_prediction_summary_csv = (project_root / args.old_prediction_summary_csv).resolve()
    merge_report_csv = (project_root / args.merge_report_csv).resolve()
    merge_report_csv.parent.mkdir(parents=True, exist_ok=True)

    if not missing_manifest_csv.exists():
        raise FileNotFoundError(f"Missing WT manifest not found: {missing_manifest_csv}")
    if not af2_output_dir.exists():
        raise FileNotFoundError(f"AF2 output dir not found: {af2_output_dir}")
    if not old_unique_manifest_csv.exists():
        raise FileNotFoundError(f"Old AF2 unique manifest not found: {old_unique_manifest_csv}")
    if not old_prediction_summary_csv.exists():
        raise FileNotFoundError(f"Old AF2 prediction summary not found: {old_prediction_summary_csv}")

    missing = pd.read_csv(missing_manifest_csv)
    old_unique = pd.read_csv(old_unique_manifest_csv)
    old_summary = pd.read_csv(old_prediction_summary_csv)

    required_missing_cols = {"af2_seq_id", "seq_md5", "wt_length", "selected_wt_sequence"}
    missing_cols = required_missing_cols - set(missing.columns)
    if missing_cols:
        raise ValueError(f"Missing WT manifest missing required columns: {sorted(missing_cols)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_backup:
        bak1 = backup_file(old_unique_manifest_csv, timestamp)
        bak2 = backup_file(old_prediction_summary_csv, timestamp)
        print(f"[OK] Backup created: {bak1}")
        print(f"[OK] Backup created: {bak2}")

    old_unique_ids = set(old_unique["af2_seq_id"].astype(str)) if "af2_seq_id" in old_unique.columns else set()
    old_unique_md5 = set(old_unique["seq_md5"].astype(str)) if "seq_md5" in old_unique.columns else set()
    old_summary_ids = set(old_summary["af2_seq_id"].astype(str)) if "af2_seq_id" in old_summary.columns else set()

    unique_append_rows = []
    summary_append_rows = []
    report_rows = []

    unique_columns = list(old_unique.columns)
    summary_columns = list(old_summary.columns)

    for _, row in missing.iterrows():
        af2_seq_id = str(row.get("af2_seq_id", "")).strip()
        seq_md5 = str(row.get("seq_md5", "")).strip()
        wt_seq = str(row.get("selected_wt_sequence", "")).strip()
        wt_length = int(row.get("wt_length", len(wt_seq))) if str(row.get("wt_length", "")).strip() else len(wt_seq)

        if not af2_seq_id:
            report_rows.append({
                "af2_seq_id": af2_seq_id,
                "seq_md5": seq_md5,
                "status": "skipped",
                "reason": "missing_af2_seq_id",
            })
            continue

        pdb_path, json_path = find_rank1_files(af2_output_dir, af2_seq_id)
        n_residues, pdb_mean_plddt = parse_pdb_residues_and_plddt(pdb_path) if pdb_path else (0, None)
        json_mean_plddt, json_n_plddt, ptm, iptm, ranking_confidence = parse_score_json(json_path)

        has_pdb = pdb_path is not None and pdb_path.exists()
        has_json = json_path is not None and json_path.exists()

        if not has_pdb:
            report_rows.append({
                "af2_seq_id": af2_seq_id,
                "seq_md5": seq_md5,
                "status": "skipped",
                "reason": "rank1_pdb_not_found",
            })
            continue

        # 1. 追加 unique manifest
        if af2_seq_id not in old_unique_ids and seq_md5 not in old_unique_md5:
            unique_row = {
                "seq_md5": seq_md5,
                "wt_length": wt_length,
                "selected_wt_sequence": wt_seq,
                "n_assays": row.get("n_assays", ""),
                "assays": row.get("assays", ""),
                "pdb_ids": row.get("pdb_ids", ""),
                "pdb_chain_ids": row.get("pdb_chain_ids", ""),
                "missing_dms_fasta_positions_union": "",
                "missing_dms_labels_union": "",
                "af2_reuse_note": "supplemental AF2 prediction for ESM-IF missing WT sequence",
                "af2_seq_id": af2_seq_id,
                "sequence_length": wt_length,
                "has_invalid_aa": False,
                "invalid_aa": "",
                "recommended_colabfold_input": row.get("recommended_colabfold_input", ""),
                "expected_output_dir": row.get("expected_output_dir", args.af2_output_dir),
            }
            unique_append_rows.append(align_row_to_columns(unique_row, unique_columns))
            old_unique_ids.add(af2_seq_id)
            old_unique_md5.add(seq_md5)
            unique_status = "appended"
        else:
            unique_status = "already_exists"

        # 2. 追加 prediction summary
        if af2_seq_id not in old_summary_ids:
            summary_row = {
                "af2_seq_id": af2_seq_id,
                "seq_md5": seq_md5,
                "sequence_length_manifest": wt_length,
                "rank1_pdb_path": relpath_or_abs(pdb_path, project_root) if pdb_path else "",
                "rank1_score_json_path": relpath_or_abs(json_path, project_root) if json_path else "",
                "n_residues_from_pdb": n_residues,
                "pdb_mean_plddt": pdb_mean_plddt,
                "json_mean_plddt": json_mean_plddt,
                "json_n_plddt": json_n_plddt,
                "ptm": ptm,
                "iptm": iptm,
                "ranking_confidence": ranking_confidence,
                "has_rank1_pdb": bool(has_pdb),
                "has_rank1_json": bool(has_json),
            }
            summary_append_rows.append(align_row_to_columns(summary_row, summary_columns))
            old_summary_ids.add(af2_seq_id)
            summary_status = "appended"
        else:
            summary_status = "already_exists"

        report_rows.append({
            "af2_seq_id": af2_seq_id,
            "seq_md5": seq_md5,
            "status": "ok",
            "unique_manifest": unique_status,
            "prediction_summary": summary_status,
            "rank1_pdb_path": relpath_or_abs(pdb_path, project_root) if pdb_path else "",
            "rank1_score_json_path": relpath_or_abs(json_path, project_root) if json_path else "",
            "wt_length": wt_length,
            "n_residues_from_pdb": n_residues,
            "pdb_mean_plddt": pdb_mean_plddt,
            "json_mean_plddt": json_mean_plddt,
            "json_n_plddt": json_n_plddt,
            "length_matches": int(n_residues == wt_length),
            "has_rank1_json": bool(has_json),
        })

    if unique_append_rows:
        old_unique_updated = pd.concat([old_unique, pd.DataFrame(unique_append_rows)], ignore_index=True)
    else:
        old_unique_updated = old_unique.copy()

    if summary_append_rows:
        old_summary_updated = pd.concat([old_summary, pd.DataFrame(summary_append_rows)], ignore_index=True)
    else:
        old_summary_updated = old_summary.copy()

    old_unique_updated.to_csv(old_unique_manifest_csv, index=False)
    old_summary_updated.to_csv(old_prediction_summary_csv, index=False)

    report = pd.DataFrame(report_rows)
    report.to_csv(merge_report_csv, index=False)

    print(f"[OK] Updated unique manifest: {old_unique_manifest_csv}")
    print(f"[OK] Updated prediction summary: {old_prediction_summary_csv}")
    print(f"[OK] Merge report written to: {merge_report_csv}")
    print(f"[INFO] Missing WT entries considered: {len(missing)}")
    print(f"[INFO] Unique manifest rows appended: {len(unique_append_rows)}")
    print(f"[INFO] Prediction summary rows appended: {len(summary_append_rows)}")
    if not report.empty:
        print("[INFO] Merge status counts:")
        print(report["status"].value_counts(dropna=False).to_string())
        if "length_matches" in report.columns:
            print("[INFO] Length match counts:")
            print(report["length_matches"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

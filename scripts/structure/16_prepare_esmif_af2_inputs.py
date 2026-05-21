#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称：
    16_prepare_esmif_af2_inputs.py

这个脚本做什么：
    为 ESM-IF 的 AF2 主线构建输入清单（manifest）。
    该主线统一使用 AlphaFold2 预测的全长单链结构作为 ESM-IF backbone 输入，
    并只保留 AF2 平均 pLDDT 达到阈值的样本。

为什么需要这个脚本：
    ESM-IF 用于突变打分时，本质是在计算 P(sequence | structure)，
    因此输入序列必须与输入结构逐残基对齐。
    实验结构常见问题是只覆盖局部片段、链名体系不一致或含有缺失残基；
    AF2 全长预测结构通常与 WT 全长序列一一对应，更适合作为统一输入主线。

这个脚本读取什么数据：
    1. data_processed/esm_inputs/*.csv
       - 每个文件通常对应一个 DMS assay / chain
       - 包含 DMS_name、pdb_id、pdb_chain_id、wt_sequence、mutant_sequence 等

    2. data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv
       - 前期为 SaProt 准备 AF2 结构时生成的唯一 WT 序列清单
       - 主要字段：seq_md5、selected_wt_sequence、af2_seq_id、sequence_length、assays 等

    3. data_processed/sa_prot_structures/af2_prediction_summary.csv
       - AF2 / ColabFold 预测结果汇总
       - 主要字段：af2_seq_id、rank1_pdb_path、n_residues_from_pdb、
         pdb_mean_plddt、json_mean_plddt、has_rank1_pdb 等

    4. 可选：data_processed/dms_reference_with_pdb.csv
       - 用于标记某个 assay 是否有实验来源结构
       - 该信息只作为 metadata 标记，不参与 AF2 主线筛选

这个脚本输出什么：
    1. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest.csv
       - 每一行对应一个 assay/chain
       - 包括 AF2 结构路径、平均 pLDDT、长度匹配状态、是否可运行、实验结构标记等

    2. data_processed/esm_if_af2_inputs/esm_if_af2_input_manifest_excluded.csv
       - 被排除的条目及原因

论文写作时可以怎么描述：
    本研究将 ESM-IF 主分析的结构输入统一设定为 AF2 预测的全长单链结构。
    对每个 assay/chain，首先从 ESM 输入表中提取 WT 全长序列，并与前期 AF2
    预测结构清单进行匹配。仅当存在 rank1 AF2 结构、结构残基数与 WT 序列长度一致，
    且平均 pLDDT 不低于设定阈值时，该 assay/chain 才进入 ESM-IF 主分析。
    对于同时具有实验结构的样本，仅在 manifest 中进行标记，后续用于比较不同结构来源
    对 ESM-IF 打分结果的影响。

使用示例：
    python scripts/structure/16_prepare_esmif_af2_inputs.py \
        --project-root /public/home/huangwenle/projects/abagym_esm/abagym \
        --min-mean-plddt 70
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def clean_sequence(seq: str) -> str:
    seq = "" if pd.isna(seq) else str(seq)
    return re.sub(r"\s+", "", seq).upper()


def md5_12(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()[:12]


def pick_first_unique_value(series: pd.Series) -> Tuple[str, int]:
    vals = series.dropna().astype(str).map(clean_sequence)
    vals = vals[vals != ""]
    uniq = vals.unique().tolist()
    return (uniq[0] if uniq else "", len(uniq))


def bool_like(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def first_nonempty(row: pd.Series, candidates: List[str]) -> str:
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return ""


def resolve_path(project_root: Path, path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((project_root / p).resolve())


def load_exp_structure_metadata(exp_ref_csv: Optional[Path]) -> Dict[str, dict]:
    if exp_ref_csv is None or not exp_ref_csv.exists():
        return {}

    df = pd.read_csv(exp_ref_csv)
    if "DMS_name" not in df.columns:
        return {}

    metadata: Dict[str, dict] = {}
    for _, row in df.iterrows():
        dms_name = str(row.get("DMS_name", "")).strip()
        if not dms_name:
            continue

        pdb_id = first_nonempty(row, ["pdb_id", "PDB", "pdb", "structure_id", "pdb_accession"])
        chains = first_nonempty(row, ["chains", "chains_raw", "chain", "pdb_chain_id", "pdb_chain_ids"])
        structure_file = first_nonempty(row, ["structure_file", "pdb_file", "cif_file", "structure_path"])

        has_exp = bool(pdb_id or structure_file)

        metadata[dms_name] = {
            "has_exp_structure": int(has_exp),
            "exp_pdb_id": pdb_id,
            "exp_chains": chains,
            "exp_structure_file": structure_file,
            "exp_structure_note": "found_in_dms_reference_with_pdb" if has_exp else "",
        }

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AF2-based ESM-IF input manifest.")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录，默认当前目录。")
    parser.add_argument("--esm-input-dir", type=str, default="data_processed/esm_inputs", help="ESM 输入 csv 目录。")
    parser.add_argument("--af2-unique-manifest-csv", type=str, default="data_processed/sa_prot_structures/af2_unique_wt_sequences_manifest.csv", help="AF2 唯一 WT 序列清单。")
    parser.add_argument("--af2-prediction-summary-csv", type=str, default="data_processed/sa_prot_structures/af2_prediction_summary.csv", help="AF2 预测结果汇总表。")
    parser.add_argument("--exp-reference-csv", type=str, default="data_processed/dms_reference_with_pdb.csv", help="实验结构参考表；若不存在则跳过。")
    parser.add_argument("--out-dir", type=str, default="data_processed/esm_if_af2_inputs", help="输出目录。")
    parser.add_argument("--min-mean-plddt", type=float, default=70.0, help="AF2 平均 pLDDT 纳入阈值，默认 70。")
    parser.add_argument("--default-chain-id", type=str, default="A", help="AF2 单链结构默认链 ID，通常为 A。")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    esm_input_dir = (project_root / args.esm_input_dir).resolve()
    af2_unique_manifest_csv = (project_root / args.af2_unique_manifest_csv).resolve()
    af2_prediction_summary_csv = (project_root / args.af2_prediction_summary_csv).resolve()
    exp_reference_csv = (project_root / args.exp_reference_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "esm_if_af2_input_manifest.csv"
    excluded_out_csv = out_dir / "esm_if_af2_input_manifest_excluded.csv"

    if not esm_input_dir.exists():
        raise FileNotFoundError(f"ESM input dir not found: {esm_input_dir}")
    if not af2_unique_manifest_csv.exists():
        raise FileNotFoundError(f"AF2 unique manifest not found: {af2_unique_manifest_csv}")
    if not af2_prediction_summary_csv.exists():
        raise FileNotFoundError(f"AF2 prediction summary not found: {af2_prediction_summary_csv}")

    af2_unique = pd.read_csv(af2_unique_manifest_csv)
    af2_pred = pd.read_csv(af2_prediction_summary_csv)
    exp_meta = load_exp_structure_metadata(exp_reference_csv)

    required_unique_cols = {"seq_md5", "selected_wt_sequence", "af2_seq_id"}
    missing_unique = required_unique_cols - set(af2_unique.columns)
    if missing_unique:
        raise ValueError(f"AF2 unique manifest missing columns: {sorted(missing_unique)}")

    required_pred_cols = {"af2_seq_id", "rank1_pdb_path", "n_residues_from_pdb", "has_rank1_pdb"}
    missing_pred = required_pred_cols - set(af2_pred.columns)
    if missing_pred:
        raise ValueError(f"AF2 prediction summary missing columns: {sorted(missing_pred)}")

    af2_unique_by_md5: Dict[str, dict] = {}
    af2_unique_by_seq: Dict[str, dict] = {}
    for _, row in af2_unique.iterrows():
        seq = clean_sequence(row.get("selected_wt_sequence", ""))
        seq_md5 = str(row.get("seq_md5", "")).strip()
        if seq_md5:
            af2_unique_by_md5[seq_md5] = row.to_dict()
        if seq:
            af2_unique_by_seq[seq] = row.to_dict()

    af2_pred_by_id: Dict[str, dict] = {}
    for _, row in af2_pred.iterrows():
        af2_seq_id = str(row.get("af2_seq_id", "")).strip()
        if af2_seq_id:
            af2_pred_by_id[af2_seq_id] = row.to_dict()

    rows: List[dict] = []
    esm_files = sorted(esm_input_dir.glob("*.csv"))
    if not esm_files:
        raise FileNotFoundError(f"No csv files found in {esm_input_dir}")

    for esm_path in esm_files:
        df = pd.read_csv(esm_path)

        required_esm_cols = {"DMS_name", "pdb_id", "pdb_chain_id", "wt_sequence", "mutant_sequence"}
        missing_esm = required_esm_cols - set(df.columns)

        dms_name = ""
        pdb_id = ""
        pdb_chain_id = ""
        wt_sequence = ""
        wt_unique_count = 0

        can_run = True
        exclude_reasons: List[str] = []

        if missing_esm:
            can_run = False
            exclude_reasons.append(f"missing_columns:{','.join(sorted(missing_esm))}")
        else:
            dms_vals = df["DMS_name"].dropna().astype(str).str.strip()
            pdb_vals = df["pdb_id"].dropna().astype(str).str.strip()
            chain_vals = df["pdb_chain_id"].dropna().astype(str).str.strip()

            dms_name = dms_vals.iloc[0] if not dms_vals.empty else ""
            pdb_id = pdb_vals.iloc[0] if not pdb_vals.empty else ""
            pdb_chain_id = chain_vals.iloc[0] if not chain_vals.empty else ""
            wt_sequence, wt_unique_count = pick_first_unique_value(df["wt_sequence"])

            if not dms_name:
                can_run = False
                exclude_reasons.append("missing_DMS_name")
            if not pdb_id:
                can_run = False
                exclude_reasons.append("missing_pdb_id")
            if not pdb_chain_id:
                can_run = False
                exclude_reasons.append("missing_pdb_chain_id")
            if not wt_sequence:
                can_run = False
                exclude_reasons.append("missing_wt_sequence")
            if wt_unique_count != 1:
                can_run = False
                exclude_reasons.append(f"wt_sequence_not_unique:{wt_unique_count}")

        wt_len = len(wt_sequence) if wt_sequence else 0
        seq_md5 = md5_12(wt_sequence) if wt_sequence else ""

        af2_seq_id = ""
        af2_structure_path = ""
        af2_score_json_path = ""
        has_rank1_pdb = False
        has_rank1_json = False
        n_residues_from_pdb = pd.NA
        pdb_mean_plddt = pd.NA
        json_mean_plddt = pd.NA
        af2_mean_plddt = pd.NA
        ptm = pd.NA
        iptm = pd.NA
        ranking_confidence = pd.NA
        af2_length_matches_wt = 0
        af2_plddt_pass = 0

        af2_unique_row = None
        if wt_sequence:
            af2_unique_row = af2_unique_by_md5.get(seq_md5)
            if af2_unique_row is None:
                af2_unique_row = af2_unique_by_seq.get(wt_sequence)

        if af2_unique_row is None:
            can_run = False
            exclude_reasons.append("af2_unique_manifest_not_found")
        else:
            af2_seq_id = str(af2_unique_row.get("af2_seq_id", "")).strip()
            if not af2_seq_id:
                can_run = False
                exclude_reasons.append("missing_af2_seq_id")

            pred_row = af2_pred_by_id.get(af2_seq_id)
            if pred_row is None:
                can_run = False
                exclude_reasons.append("af2_prediction_summary_not_found")
            else:
                af2_structure_path = resolve_path(project_root, str(pred_row.get("rank1_pdb_path", "")).strip())
                af2_score_json_path = resolve_path(project_root, str(pred_row.get("rank1_score_json_path", "")).strip())
                has_rank1_pdb = bool_like(pred_row.get("has_rank1_pdb", False))
                has_rank1_json = bool_like(pred_row.get("has_rank1_json", False))
                n_residues_from_pdb = pred_row.get("n_residues_from_pdb", pd.NA)
                pdb_mean_plddt = pred_row.get("pdb_mean_plddt", pd.NA)
                json_mean_plddt = pred_row.get("json_mean_plddt", pd.NA)
                ptm = pred_row.get("ptm", pd.NA)
                iptm = pred_row.get("iptm", pd.NA)
                ranking_confidence = pred_row.get("ranking_confidence", pd.NA)

                try:
                    if pd.notna(json_mean_plddt) and str(json_mean_plddt).strip() != "":
                        af2_mean_plddt = float(json_mean_plddt)
                    elif pd.notna(pdb_mean_plddt) and str(pdb_mean_plddt).strip() != "":
                        af2_mean_plddt = float(pdb_mean_plddt)
                except Exception:
                    af2_mean_plddt = pd.NA

                if not has_rank1_pdb:
                    can_run = False
                    exclude_reasons.append("has_rank1_pdb_false")
                if not af2_structure_path or not Path(af2_structure_path).exists():
                    can_run = False
                    exclude_reasons.append("af2_structure_path_not_found")

                try:
                    n_res = int(n_residues_from_pdb)
                    af2_length_matches_wt = int(n_res == wt_len and wt_len > 0)
                    if n_res != wt_len:
                        can_run = False
                        exclude_reasons.append(f"af2_length_mismatch:{n_res}!={wt_len}")
                except Exception:
                    can_run = False
                    exclude_reasons.append("invalid_n_residues_from_pdb")

                if pd.isna(af2_mean_plddt):
                    can_run = False
                    exclude_reasons.append("missing_af2_mean_plddt")
                else:
                    af2_plddt_pass = int(float(af2_mean_plddt) >= args.min_mean_plddt)
                    if not af2_plddt_pass:
                        can_run = False
                        exclude_reasons.append(f"low_af2_mean_plddt:{float(af2_mean_plddt):.3f}<{args.min_mean_plddt:.3f}")

        meta = exp_meta.get(dms_name, {})
        rows.append({
            "input_filename": esm_path.name,
            "DMS_name": dms_name,
            "pdb_id": pdb_id,
            "pdb_chain_id": pdb_chain_id,
            "wt_sequence": wt_sequence,
            "sequence_length": wt_len,
            "wt_sequence_unique_count": wt_unique_count,
            "n_variants": len(df) if not missing_esm else 0,
            "seq_md5": seq_md5,
            "af2_seq_id": af2_seq_id,
            "af2_structure_path": af2_structure_path,
            "af2_score_json_path": af2_score_json_path,
            "has_rank1_pdb": int(has_rank1_pdb),
            "has_rank1_json": int(has_rank1_json),
            "n_residues_from_pdb": n_residues_from_pdb,
            "af2_length_matches_wt": af2_length_matches_wt,
            "pdb_mean_plddt": pdb_mean_plddt,
            "json_mean_plddt": json_mean_plddt,
            "af2_mean_plddt": af2_mean_plddt,
            "af2_plddt_pass": af2_plddt_pass,
            "ptm": ptm,
            "iptm": iptm,
            "ranking_confidence": ranking_confidence,
            "esm_if_chain_id": args.default_chain_id,
            "can_run_esm_if": int(can_run),
            "exclude_reason": ";".join(exclude_reasons),
            "has_exp_structure": int(meta.get("has_exp_structure", 0)),
            "exp_pdb_id": meta.get("exp_pdb_id", ""),
            "exp_chains": meta.get("exp_chains", ""),
            "exp_structure_file": meta.get("exp_structure_file", ""),
            "exp_structure_note": meta.get("exp_structure_note", ""),
        })

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["can_run_esm_if", "DMS_name", "pdb_chain_id"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

    excluded = manifest[manifest["can_run_esm_if"] == 0].copy()
    manifest.to_csv(out_csv, index=False)
    excluded.to_csv(excluded_out_csv, index=False)

    print(f"[OK] AF2 ESM-IF manifest written to: {out_csv}")
    print(f"[OK] Excluded manifest written to: {excluded_out_csv}")
    print(f"[INFO] Total rows: {len(manifest)}")
    print(f"[INFO] Runnable rows: {(manifest['can_run_esm_if'] == 1).sum()}")
    print(f"[INFO] Excluded rows: {(manifest['can_run_esm_if'] == 0).sum()}")
    print(f"[INFO] pLDDT threshold: {args.min_mean_plddt}")
    if "has_exp_structure" in manifest.columns:
        print(f"[INFO] Rows with experimental structure metadata: {(manifest['has_exp_structure'] == 1).sum()}")


if __name__ == "__main__":
    main()

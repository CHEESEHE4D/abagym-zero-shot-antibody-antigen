from pathlib import Path
import pandas as pd
import re

"""
02_build_reference_skeleton.py

作用：
1. 读取 data_processed/dms_split_summary.csv
2. 将 chains_raw_values 解析为 AbAgym 内部链标签
3. 生成 data_processed/dms_reference.csv 的“待匹配链表”

重要说明：
- 这里的 abagym_chain_label 不是 PDB 真实链名
- 它只是 AbAgym 数据内部使用的链标签
- 真正的 PDB 链匹配将在下一步 03_match_abagym_chain_to_pdb.py 中完成
"""


def split_chain_group(chain_text: str):
    """
    Examples:
    'A' -> ['A']
    'ABC' -> ['A', 'B', 'C']
    'AC,BD' -> ['A', 'C', 'B', 'D']
    'ACE,BDF' -> ['A', 'C', 'E', 'B', 'D', 'F']
    """
    if pd.isna(chain_text):
        return []
    s = str(chain_text).strip().upper()
    if not s or s == "NAN":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    in_path = Path("data_processed/dms_split_summary.csv")
    out_path = Path("data_processed/dms_reference.csv")

    print(f"[read] {in_path}")
    df = pd.read_csv(in_path)

    required_cols = ["DMS_name", "pdb_ids", "chains_raw_values"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in split summary: {missing}")

    rows = []

    for _, r in df.iterrows():
        dms_name = str(r["DMS_name"]).strip()
        pdb_ids = str(r["pdb_ids"]).strip()
        chains_raw = str(r["chains_raw_values"]).strip()

        pdb_id = pdb_ids.split(",")[0].strip() if pdb_ids and pdb_ids.lower() != "nan" else ""
        chain_list = split_chain_group(chains_raw)

        if not chain_list:
            rows.append({
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "abagym_chain_label": "",
                "chains_raw": chains_raw,
                "pdb_chain_id": "",
                "wt_sequence": "",
                "source": "",
                "status": "manual_check",
                "notes": "no chain parsed from chains_raw_values",
            })
            continue

        for ch in chain_list:
            rows.append({
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "abagym_chain_label": ch,
                "chains_raw": chains_raw,
                "pdb_chain_id": "",
                "wt_sequence": "",
                "source": "",
                "status": "pending",
                "notes": "",
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["DMS_name", "abagym_chain_label"]).reset_index(drop=True)

    for col in ["pdb_chain_id", "wt_sequence", "source", "status", "notes"]:
        out_df[col] = out_df[col].astype("object")

    out_df.to_csv(out_path, index=False)
    print(f"[saved] reference skeleton -> {out_path}")
    print(f"[info] rows = {len(out_df)}")


if __name__ == "__main__":
    main()
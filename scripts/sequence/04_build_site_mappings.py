from pathlib import Path
import pandas as pd
import re

""""n_unique_sites"
04_build_site_mappings.py

作用：
1. 读取 data_processed/dms_reference.csv
2. 读取每个 DMS 的原始子表 data_processed/dms_subsets/<DMS_name>.csv
3. 对每个 (DMS_name, abagym_chain_label) ，在已知 wt_sequence 的前提下建立：
      site_raw -> seq_index_0based / seq_pos_1based
4. 优先尝试线性映射；失败时尝试顺序映射
5. 输出：
      - data_processed/site_mappings.csv
      - data_processed/site_mapping_summary.csv
      - data_processed/site_mapping_errors.csv
"""

# -----------------------------
# 可调参数
# -----------------------------
LINEAR_ACCEPT_RATIO = 0.80
LINEAR_ACCEPT_COUNT = 8
ORDERED_ACCEPT_RATIO = 0.60
ORDERED_ACCEPT_COUNT = 4


# -----------------------------
# 基础函数
# -----------------------------
def safe_dms_filename(dms_name: str) -> str:
    return str(dms_name).replace("/", "_")


def contains_chain_group(chains_raw_value: str, abagym_chain_label: str) -> bool:
    if pd.isna(chains_raw_value):
        return False
    groups = [x.strip().upper() for x in str(chains_raw_value).split(",") if x.strip()]
    return str(abagym_chain_label).strip().upper() in groups


def parse_site(site_raw: str):
    """
    '321'  -> (321, '')
    '321a' -> (321, 'a')
    """
    s = str(site_raw).strip().lower()
    m = re.fullmatch(r"(\d+)([a-z]?)", s)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def site_sort_key(site_raw: str):
    base, suffix = parse_site(site_raw)
    if base is None:
        return (10**18, "zzz")
    return (base, suffix)


# -----------------------------
# 提取某个链组的唯一 site->WT
# -----------------------------
def extract_unique_site_wt_pairs(sub_df: pd.DataFrame, abagym_chain_label: str):
    """
    从某个 DMS 子表中，抽取属于该 abagym_chain_label 的唯一 (site_raw, WT) 集合
    若同一 site_raw 对应多个 WT，则报冲突
    """
    if "chains_raw" in sub_df.columns:
        mask = sub_df["chains_raw"].apply(lambda x: contains_chain_group(x, abagym_chain_label))
        sub2 = sub_df.loc[mask].copy()
        if len(sub2) == 0:
            sub2 = sub_df.copy()
    else:
        sub2 = sub_df.copy()

    pairs = (
        sub2[["site_raw", "wildtype"]]
        .dropna()
        .assign(
            site_raw=lambda x: x["site_raw"].astype(str).str.strip().str.lower(),
            wildtype=lambda x: x["wildtype"].astype(str).str.strip().str.upper(),
        )
        .drop_duplicates()
        .sort_values(by="site_raw", key=lambda s: s.map(site_sort_key))
        .reset_index(drop=True)
    )

    conflict_counts = pairs.groupby("site_raw")["wildtype"].nunique()
    conflicting_sites = conflict_counts[conflict_counts > 1].index.tolist()
    if conflicting_sites:
        return None, f"conflicting wildtype letters for sites: {conflicting_sites}"

    return list(zip(pairs["site_raw"], pairs["wildtype"])), ""


# -----------------------------
# 线性映射
# -----------------------------
def build_linear_mapping(seq: str, site_wt_pairs):
    """
    尝试找到一个 start_pos，使最多 site 满足：
        seq[base_site - start_pos] == WT
    """
    parsed = []
    for site_raw, wt in site_wt_pairs:
        base, suffix = parse_site(site_raw)
        if base is None:
            continue
        parsed.append((site_raw, base, suffix, wt))

    if not parsed:
        return None, 0, 0.0, "no parsable numeric sites"

    candidate_starts = set()
    for _, base, _, wt in parsed:
        for idx, aa in enumerate(seq):
            if aa == wt:
                candidate_starts.add(base - idx)

    if not candidate_starts:
        return None, 0, 0.0, "no candidate start_pos found"

    best = None
    best_mapping = None

    for start_pos in candidate_starts:
        mapping_rows = []
        matched = 0

        for site_raw, base, suffix, wt in parsed:
            idx = base - start_pos
            if 0 <= idx < len(seq):
                wt_in_seq = seq[idx]
                ok = (wt_in_seq == wt)
                if ok:
                    matched += 1
                    mapping_rows.append({
                        "site_raw": site_raw,
                        "seq_index_0based": idx,
                        "seq_pos_1based": idx + 1,
                        "wt_expected": wt,
                        "wt_in_sequence": wt_in_seq,
                        "mapping_status": "ok",
                        "mapping_method": "linear",
                    })

        ratio = matched / len(parsed) if parsed else 0.0
        score = (matched, ratio)

        if best is None or score > best:
            best = score
            best_mapping = (start_pos, mapping_rows, matched, ratio)

    if best_mapping is None:
        return None, 0, 0.0, "linear scan failed"

    start_pos, mapping_rows, matched, ratio = best_mapping
    return {
        "start_pos": start_pos,
        "rows": mapping_rows,
        "matched": matched,
        "ratio": ratio,
    }, matched, ratio, ""


# -----------------------------
# 顺序映射（非线性）
# -----------------------------
def build_ordered_mapping(seq: str, site_wt_pairs):
    """
    非线性/跳号的兜底：
    - 只要求 site_raw 的顺序和序列位置顺序一致
    - 按 WT 字母在序列中从前到后依次匹配
    - 若最佳方案唯一，则接受

    这不是最终最强算法，但非常适合你现在这个阶段。
    """
    pairs = []
    for site_raw, wt in site_wt_pairs:
        base, suffix = parse_site(site_raw)
        if base is None:
            continue
        pairs.append((site_raw, wt))

    if not pairs:
        return None, 0, 0.0, "no parsable numeric sites"

    pairs = sorted(pairs, key=lambda x: site_sort_key(x[0]))

    all_paths = []

    # 以第一个 WT 在序列中的所有出现位置作为起点
    first_site, first_wt = pairs[0]
    starts = [i for i, aa in enumerate(seq) if aa == first_wt]

    for start_idx in starts:
        path = [{
            "site_raw": first_site,
            "seq_index_0based": start_idx,
            "seq_pos_1based": start_idx + 1,
            "wt_expected": first_wt,
            "wt_in_sequence": first_wt,
            "mapping_status": "ok",
            "mapping_method": "ordered",
        }]
        cur_idx = start_idx

        for site_raw, wt in pairs[1:]:
            found = None
            for j in range(cur_idx + 1, len(seq)):
                if seq[j] == wt:
                    found = j
                    break
            if found is None:
                break
            path.append({
                "site_raw": site_raw,
                "seq_index_0based": found,
                "seq_pos_1based": found + 1,
                "wt_expected": wt,
                "wt_in_sequence": wt,
                "mapping_status": "ok",
                "mapping_method": "ordered",
            })
            cur_idx = found

        all_paths.append(path)

    if not all_paths:
        return None, 0, 0.0, "ordered mapping found no path"

    all_paths = sorted(all_paths, key=lambda x: len(x), reverse=True)
    best = all_paths[0]
    best_len = len(best)
    ratio = best_len / len(pairs) if pairs else 0.0

    # 看最佳是否唯一
    n_best = sum(1 for p in all_paths if len(p) == best_len)
    if n_best > 1:
        return None, best_len, ratio, f"ordered mapping ambiguous: {n_best} equally good paths"

    return {
        "rows": best,
        "matched": best_len,
        "ratio": ratio,
    }, best_len, ratio, ""


# -----------------------------
# 主程序
# -----------------------------
def main():
    ref_path = Path("data_processed/dms_reference.csv")
    subset_dir = Path("data_processed/dms_subsets")
    mapping_path = Path("data_processed/site_mappings.csv")
    summary_path = Path("data_processed/site_mapping_summary.csv")
    error_path = Path("data_processed/site_mapping_errors.csv")

    print(f"[read] {ref_path}")
    ref_df = pd.read_csv(ref_path)

    required_cols = ["DMS_name", "abagym_chain_label", "wt_sequence", "status"]
    missing = [c for c in required_cols if c not in ref_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dms_reference.csv: {missing}")

    all_mapping_rows = []
    summary_rows = []
    error_rows = []

    subset_cache = {}

    for _, row in ref_df.iterrows():
        dms_name = str(row["DMS_name"]).strip()
        abagym_chain_label = str(row["abagym_chain_label"]).strip().upper()
        pdb_chain_id = str(row.get("pdb_chain_id", "")).strip().upper()
        wt_sequence = str(row["wt_sequence"]).strip().upper()
        ref_status = str(row.get("status", "")).strip()

        if not wt_sequence or wt_sequence.lower() == "nan":
            error_rows.append({
                "DMS_name": dms_name,
                "abagym_chain_label": abagym_chain_label,
                "pdb_chain_id": pdb_chain_id,
                "error": "missing wt_sequence",
            })
            continue

        if "ambiguous" in ref_status.lower() or ref_status == "manual_check":
            error_rows.append({
                "DMS_name": dms_name,
                "abagym_chain_label": abagym_chain_label,
                "pdb_chain_id": pdb_chain_id,
                "error": f"reference status not usable: {ref_status}",
            })
            continue

        safe_name = safe_dms_filename(dms_name)
        subset_path = subset_dir / f"{safe_name}.csv"
        if not subset_path.exists():
            error_rows.append({
                "DMS_name": dms_name,
                "abagym_chain_label": abagym_chain_label,
                "pdb_chain_id": pdb_chain_id,
                "error": f"subset not found: {subset_path.name}",
            })
            continue

        if safe_name not in subset_cache:
            subset_cache[safe_name] = pd.read_csv(subset_path)

        sub_df = subset_cache[safe_name]
        site_wt_pairs, pair_err = extract_unique_site_wt_pairs(sub_df, abagym_chain_label)
        if pair_err:
            error_rows.append({
                "DMS_name": dms_name,
                "abagym_chain_label": abagym_chain_label,
                "pdb_chain_id": pdb_chain_id,
                "error": pair_err,
            })
            continue

        # 1) 先尝试线性映射
        linear_result, linear_matched, linear_ratio, linear_err = build_linear_mapping(wt_sequence, site_wt_pairs)

        chosen = None
        chosen_method = None

        if linear_result is not None and linear_matched >= LINEAR_ACCEPT_COUNT and linear_ratio >= LINEAR_ACCEPT_RATIO:
            chosen = linear_result
            chosen_method = "linear"

        else:
            # 2) 线性不够好时尝试顺序映射
            ordered_result, ordered_matched, ordered_ratio, ordered_err = build_ordered_mapping(wt_sequence, site_wt_pairs)

            if ordered_result is not None and ordered_matched >= ORDERED_ACCEPT_COUNT and ordered_ratio >= ORDERED_ACCEPT_RATIO:
                chosen = ordered_result
                chosen_method = "ordered"
            else:
                error_msg = (
                    f"no valid mapping | "
                    f"linear: matched={linear_matched}, ratio={linear_ratio:.3f}, err={linear_err}; "
                    f"ordered: matched={ordered_matched}, ratio={ordered_ratio:.3f}, err={ordered_err}"
                )
                error_rows.append({
                    "DMS_name": dms_name,
                    "abagym_chain_label": abagym_chain_label,
                    "pdb_chain_id": pdb_chain_id,
                    "error": error_msg,
                })
                summary_rows.append({
                    "DMS_name": dms_name,
                    "abagym_chain_label": abagym_chain_label,
                    "pdb_chain_id": pdb_chain_id,
                    "mapping_method": "",
                    "start_pos": "",
                    "n_unique_sites": len(site_wt_pairs),
                    "n_mapped_sites": 0,
                    "mapping_ratio": 0.0,
                    "status": "manual_check",
                    "notes": error_msg,
                })
                print(f"[manual-check] {dms_name} / {abagym_chain_label}: {error_msg}")
                continue

        chosen_rows = chosen["rows"]
        for r in chosen_rows:
            r["DMS_name"] = dms_name
            r["abagym_chain_label"] = abagym_chain_label
            r["pdb_chain_id"] = pdb_chain_id
            r["start_pos"] = chosen.get("start_pos", "")
            all_mapping_rows.append(r)

        summary_rows.append({
            "DMS_name": dms_name,
            "abagym_chain_label": abagym_chain_label,
            "pdb_chain_id": pdb_chain_id,
            "mapping_method": chosen_method,
            "start_pos": chosen.get("start_pos", ""),
            "n_unique_sites": len(site_wt_pairs),
            "n_mapped_sites": len(chosen_rows),
            "mapping_ratio": len(chosen_rows) / len(site_wt_pairs) if site_wt_pairs else 0.0,
            "status": "ok",
            "notes": "",
        })

        print(f"[mapped] {dms_name} / {abagym_chain_label} -> {chosen_method} ({len(chosen_rows)}/{len(site_wt_pairs)})")

    mapping_df = pd.DataFrame(all_mapping_rows)
    summary_df = pd.DataFrame(summary_rows)
    error_df = pd.DataFrame(error_rows)

    mapping_df.to_csv(mapping_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    error_df.to_csv(error_path, index=False)

    print(f"[saved] {mapping_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {error_path}")
    print(f"[done] mapped_rows={len(mapping_df)}, mapped_groups={len(summary_df)}, errors={len(error_df)}")


if __name__ == "__main__":
    main()
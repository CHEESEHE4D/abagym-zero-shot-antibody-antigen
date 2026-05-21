from pathlib import Path
import pandas as pd
import re
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

MIN_MATCH_RATIO = 0.60
MIN_MATCH_COUNT = 8
AMBIGUOUS_DELTA = 0.05
DOWNLOAD_SLEEP_SEC = 0.2


def extract_pdb_code(pdb_id: str) -> str:
    s = str(pdb_id).strip()
    return s.split("_")[-1].strip().lower()


def safe_dms_filename(dms_name: str) -> str:
    return str(dms_name).replace("/", "_")


def parse_site_base(site_raw: str):
    s = str(site_raw).strip().lower()
    m = re.fullmatch(r"(\d+)([a-z]?)", s)
    if not m:
        return None
    return int(m.group(1))


def parse_chain_ids_from_header(header: str):
    h = header.strip()

    m1 = re.search(r"\bChain\s+([A-Za-z0-9])\b", h)
    if m1:
        return [m1.group(1).upper()]

    m2 = re.search(r"\bChains\s+([A-Za-z0-9,\s]+)\b", h)
    if m2:
        text = m2.group(1)
        chains = re.findall(r"[A-Za-z0-9]", text)
        return [c.upper() for c in chains]

    return []


def read_fasta_by_chain(fasta_path: Path):
    chain_to_seq = {}
    header = None
    seq_lines = []

    def flush_record(h, lines):
        if h is None:
            return
        seq = "".join(lines).strip().upper()
        if not seq:
            return
        chains = parse_chain_ids_from_header(h)
        for ch in chains:
            chain_to_seq[ch] = seq

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush_record(header, seq_lines)
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())

    flush_record(header, seq_lines)
    return chain_to_seq


def download_fasta_if_missing(pdb_code: str, fasta_dir: Path):
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = fasta_dir / f"{pdb_code}.fasta"

    if fasta_path.exists() and fasta_path.stat().st_size > 0:
        return fasta_path, "exists"

    url = f"https://www.rcsb.org/fasta/entry/{pdb_code}/download"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", errors="replace")
        if text.strip():
            fasta_path.write_text(text)
            time.sleep(DOWNLOAD_SLEEP_SEC)
            return fasta_path, "downloaded"
        return None, "empty_response"
    except HTTPError as e:
        return None, f"http_{e.code}"
    except URLError as e:
        return None, f"url_error: {e.reason}"
    except Exception as e:
        return None, f"download_error: {e}"


def contains_chain_group(chains_raw_value: str, abagym_chain_label: str) -> bool:
    if pd.isna(chains_raw_value):
        return False
    groups = [x.strip().upper() for x in str(chains_raw_value).split(",") if x.strip()]
    return str(abagym_chain_label).strip().upper() in groups


def extract_site_wt_pairs_for_chain(sub_df: pd.DataFrame, abagym_chain_label: str):
    abagym_chain_label = str(abagym_chain_label).strip().upper()

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
        .sort_values(["site_raw", "wildtype"])
        .reset_index(drop=True)
    )

    conflict_counts = pairs.groupby("site_raw")["wildtype"].nunique()
    conflicting_sites = conflict_counts[conflict_counts > 1].index.tolist()

    if conflicting_sites:
        return None, f"conflicting wildtype letters for sites: {conflicting_sites}"

    return list(zip(pairs["site_raw"], pairs["wildtype"])), ""


def best_linear_match_for_sequence(seq: str, site_wt_pairs):
    parsed = []
    for site_raw, wt in site_wt_pairs:
        base = parse_site_base(site_raw)
        if base is None:
            continue
        parsed.append((site_raw, base, wt))

    if not parsed:
        return {"start_pos": None, "matched": 0, "total": 0, "ratio": 0.0}

    starts = set()
    for _, base, wt in parsed:
        for idx, aa in enumerate(seq):
            if aa == wt:
                starts.add(base - idx)

    if not starts:
        return {"start_pos": None, "matched": 0, "total": len(parsed), "ratio": 0.0}

    best = None
    for start_pos in starts:
        matched = 0
        for _, base, wt in parsed:
            idx = base - start_pos
            if 0 <= idx < len(seq) and seq[idx] == wt:
                matched += 1
        ratio = matched / len(parsed) if parsed else 0.0
        cand = {"start_pos": start_pos, "matched": matched, "total": len(parsed), "ratio": ratio}
        if best is None or (cand["matched"], cand["ratio"]) > (best["matched"], best["ratio"]):
            best = cand

    return best


def score_all_pdb_chains(chain_to_seq: dict, site_wt_pairs):
    results = []
    for pdb_chain_id, seq in chain_to_seq.items():
        score = best_linear_match_for_sequence(seq, site_wt_pairs)
        score["pdb_chain_id"] = pdb_chain_id
        score["seq_len"] = len(seq)
        score["seq"] = seq
        results.append(score)

    results = sorted(results, key=lambda x: (x["matched"], x["ratio"], -x["seq_len"], x["pdb_chain_id"]), reverse=True)
    return results


def find_top_equivalent_candidates(results, delta=AMBIGUOUS_DELTA):
    if not results:
        return []

    best = results[0]
    top = []
    for r in results:
        if (
            r["matched"] == best["matched"]
            and abs(r["ratio"] - best["ratio"]) < delta
        ):
            top.append(r)
    return top


def resolve_equivalent_top_chains(top_candidates):
    """
    若 top 候选链序列完全相同，则自动选一条（字母序最小）。
    若序列不同，则返回 None，保留 manual_check。
    """
    if not top_candidates:
        return None, []

    seq_groups = {}
    for r in top_candidates:
        seq_groups.setdefault(r["seq"], []).append(r["pdb_chain_id"])

    if len(seq_groups) == 1:
        # 所有并列最佳链序列相同，自动定链
        all_chains = sorted([r["pdb_chain_id"] for r in top_candidates])
        selected = all_chains[0]
        return selected, all_chains

    return None, []

def get_matched_unmatched_sites(seq, site_wt_pairs, start_pos):
    """
    seq: selected_chain 序列
    site_wt_pairs: list of (site_raw, WT residue)
    start_pos: 最佳线性匹配起点
    返回：
        matched_sites: list of site_raw matched
        unmatched_sites: list of tuples (site_raw, WT residue, chain residue)
    """
    matched_sites = []
    unmatched_sites = []

    for site_raw, wt in site_wt_pairs:
        site_idx = parse_site_base(site_raw)
        if site_idx is None:
            continue
        idx_in_seq = site_idx - start_pos
        if 0 <= idx_in_seq < len(seq):
            chain_res = seq[idx_in_seq]
            if chain_res == wt:
                matched_sites.append(site_raw)
            else:
                unmatched_sites.append((site_raw, wt, chain_res))
        else:
            unmatched_sites.append((site_raw, wt, None))  # 超出范围
    return matched_sites, unmatched_sites


def main():
    ref_path = Path("data_processed/dms_reference.csv")
    subset_dir = Path("data_processed/dms_subsets")
    fasta_dir = Path("raw_data/pdb_fastas")
    summary_path = Path("data_processed/dms_reference_match_summary.csv")

    print(f"[read] {ref_path}")
    ref_df = pd.read_csv(ref_path)

    required_cols = ["DMS_name", "pdb_id", "abagym_chain_label"]
    missing = [c for c in required_cols if c not in ref_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dms_reference.csv: {missing}")

    for col in ["pdb_chain_id", "wt_sequence", "source", "status", "notes"]:
        if col not in ref_df.columns:
            ref_df[col] = ""
        ref_df[col] = ref_df[col].astype("object")

    unique_pdb_codes = sorted(
        {
            extract_pdb_code(x)
            for x in ref_df["pdb_id"].astype(str).tolist()
            if str(x).strip() and str(x).strip().lower() != "nan"
        }
    )

    print(f"[info] unique pdb codes = {len(unique_pdb_codes)}")
    for pdb_code in unique_pdb_codes:
        fasta_path, status = download_fasta_if_missing(pdb_code, fasta_dir)
        if fasta_path is not None:
            print(f"[fasta] {pdb_code}: {status}")
        else:
            print(f"[fasta-fail] {pdb_code}: {status}")

    fasta_cache = {}
    subset_cache = {}
    summary_rows = []

    for i, row in ref_df.iterrows():
        dms_name = str(row["DMS_name"]).strip()
        pdb_id = str(row["pdb_id"]).strip()
        abagym_chain_label = str(row["abagym_chain_label"]).strip().upper()

        safe_name = safe_dms_filename(dms_name)
        subset_path = subset_dir / f"{safe_name}.csv"
        if not subset_path.exists():
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = f"subset not found: {subset_path.name}"
            continue

        if safe_name not in subset_cache:
            subset_cache[safe_name] = pd.read_csv(subset_path)

        sub_df = subset_cache[safe_name]
        site_wt_pairs, pair_err = extract_site_wt_pairs_for_chain(sub_df, abagym_chain_label)
        if pair_err:
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = pair_err
            summary_rows.append({
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "abagym_chain_label": abagym_chain_label,
                "best_pdb_chain": "",
                "match_method": "",
                "match_score": 0,
                "linear_ratio": 0.0,
                "status": "manual_check",
                "notes": pair_err,
            })
            print(f"[manual-check] {dms_name} / {abagym_chain_label}: {pair_err}")
            continue

        pdb_code = extract_pdb_code(pdb_id)
        fasta_path = fasta_dir / f"{pdb_code}.fasta"
        if not fasta_path.exists() or fasta_path.stat().st_size == 0:
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = f"missing fasta file: {fasta_path.name}"
            continue

        if pdb_code not in fasta_cache:
            fasta_cache[pdb_code] = read_fasta_by_chain(fasta_path)

        chain_to_seq = fasta_cache[pdb_code]
        if not chain_to_seq:
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = "no chains parsed from fasta"
            continue

        results = score_all_pdb_chains(chain_to_seq, site_wt_pairs)
        best = results[0] if results else None
        if best is None:
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = "no candidate chain scored"
            continue

        best_chain = best["pdb_chain_id"]
        best_ratio = best["ratio"]
        best_matched = best["matched"]

        # 先做基础阈值判断
        if best_ratio == 1.0:
            ref_df.at[i, 'matched_sites'] = ''
            ref_df.at[i, 'unmatched_sites'] = ''
        else:
            ref_df.at[i, 'matched_sites'] = ",".join(matched_sites)
            ref_df.at[i, 'unmatched_sites'] = ",".join([f"{s}:{w}:{r}" for s, w, r in unmatched_sites])
        

        if best_ratio < MIN_MATCH_RATIO or best_matched < MIN_MATCH_COUNT:
            notes = f"best={best_chain}, matched={best_matched}/{best['total']}, ratio={best_ratio:.3f}"
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = notes
            summary_rows.append({
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "abagym_chain_label": abagym_chain_label,
                "best_pdb_chain": best_chain,
                "match_method": "linear_scan",
                "match_score": best_matched,
                "linear_ratio": best_ratio,
                "status": "manual_check",
                "notes": notes,
            })
            print(f"[manual-check] {dms_name} / {abagym_chain_label}: {notes}")
            continue

        # 处理“并列最佳链”
        top_candidates = find_top_equivalent_candidates(results, delta=AMBIGUOUS_DELTA)
        selected_chain, equivalent_chains = resolve_equivalent_top_chains(top_candidates)

        if selected_chain is None and len(top_candidates) > 1:
            cand_text = ",".join([f"{r['pdb_chain_id']}:{r['ratio']:.3f}" for r in top_candidates])
            notes = f"ambiguous top chains with different sequences -> {cand_text}"
            ref_df.at[i, "status"] = "manual_check"
            ref_df.at[i, "notes"] = notes
            summary_rows.append({
                "DMS_name": dms_name,
                "pdb_id": pdb_id,
                "abagym_chain_label": abagym_chain_label,
                "best_pdb_chain": best_chain,
                "match_method": "linear_scan",
                "match_score": best_matched,
                "linear_ratio": best_ratio,
                "status": "manual_check",
                "notes": notes,
            })
            print(f"[manual-check] {dms_name} / {abagym_chain_label}: {notes}")
            continue

        if selected_chain is None:
            selected_chain = best_chain
        
        seq = chain_to_seq[selected_chain]
        matched_sites, unmatched_sites = get_matched_unmatched_sites(seq, site_wt_pairs, best['start_pos'])
        ref_df.at[i, "pdb_chain_id"] = selected_chain
        ref_df.at[i, "best_wt_sequence"] = seq
        ref_df.at[i, "source"] = "matched_by_wt_linear"
        ref_df.at[i, "status"] = "ok"
        ref_df.at[i, 'matched_sites'] = ",".join(matched_sites)
        ref_df.at[i, 'unmatched_sites'] = ",".join([f"{s}:{w}:{r}" for s, w, r in unmatched_sites])

        note = f"matched={best_matched}/{best['total']}; ratio={best_ratio:.3f}; start_pos={best['start_pos']}"
        if equivalent_chains:
            note += f"; equivalent_chains={','.join(equivalent_chains)}; selected={selected_chain}"
        ref_df.at[i, "notes"] = note

        summary_rows.append({
            "DMS_name": dms_name,
            "pdb_id": pdb_id,
            "abagym_chain_label": abagym_chain_label,
            "best_pdb_chain": selected_chain,
            "match_method": "linear_scan",
            "match_score": best_matched,
            "linear_ratio": best_ratio,
            "status": "ok",
            "notes": note,
        })
        print(f"[matched] {dms_name} / {abagym_chain_label} -> PDB chain {selected_chain} ({best_matched}/{best['total']}, ratio={best_ratio:.3f})")

    ref_df.to_csv(ref_path, index=False)
    print(f"[saved] updated reference table -> {ref_path}")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[saved] match summary -> {summary_path}")

    n_ok = sum(1 for x in summary_rows if x["status"] == "ok")
    n_bad = sum(1 for x in summary_rows if x["status"] != "ok")
    print(f"[done] ok={n_ok}, manual_check={n_bad}")


if __name__ == "__main__":
    main()
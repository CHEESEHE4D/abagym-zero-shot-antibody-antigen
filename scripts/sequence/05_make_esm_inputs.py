from pathlib import Path
import pandas as pd


"""
05_make_esm_inputs.py

作用：
1. 读取：
   - data_processed/dms_reference.csv
   - data_processed/site_mappings.csv
   - data_processed/site_mapping_summary.csv
   - data_processed/dms_subsets/*.csv
2. 只保留 site_mapping_summary.csv 中 status == "ok" 的链组
3. 将原始突变翻译成：
   - seq_pos_1based
   - esm_mut
   - mutant_sequence
4. 输出：
   - data_processed/esm_inputs/<DMS_name>__<abagym_chain_label>.csv
   - data_processed/esm_input_summary.csv
"""

def safe_dms_filename(dms_name: str) -> str:
    return str(dms_name).replace("/", "_")


def contains_chain_group(chains_raw_value: str, abagym_chain_label: str) -> bool:
    if pd.isna(chains_raw_value):
        return False
    groups = [x.strip().upper() for x in str(chains_raw_value).split(",") if x.strip()]
    return str(abagym_chain_label).strip().upper() in groups


def build_mutant_sequence(wt_sequence: str, seq_index_0based: int, wt: str, mut: str):
    if not (0 <= seq_index_0based < len(wt_sequence)):
        raise ValueError(f"seq_index out of range: {seq_index_0based}")
    real_wt = wt_sequence[seq_index_0based]
    if real_wt != wt:
        raise ValueError(f"WT mismatch in wt_sequence at idx={seq_index_0based}: expected {wt}, found {real_wt}")
    return wt_sequence[:seq_index_0based] + mut + wt_sequence[seq_index_0based + 1:]


def main():
    ref_path = Path("data_processed/dms_reference.csv")
    mapping_path = Path("data_processed/site_mappings.csv")
    mapping_summary_path = Path("data_processed/site_mapping_summary.csv")
    subset_dir = Path("data_processed/dms_subsets")
    out_dir = Path("data_processed/esm_inputs")
    summary_out = Path("data_processed/esm_input_summary.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[read] {ref_path}")
    ref_df = pd.read_csv(ref_path)

    print(f"[read] {mapping_path}")
    mapping_df = pd.read_csv(mapping_path)

    print(f"[read] {mapping_summary_path}")
    mapping_summary_df = pd.read_csv(mapping_summary_path)

    # 只保留映射成功的链组
    ok_groups = mapping_summary_df[mapping_summary_df["status"] == "ok"].copy()

    if len(ok_groups) == 0:
        raise ValueError("No usable groups found in site_mapping_summary.csv (status == 'ok').")

    # 建 reference lookup
    ref_lookup = {}
    for _, r in ref_df.iterrows():
        key = (str(r["DMS_name"]).strip(), str(r["abagym_chain_label"]).strip().upper())
        wt_seq = str(r.get("wt_sequence", "")).strip().upper()
        if wt_seq and wt_seq.lower() != "nan":
            ref_lookup[key] = {
                "wt_sequence": wt_seq,
                "pdb_chain_id": str(r.get("pdb_chain_id", "")).strip().upper(),
            }

    # 建 mapping lookup: (DMS_name, abagym_chain_label, site_raw) -> mapping row
    map_lookup = {}
    for _, r in mapping_df.iterrows():
        key = (
            str(r["DMS_name"]).strip(),
            str(r["abagym_chain_label"]).strip().upper(),
            str(r["site_raw"]).strip().lower(),
        )
        map_lookup[key] = {
            "seq_index_0based": int(r["seq_index_0based"]),
            "seq_pos_1based": int(r["seq_pos_1based"]),
            "wt_expected": str(r["wt_expected"]).strip().upper(),
            "mapping_method": str(r.get("mapping_method", "")).strip(),
            "pdb_chain_id": str(r.get("pdb_chain_id", "")).strip().upper(),
        }

    summary_rows = []

    for _, g in ok_groups.iterrows():
        dms_name = str(g["DMS_name"]).strip()
        abagym_chain_label = str(g["abagym_chain_label"]).strip().upper()

        safe_name = safe_dms_filename(dms_name)
        subset_path = subset_dir / f"{safe_name}.csv"
        if not subset_path.exists():
            print(f"[skip] subset not found: {subset_path.name}")
            continue

        key_ref = (dms_name, abagym_chain_label)
        if key_ref not in ref_lookup:
            print(f"[skip] no reference wt_sequence: {dms_name} / {abagym_chain_label}")
            continue

        wt_sequence = ref_lookup[key_ref]["wt_sequence"]
        pdb_chain_id = ref_lookup[key_ref]["pdb_chain_id"]

        sub_df = pd.read_csv(subset_path)

        # 只保留这个链组对应的原始记录
        if "chains_raw" in sub_df.columns:
            mask = sub_df["chains_raw"].apply(lambda x: contains_chain_group(x, abagym_chain_label))
            sub_df = sub_df.loc[mask].copy()
            if len(sub_df) == 0:
                print(f"[skip] no rows for chain group: {dms_name} / {abagym_chain_label}")
                continue

        sub_df["site_raw"] = sub_df["site_raw"].astype(str).str.strip().str.lower()
        sub_df["wildtype"] = sub_df["wildtype"].astype(str).str.strip().str.upper()
        sub_df["mutation"] = sub_df["mutation"].astype(str).str.strip().str.upper()

        out_rows = []
        bad_count = 0

        for _, r in sub_df.iterrows():
            site_raw = str(r["site_raw"]).strip().lower()
            wt = str(r["wildtype"]).strip().upper()
            mut = str(r["mutation"]).strip().upper()

            map_key = (dms_name, abagym_chain_label, site_raw)
            if map_key not in map_lookup:
                bad_count += 1
                continue

            m = map_lookup[map_key]
            seq_index_0based = m["seq_index_0based"]
            seq_pos_1based = m["seq_pos_1based"]
            wt_expected = m["wt_expected"]

            # 再核对一下 WT
            if wt != wt_expected:
                bad_count += 1
                continue

            try:
                mutant_sequence = build_mutant_sequence(
                    wt_sequence=wt_sequence,
                    seq_index_0based=seq_index_0based,
                    wt=wt,
                    mut=mut,
                )
            except Exception:
                bad_count += 1
                continue

            esm_mut = f"{wt}{seq_pos_1based}{mut}"

            row_out = r.to_dict()
            row_out["abagym_chain_label"] = abagym_chain_label
            row_out["pdb_chain_id"] = pdb_chain_id
            row_out["seq_index_0based"] = seq_index_0based
            row_out["seq_pos_1based"] = seq_pos_1based
            row_out["esm_mut"] = esm_mut
            row_out["wt_sequence"] = wt_sequence
            row_out["mutant_sequence"] = mutant_sequence
            out_rows.append(row_out)

        out_df = pd.DataFrame(out_rows)
        out_path = out_dir / f"{safe_name}__{abagym_chain_label}.csv"

        if len(out_df) > 0:
            out_df.to_csv(out_path, index=False)
            print(f"[saved] {out_path} ({len(out_df)} rows, {bad_count} skipped)")
        else:
            print(f"[skip] empty output for {dms_name} / {abagym_chain_label}")
            continue

        summary_rows.append({
            "DMS_name": dms_name,
            "abagym_chain_label": abagym_chain_label,
            "pdb_chain_id": pdb_chain_id,
            "n_rows_out": len(out_df),
            "n_rows_skipped": bad_count,
            "output_file": out_path.name,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_out, index=False)
    print(f"[saved] {summary_out}")
    print(f"[done] esm input groups = {len(summary_df)}")


if __name__ == "__main__":
    main()
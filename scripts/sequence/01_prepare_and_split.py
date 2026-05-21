from pathlib import Path
import pandas as pd


def main():
    """
	读入 AbAgym_covid_hiv_flu_interface.csv
	统一列名
	规范 site_raw / wildtype / mutation
	保存总表 abagym_clean.csv
	按 DMS_name 拆分保存到 dms_subsets/
    Step 1:
    1) Read raw AbAgym subset table
    2) Standardize key column names and value formats
    3) Save one cleaned master table
    4) Split into one CSV per DMS_name

    Notes:
    - site_raw is always treated as string, never forced to int
    - This step does NOT infer start_pos, chain mapping, or ESM mutation format
    """

    raw_dir = Path("raw_data")
    out_dir = Path("data_processed")
    subset_dir = out_dir / "dms_subsets"

    out_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    input_csv = raw_dir / "AbAgym_covid_hiv_flu_interface.csv"
    clean_csv = out_dir / "abagym_clean.csv"

    print(f"[read] {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"[info] raw shape = {df.shape}")

    rename_map = {
        "site": "site_raw",
        "chains": "chains_raw",
        "PDB_file": "pdb_id",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["DMS_name", "site_raw", "wildtype", "mutation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["DMS_name"] = df["DMS_name"].astype(str).str.strip()
    df["site_raw"] = df["site_raw"].astype(str).str.strip().str.lower()
    df["wildtype"] = df["wildtype"].astype(str).str.strip().str.upper()
    df["mutation"] = df["mutation"].astype(str).str.strip().str.upper()

    if "chains_raw" in df.columns:
        df["chains_raw"] = df["chains_raw"].astype(str).str.strip()

    if "pdb_id" in df.columns:
        df["pdb_id"] = df["pdb_id"].astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["DMS_name", "site_raw", "wildtype", "mutation"]).copy()
    after = len(df)
    print(f"[info] kept rows after basic dropna: {after} / {before}")

    df.to_csv(clean_csv, index=False)
    print(f"[saved] cleaned table -> {clean_csv}")

    dms_names = sorted(df["DMS_name"].dropna().unique())
    print(f"[info] total DMS datasets = {len(dms_names)}")

    summary_rows = []
    for dms_name, sub_df in df.groupby("DMS_name", sort=True):
        safe_name = str(dms_name).replace("/", "_")
        out_path = subset_dir / f"{safe_name}.csv"
        sub_df = sub_df.copy().reset_index(drop=True)
        sub_df.to_csv(out_path, index=False)

        summary_rows.append({
            "DMS_name": dms_name,
            "n_rows": len(sub_df),
            "n_unique_sites": sub_df["site_raw"].nunique(),
            "pdb_ids": ",".join(sorted(sub_df["pdb_id"].dropna().astype(str).unique())) if "pdb_id" in sub_df.columns else "",
            "chains_raw_values": ",".join(sorted(sub_df["chains_raw"].dropna().astype(str).unique())) if "chains_raw" in sub_df.columns else "",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "dms_split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] split summary -> {summary_path}")
    print(f"[done] wrote {len(summary_rows)} DMS subset files")


if __name__ == "__main__":
    main()
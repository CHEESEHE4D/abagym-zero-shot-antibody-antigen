from pathlib import Path
import pandas as pd
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import time

DOWNLOAD_SLEEP_SEC = 0.2

def extract_pdb_code(pdb_id: str) -> str:
    """从 pdb_id 提取 PDB code"""
    s = str(pdb_id).strip()
    return s.split("_")[-1].strip().upper()

def download_pdb_or_cif(pdb_code: str, pdb_dir: Path):
    """
    尝试下载 PDB，如果失败，再尝试 mmCIF/mmCIF.gz
    返回文件路径和状态
    """
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    # 先尝试 .pdb
    pdb_path = pdb_dir / f"{pdb_code}.pdb"
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return pdb_path, "exists"
    
    url_list = [
        f"https://files.rcsb.org/download/{pdb_code}.pdb",
        f"https://files.rcsb.org/download/{pdb_code}.cif",
        f"https://files.rcsb.org/download/{pdb_code}.cif.gz"
    ]
    
    for url in url_list:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as r:
                data = r.read()
            if data:
                # 根据扩展名保存文件
                file_ext = url.split(".")[-1]
                path = pdb_dir / f"{pdb_code}.{file_ext}"
                path.write_bytes(data)
                time.sleep(DOWNLOAD_SLEEP_SEC)
                return path, f"downloaded ({file_ext})"
        except HTTPError as e:
            if e.code == 404:
                continue
        except URLError as e:
            continue
        except Exception:
            continue
    
    return None, "no experimental structure, may need AF2"

def main():
    ref_path = Path("data_processed/dms_reference.csv")
    pdb_dir = Path("raw_data/pdb_structs")
    output_csv = Path("data_processed/dms_reference_with_pdb.csv")

    df = pd.read_csv(ref_path)

    # 只处理 status=ok
    df_ok = df[df["status"] == "ok"].copy()
    df_ok["pdb_file_status"] = ""

    for i, row in df_ok.iterrows():
        pdb_id = row["pdb_id"]
        pdb_chain = row["pdb_chain_id"]
        pdb_code = extract_pdb_code(pdb_id)

        file_path, status = download_pdb_or_cif(pdb_code, pdb_dir)
        df_ok.at[i, "pdb_file_status"] = status
        print(f"[{status}] {row['DMS_name']} / PDB {pdb_code} chain {pdb_chain}")

    df_ok.to_csv(output_csv, index=False)
    print(f"[saved] updated ok assay reference -> {output_csv}")

if __name__ == "__main__":
    main()
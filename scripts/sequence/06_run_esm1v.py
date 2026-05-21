import argparse
import subprocess
from pathlib import Path
import pandas as pd


"""
06_run_esm1v.py

作用：
1. 读取 data_processed/esm_input_summary.csv
2. 对每个 data_processed/esm_inputs/*.csv 调用官方 ESM-1v predict.py
3. 一次性并列传入 5 个 esm1v 模型
4. 每个输入文件输出 1 个结果 csv（其中包含 5 个模型分数列）
5. 输出到 data_processed/esm_scores/

前提：
- 输入表里必须有：
    wt_sequence
    esm_mut
- esm_mut 已经是线性编号格式，例如 A512C
- 因此这里统一使用 --offset-idx 1
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_py",
        type=str,
        default="/public/home/huangwenle/projects/abagym_esm/esm/examples/variant-prediction/predict.py",
        help="官方 ESM predict.py 的路径",
    )
    parser.add_argument(
        "--model_locations",
        type=str,
        nargs="+",
        required=True,
        help="多个 ESM-1v 模型路径",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="/public/home/huangwenle/projects/abagym_esm/abagym/data_processed/esm_input_summary.csv",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/public/home/huangwenle/projects/abagym_esm/abagym/data_processed/esm_inputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/public/home/huangwenle/projects/abagym_esm/abagym/data_processed/esm_scores",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default="python",
        help="运行 predict.py 用的 python，可填 python 或具体解释器路径",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若输出已存在则跳过",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    predict_py = Path(args.predict_py)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")
    if not predict_py.exists():
        raise FileNotFoundError(f"predict.py not found: {predict_py}")

    print(f"[read] {summary_path}")
    summary_df = pd.read_csv(summary_path)

    required_cols = ["DMS_name", "abagym_chain_label", "output_file"]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in esm_input_summary.csv: {missing}")

    ok = 0
    fail = 0
    skip = 0

    print(f"[info] total input groups = {len(summary_df)}")
    print(f"[info] total models = {len(args.model_locations)}")

    for _, row in summary_df.iterrows():
        dms_name = str(row["DMS_name"]).strip()
        abagym_chain_label = str(row["abagym_chain_label"]).strip().upper()
        input_file = str(row["output_file"]).strip()

        in_path = input_dir / input_file
        if not in_path.exists():
            print(f"[skip] input not found: {in_path}")
            skip += 1
            continue

        df = pd.read_csv(in_path)
        if "wt_sequence" not in df.columns or "esm_mut" not in df.columns:
            print(f"[skip] missing wt_sequence/esm_mut in {in_path.name}")
            skip += 1
            continue

        wt_sequences = df["wt_sequence"].dropna().astype(str).unique()
        if len(wt_sequences) != 1:
            print(f"[fail] {in_path.name}: wt_sequence is not unique")
            fail += 1
            continue

        wt_sequence = wt_sequences[0].strip().upper()

        out_path = output_dir / f"{in_path.stem}_esm1v_1to5.csv"
        log_path = output_dir / f"{in_path.stem}_esm1v_1to5.log"

        if args.skip_if_exists and out_path.exists():
            print(f"[skip-exists] {out_path.name}")
            skip += 1
            continue

        cmd = [
            args.python_bin,
            str(predict_py),
            "--model-location",
            *args.model_locations,
            "--sequence",
            wt_sequence,
            "--dms-input",
            str(in_path),
            "--mutation-col",
            "esm_mut",
            "--dms-output",
            str(out_path),
            "--offset-idx",
            "1",
        ]

        print(f"[run] {dms_name} / {abagym_chain_label} -> {out_path.name}")
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=logf)

        if proc.returncode == 0 and out_path.exists():
            ok += 1
            print(f"[ok] {out_path.name}")
        else:
            fail += 1
            print(f"[fail] {in_path.name} (see {log_path.name})")

    print(f"[done] ok={ok}, fail={fail}, skip={skip}")


if __name__ == "__main__":
    main()
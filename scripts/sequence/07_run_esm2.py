import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from typing import Optional

"""
07_run_esm2.py

作用：
1. 读取 abagym/data_processed/esm_input_summary.csv
2. 对每个 abagym/data_processed/esm_inputs/*.csv 进行 ESM-2 zero-shot 打分
3. 单模型方案：默认跑 facebook/esm2_t33_650M_UR50D
4. 每个输入文件输出 1 个结果 csv 到 abagym/data_processed/esm_scores/
5. 保留原始列，并新增：
   - esm2_t33_650M_score
   - esm2_status
   - esm2_error

输入文件前提：
- 必须有 wt_sequence
- 必须有 esm_mut
- esm_mut 为线性编号格式，例如：
    K417A
    N501Y
    K417A,N501Y
    K417A:N501Y
  （脚本内部用正则提取 [A-Z][0-9]+[A-Z]，因此分隔符不敏感）
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace 模型名，或本地下载好的模型目录路径",
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若输出已存在则跳过",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_esm2_t33_650M.csv",
        help="输出文件后缀",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="esm2_t33_650M_score",
        help="输出分数列名",
    )

    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def parse_esm_mut(esm_mut: str):
    """
    从 esm_mut 中提取一个或多个突变，返回：
    [(wt_aa, pos_1based, mut_aa), ...]
    例如 "K417A,N501Y" -> [("K",417,"A"), ("N",501,"Y")]
    """
    s = str(esm_mut).strip().upper()
    muts = re.findall(r"([A-Z])(\d+)([A-Z])", s)
    if not muts:
        raise ValueError(f"Cannot parse esm_mut: {esm_mut}")
    return [(wt, int(pos), mut) for wt, pos, mut in muts]


def canonicalize_sequence(seq: str) -> str:
    return str(seq).strip().replace(" ", "").replace("\n", "").upper()


def get_model_max_aa_len(model) -> Optional[int]:
    """
    ESM 一般带有 BOS/EOS，因此可用氨基酸长度近似为 max_position_embeddings - 2
    """
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is None:
        return None
    return max_pos - 2


def validate_mutations_against_wt(wt_sequence: str, muts):
    """
    检查：
    1. 位点是否越界
    2. esm_mut 中 wildtype 字母是否与 wt_sequence 一致
    """
    seq_len = len(wt_sequence)
    for wt_aa, pos1, mut_aa in muts:
        if pos1 < 1 or pos1 > seq_len:
            raise ValueError(
                f"Position out of range: {wt_aa}{pos1}{mut_aa}, seq_len={seq_len}"
            )
        real_wt = wt_sequence[pos1 - 1]
        if real_wt != wt_aa:
            raise ValueError(
                f"WT mismatch at {pos1}: esm_mut says {wt_aa}, "
                f"but wt_sequence has {real_wt}"
            )


def score_variant_masked_marginal(
    wt_sequence: str,
    esm_mut: str,
    tokenizer,
    model,
    device: str,
    cached_wt_inputs: dict,
) -> float:
    """
    masked marginal:
    对所有突变位点同时 mask，一次前向，
    然后累加 log p(mut) - log p(wt)
    """
    muts = parse_esm_mut(esm_mut)
    validate_mutations_against_wt(wt_sequence, muts)

    input_ids = cached_wt_inputs["input_ids"].clone()
    attention_mask = cached_wt_inputs["attention_mask"]

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id")

    # ESM tokenizer 编码后通常:
    # [CLS] A A A ... [EOS]
    # 因此序列第 1 位残基对应 token index = 1
    token_positions = []
    for wt_aa, pos1, mut_aa in muts:
        token_idx = pos1
        token_positions.append((wt_aa, token_idx, mut_aa))
        input_ids[token_idx] = mask_token_id

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
        )
        logits = outputs.logits[0]  # [seq_len_with_special, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

    score = 0.0
    for wt_aa, token_idx, mut_aa in token_positions:
        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        if wt_id is None or mut_id is None:
            raise ValueError(f"Unknown token id for wt={wt_aa}, mut={mut_aa}")

        score += (log_probs[token_idx, mut_id] - log_probs[token_idx, wt_id]).item()

    return float(score)


def build_cached_wt_inputs(wt_sequence: str, tokenizer, model):
    wt_inputs = tokenizer(
        wt_sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = wt_inputs["input_ids"][0].cpu()
    attention_mask = wt_inputs["attention_mask"][0].cpu()

    # 长度检查
    max_aa_len = get_model_max_aa_len(model)
    if max_aa_len is not None and len(wt_sequence) > max_aa_len:
        raise ValueError(
            f"WT sequence too long for model: len={len(wt_sequence)} > max_aa_len={max_aa_len}"
        )

    # 一般应满足：编码长度 = 氨基酸长度 + 2（BOS/EOS）
    expected_len = len(wt_sequence) + 2
    if input_ids.shape[0] != expected_len:
        raise ValueError(
            f"Unexpected tokenized length: got {input_ids.shape[0]}, expected {expected_len}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def main():
    args = parse_args()
    device = choose_device(args.device)

    summary_path = Path(args.summary_csv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")

    print(f"[info] device = {device}")
    print(f"[load] tokenizer/model from: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = EsmForMaskedLM.from_pretrained(args.model_name_or_path)
    model.eval()
    model.to(device)

    print(f"[read] {summary_path}")
    summary_df = pd.read_csv(summary_path)

    required_summary_cols = ["DMS_name", "abagym_chain_label", "output_file"]
    missing_summary = [c for c in required_summary_cols if c not in summary_df.columns]
    if missing_summary:
        raise ValueError(
            f"Missing required columns in esm_input_summary.csv: {missing_summary}"
        )

    ok = 0
    fail = 0
    skip = 0

    print(f"[info] total input groups = {len(summary_df)}")

    for _, row in summary_df.iterrows():
        dms_name = str(row["DMS_name"]).strip()
        abagym_chain_label = str(row["abagym_chain_label"]).strip().upper()
        input_file = str(row["output_file"]).strip()

        in_path = input_dir / input_file
        if not in_path.exists():
            print(f"[skip] input not found: {in_path}")
            skip += 1
            continue

        out_path = output_dir / f"{in_path.stem}{args.suffix}"

        if args.skip_if_exists and out_path.exists():
            print(f"[skip-exists] {out_path.name}")
            skip += 1
            continue

        print(f"[run] {dms_name} / {abagym_chain_label} -> {out_path.name}")

        try:
            df = pd.read_csv(in_path)

            required_input_cols = ["wt_sequence", "esm_mut"]
            missing_input = [c for c in required_input_cols if c not in df.columns]
            if missing_input:
                raise ValueError(
                    f"Missing required columns in {in_path.name}: {missing_input}"
                )

            wt_sequences = df["wt_sequence"].dropna().astype(str).map(canonicalize_sequence).unique()
            if len(wt_sequences) != 1:
                raise ValueError(f"{in_path.name}: wt_sequence is not unique")

            wt_sequence = wt_sequences[0]
            cached_wt_inputs = build_cached_wt_inputs(wt_sequence, tokenizer, model)

            score_cache = {}
            scores = []
            statuses = []
            errors = []

            for esm_mut in df["esm_mut"].astype(str):
                esm_mut_key = esm_mut.strip().upper()

                if esm_mut_key in score_cache:
                    score, status, err = score_cache[esm_mut_key]
                else:
                    try:
                        score = score_variant_masked_marginal(
                            wt_sequence=wt_sequence,
                            esm_mut=esm_mut_key,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            cached_wt_inputs=cached_wt_inputs,
                        )
                        status = "ok"
                        err = ""
                    except Exception as e:
                        score = float("nan")
                        status = "fail"
                        err = str(e)

                    score_cache[esm_mut_key] = (score, status, err)

                scores.append(score)
                statuses.append(status)
                errors.append(err)

            df[args.score_col] = scores
            df["esm2_status"] = statuses
            df["esm2_error"] = errors

            df.to_csv(out_path, index=False)

            n_fail_rows = (df["esm2_status"] != "ok").sum()
            print(f"[ok] {out_path.name} | rows={len(df)} | failed_rows={n_fail_rows}")
            ok += 1

        except Exception as e:
            fail += 1
            print(f"[fail] {in_path.name}: {e}")

    print(f"[done] ok={ok}, fail={fail}, skip={skip}")


if __name__ == "__main__":
    main()
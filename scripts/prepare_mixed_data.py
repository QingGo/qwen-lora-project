import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "你是一个乐于助人的AI助手。请使用 Markdown 格式回答：用 **加粗** 突出重点、用 ### 标题分层、用数字或项目符号分点列出内容。回答应当结构清晰、内容详实、易于阅读。"

MIX_RATIOS = {"alpaca": 0.70, "belle": 0.20, "replay": 0.10}


def load_alpaca(csv_path: str, n_samples: int, seed: int):
    df = pd.read_csv(csv_path)
    if n_samples > 0 and n_samples < len(df):
        random.seed(seed)
        indices = random.sample(range(len(df)), n_samples)
        df = df.iloc[indices].reset_index(drop=True)
    return df


def load_belle(json_path: str, n_samples: int, seed: int):
    with open(json_path) as f:
        lines = f.readlines()
    random.seed(seed)
    sampled = random.sample(lines, min(n_samples, len(lines)))
    return [json.loads(line) for line in sampled]


def build_conversations(row_or_dict, source: str, include_system: bool = True):
    if source == "alpaca":
        user = row_or_dict["instruction"].strip()
        inp = str(row_or_dict.get("input", "")).strip()
        if inp and inp not in ("nan", ""):
            user += "\n" + inp
        assistant = str(row_or_dict["output"]).strip()
    elif source == "belle":
        user = row_or_dict["instruction"].strip()
        inp = str(row_or_dict.get("input", "")).strip()
        if inp and inp not in ("nan", ""):
            user += "\n" + inp
        assistant = str(row_or_dict["output"]).strip()
    elif source == "replay":
        user = row_or_dict["instruction"].strip()
        inp = str(row_or_dict.get("input", "")).strip()
        if inp and inp not in ("nan", ""):
            user += "\n" + inp
        assistant = row_or_dict["output"].strip()
    else:
        raise ValueError(f"Unknown source: {source}")

    conversations = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if include_system:
        conversations.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"conversations": conversations}


def generate_replay(
    model, tokenizer, alpaca_df, n_replay: int, seed: int,
) -> list[dict]:
    random.seed(seed)
    indices = random.sample(range(len(alpaca_df)), min(n_replay, len(alpaca_df)))
    replay_data = []

    for idx in tqdm(indices, desc="Generating replay buffer"):
        row = alpaca_df.iloc[idx]
        user = row["instruction"].strip()
        inp = str(row.get("input", "")).strip()
        if inp and inp not in ("nan", ""):
            user += "\n" + inp

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        replay_data.append({"instruction": row["instruction"], "input": inp, "output": response})

    empty = sum(1 for r in replay_data if not r["output"].strip())
    print(f"Replay generated: {len(replay_data)} answers ({empty} empty)")
    return replay_data


def process_records(
    data, source: str, records: list, min_answer_len: int, include_system: bool
):
    for item in data:
        record = build_conversations(item, source, include_system)
        answer = record["conversations"][-1]["content"]
        if len(answer) < min_answer_len:
            continue
        record["source"] = source
        records.append(record)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed training data from Alpaca + BELLE + Replay buffer"
    )
    parser.add_argument("--alpaca_csv", default="data/alpaca-gpt4-data-zh/train.csv")
    parser.add_argument("--belle_json", default="data/belle-0.5M/Belle_open_source_0.5M.json")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--total_samples", type=int, default=3000)
    parser.add_argument("--min_answer_len", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_replay", action="store_true", help="Skip replay buffer generation")
    parser.add_argument("--no_system_prompt", action="store_true", help="Disable system prompt")
    parser.add_argument(
        "--replay_file", default=None, help="Pre-generated replay JSONL to reuse"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    n_alpaca = int(args.total_samples * MIX_RATIOS["alpaca"])
    n_belle = int(args.total_samples * MIX_RATIOS["belle"])
    n_replay = int(args.total_samples * MIX_RATIOS["replay"])
    if args.total_samples - n_alpaca - n_belle - n_replay > 0:
        n_alpaca += args.total_samples - n_alpaca - n_belle - n_replay

    include_system = not args.no_system_prompt

    # 1. Load Alpaca
    print(f"=== Loading Alpaca ({n_alpaca} target) ===")
    alpaca_df = load_alpaca(args.alpaca_csv, n_alpaca * 2, args.seed)

    # 2. Load BELLE
    print(f"=== Loading BELLE ({n_belle} target) ===")
    belle_raw = load_belle(args.belle_json, n_belle * 2, args.seed)

    # 3. Generate or load replay buffer
    replay_raw = []
    if not args.no_replay and n_replay > 0:
        if args.replay_file and Path(args.replay_file).exists():
            print(f"=== Loading cached replay buffer: {args.replay_file} ===")
            with open(args.replay_file) as f:
                replay_raw = [json.loads(line) for line in f if line.strip()]
        else:
            print(f"=== Generating replay buffer ({n_replay} answers) ===")
            print("Loading base model...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, trust_remote_code=True, use_fast=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            replay_raw = generate_replay(model, tokenizer, alpaca_df, n_replay * 2, args.seed)

            if args.replay_file:
                replay_path = Path(args.replay_file)
                replay_path.parent.mkdir(parents=True, exist_ok=True)
                with open(replay_path, "w", encoding="utf-8") as f:
                    for r in replay_raw:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"Cached replay buffer to: {args.replay_file}")
    else:
        print("=== Replay buffer skipped ===")

    # 4. Build conversations and filter
    records = []
    for source, data, target_n in [
        ("alpaca", [alpaca_df.iloc[i] for i in range(len(alpaca_df))], n_alpaca),
        ("belle", belle_raw, n_belle),
        ("replay", replay_raw, n_replay),
    ]:
        process_records(data, source, records, args.min_answer_len, include_system)
        source_records = [r for r in records if r["source"] == source]
        if len(source_records) > target_n:
            excess = len(source_records) - target_n
            to_remove = random.sample(
                [i for i, r in enumerate(records) if r["source"] == source], excess
            )
            for i in sorted(to_remove, reverse=True):
                records.pop(i)
        n_final = len([r for r in records if r["source"] == source])
        print(f"  {source}: {len(data)} raw → {n_final} filtered (target={target_n})")

    # 5. Shuffle and split
    random.shuffle(records)
    total = len(records)
    val_size = max(1, int(total * args.val_ratio))
    train_records = records[val_size:]
    val_records = records[:val_size]

    # 6. Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, recs in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                convs = r["conversations"]
                f.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")

    source_counts = {}
    for r in train_records + val_records:
        s = r["source"]
        source_counts[s] = source_counts.get(s, 0) + 1

    print(f"\n=== Mixed data ready ===")
    print(f"Train: {len(train_records)}  Val: {len(val_records)}  Total: {total}")
    print(f"Breakdown: {dict(source_counts)}")


if __name__ == "__main__":
    main()

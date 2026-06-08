import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from Alpaca-GPT4-ZH CSV")
    parser.add_argument(
        "--input",
        default="data/alpaca-gpt4-data-zh/train.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to extract (0 = all)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")

    if args.num_samples > 0 and args.num_samples < len(df):
        random.seed(args.seed)
        indices = random.sample(range(len(df)), args.num_samples)
        df = df.iloc[indices].reset_index(drop=True)
        print(f"Sampled {args.num_samples} items")

    records = []
    for _, row in df.iterrows():
        user_content = row["instruction"].strip()
        input_text = str(row.get("input", "")).strip() if pd.notna(row.get("input")) else ""
        if input_text and input_text not in ("nan", ""):
            user_content += "\n" + input_text
        assistant_content = str(row["output"]).strip()
        if not assistant_content:
            continue
        records.append({
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    random.seed(args.seed)
    random.shuffle(records)

    val_size = max(1, int(len(records) * args.val_ratio))
    train_records = records[val_size:]
    val_records = records[:val_size]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(train_records)} training samples to {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(val_records)} validation samples to {val_path}")


if __name__ == "__main__":
    main()

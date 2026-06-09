import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. "
    "Given a database schema, write a correct SQL query for the user's question."
)


def process_chunk(args):
    """Process one chunk of indices, return records that fit within max_len."""
    chunk_indices, model_path, max_len = args

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")

    records = []
    for idx in chunk_indices:
        item = ds[int(idx)]
        user = f"### Schema:\n{item['schema']}\n\n### Question:\n{item['question']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": item["query"].strip()},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if len(tokenizer.encode(text)) <= max_len:
            record = {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": item["query"].strip()},
                ]
            }
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare Text2SQL training data from SQaLe")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_samples", type=int, default=3000)
    parser.add_argument("--val_samples", type=int, default=300)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--num_proc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset metadata...")
    ds = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    total = len(ds)
    print(f"Total SQaLe samples: {total}")

    # Sample enough indices to likely find what we need
    # At max_len=2048, about 14% of samples fit. Sample 5x needed.
    sample_pool = min(int((args.train_samples + args.val_samples) * 8), total)
    indices = random.sample(range(total), sample_pool)

    # Split into chunks for parallel processing
    chunk_size = max(1, len(indices) // args.num_proc)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    print(f"Processing {len(indices)} samples across {len(chunks)} chunks ({args.num_proc} workers)...")

    all_records = []
    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [
            executor.submit(process_chunk, (chunk, args.model_path, args.max_length))
            for chunk in chunks
        ]
        for i, future in enumerate(futures):
            records = future.result()
            all_records.extend(records)
            if i % 3 == 0:
                print(f"  Chunk {i+1}/{len(chunks)}: +{len(records)} records, "
                      f"total={len(all_records)}")

    print(f"Kept {len(all_records)} / {len(indices)} ({len(all_records)/len(indices)*100:.1f}%)")

    # Shuffle and split
    random.shuffle(all_records)
    n_train = min(args.train_samples, len(all_records))
    n_val = min(args.val_samples, len(all_records) - n_train)
    train = all_records[:n_train]
    val = all_records[n_train : n_train + n_val]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {path}")

    print("Done.")


if __name__ == "__main__":
    main()

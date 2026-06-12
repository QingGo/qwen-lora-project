import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. "
    "Given a database schema, write a correct SQL query for the user's question."
)

STRONG_SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant.\n\n"
    "CRITICAL: Output ONLY the raw SQL query. Do NOT include any explanations, "
    "markdown code blocks (```sql), introductory text, or analysis. "
    "Just the SQL statement itself, nothing else."
)


def is_question_noise(question: str) -> tuple[bool, str]:
    q = question.strip()
    if not q:
        return True, "empty"

    if re.match(r'^\s*<Example\s+question\s+\d+>\s*$', q, re.IGNORECASE):
        return True, "placeholder"

    if len(q) < 35:
        instruction_patterns = (
            r'(?i)^(here\s+is|just\s+the|output\s+only|no\s+(numbering|trailing|markdown)'
            r'|do\s+not|write\s+the|looking\s+at|their\s+\w+\s+and'
            r'|from\s+\w+|user\s*->|#\s*\w|the\s+code\s+block'
            r'|make\s+sure|ensure\s+that|first,?\s+i\s+need'
            r'|only\s+(plain\s+text|the\s+code)|^no\s+\w+\.?\s*(just|only|the))'
        )
        if re.match(r'^(SELECT|WITH|FROM|CREATE|INSERT)\s', q, re.IGNORECASE):
            return True, "sql_fragment_in_question"
        if re.match(instruction_patterns, q):
            return True, "instruction_fragment"

    return False, ""


def is_sql_field_anomaly(question: str, sql: str) -> tuple[bool, str]:
    sq = sql.strip()

    if re.match(
        r'^\s*(CREATE\s+(TABLE|INDEX|VIEW|TRIGGER|DATABASE|SCHEMA)'
        r'|INSERT\s+(OR\s+REPLACE\s+)?INTO'
        r'|ALTER\s+(TABLE|INDEX)'
        r'|DROP\s+(TABLE|INDEX|VIEW)'
        r'|UPDATE\s+\w+\s+SET'
        r'|DELETE\s+FROM)',
        sq, re.IGNORECASE,
    ):
        ql = question.lower()
        write_verbs = (
            'create table', 'create index', 'create view',
            'insert into', 'insert data', 'add record', 'add a record',
            'update ', 'modify ', 'alter table',
            'delete ', 'remove ', 'drop ',
        )
        if not any(v in ql for v in write_verbs):
            return True, "ddl_or_write_mismatch"

    if re.search(
        r"SELECT\s+'(No\s|The\s+schema\s+does\s+not|Schema\s+does\s+not"
        r"|Unable\s+to|Cannot\s+find|No\s+relevant|No\s+direct|No\s+such"
        r"|The\s+required\s+data\s+is\s+not|Schema\s+limitation)",
        sq, re.IGNORECASE,
    ):
        return True, "placeholder_sql"

    return False, ""


def clean_question(question: str) -> str:
    q = question.strip()
    m = re.match(r'^\s*<([^>]{10,})>\s*$', q)
    if m:
        return m.group(1).strip()
    return q


def process_chunk(args):
    chunk_indices, model_path, max_len, use_strong, noise_stats_list = args

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")

    local_stats = defaultdict(int)
    sp = STRONG_SYSTEM_PROMPT if use_strong else SYSTEM_PROMPT

    records = []
    for idx in chunk_indices:
        item = ds[int(idx)]
        question = clean_question(item["question"])
        sql = item["query"].strip()

        is_noise_q, noise_type_q = is_question_noise(question)
        if is_noise_q:
            local_stats[f"filtered_q_{noise_type_q}"] += 1
            continue
        is_noise_sql, noise_type_sql = is_sql_field_anomaly(question, sql)
        if is_noise_sql:
            local_stats[f"filtered_sql_{noise_type_sql}"] += 1
            continue

        user_content = f"### Schema:\n{item['schema']}\n\n### Question:\n{question}"
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sql},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer.encode(full_text))
        if n_tokens <= max_len:
            local_stats["kept_after_length"] += 1
            record = {
                "conversations": [
                    {"role": "system", "content": sp},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": sql},
                ]
            }
            records.append(record)
        else:
            local_stats["filtered_length"] += 1
            if local_stats["filtered_length"] <= 3:
                q_tokens = len(tokenizer.encode(f"### Schema:\n{item['schema']}\n\n### Question:\n{question}"))
                a_tokens = len(tokenizer.encode(sql))
                sys_tokens = len(tokenizer.encode(f"<|im_start|>system\n{sp}<|im_end|>\n"))
                local_stats["_last_length_detail"] = f"  total={n_tokens} sys={sys_tokens} q={q_tokens} a={a_tokens}"

    noise_stats_list.append(dict(local_stats))
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
    parser.add_argument("--strong_prompt", action="store_true",
                        help="Use strong system prompt (Output ONLY raw SQL)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Disable noise filtering")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset metadata...")
    ds = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    total = len(ds)
    print(f"Total SQaLe samples: {total}")
    print(f"Max length: {args.max_length}")
    print(f"Noise filter: {'OFF' if args.no_filter else 'ON'}")
    print(f"System prompt: {'STRONG' if args.strong_prompt else 'weak'}")

    sample_pool = min(int((args.train_samples + args.val_samples) * 8), total)
    indices = random.sample(range(total), sample_pool)

    chunk_size = max(1, len(indices) // args.num_proc)
    chunks = [indices[i: i + chunk_size] for i in range(0, len(indices), chunk_size)]

    print(f"Processing {len(indices)} samples across {len(chunks)} chunks ({args.num_proc} workers)...")

    noise_stats_list = []
    all_records = []
    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [
            executor.submit(
                process_chunk,
                (chunk, args.model_path, args.max_length,
                 args.strong_prompt, noise_stats_list),
            )
            for chunk in chunks
        ]
        for i, future in enumerate(futures):
            records = future.result()
            all_records.extend(records)
            if i % 3 == 0:
                print(f"  Chunk {i+1}/{len(chunks)}: +{len(records)} records, "
                      f"total={len(all_records)}")

    total_processed = sum(
        sum(v for k, v in s.items() if not k.startswith("_"))
        for s in noise_stats_list
    ) + len(all_records)

    merged_stats = defaultdict(int)
    for s in noise_stats_list:
        for k, v in s.items():
            merged_stats[k] += v

    print(f"\nKept {len(all_records)} / {len(indices)} ({len(all_records)/len(indices)*100:.1f}%)")
    print(f"\nFilter stats:")
    for k in sorted(merged_stats):
        if k.startswith("filtered_"):
            print(f"  {k}: {merged_stats[k]} ({merged_stats[k]/len(indices)*100:.2f}%)")
        elif k == "kept_after_length":
            continue
    for s in noise_stats_list:
        if "_last_length_detail" in s:
            print(s["_last_length_detail"])
            break

    random.shuffle(all_records)
    n_train = min(args.train_samples, len(all_records))
    n_val = min(args.val_samples, len(all_records) - n_train)
    train = all_records[:n_train]
    val = all_records[n_train:n_train + n_val]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {path}")

    if train:
        sample = train[0]
        c = sample["conversations"]
        q_content = c[1]["content"]
        a_content = c[2]["content"]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        n_sys = len(tokenizer.encode(f"<|im_start|>system\n{c[0]['content']}<|im_end|>\n"))
        n_q = len(tokenizer.encode(c[1]["content"]))
        n_a = len(tokenizer.encode(c[2]["content"]))
        n_total = len(tokenizer.encode(tokenizer.apply_chat_template(c, tokenize=False)))
        print(f"\nSample token profile (first train record):")
        print(f"  system overhead: {n_sys}")
        print(f"  question+schema: {n_q}")
        print(f"  answer (SQL):    {n_a}")
        print(f"  total:           {n_total}")

    print("\nDone.")


if __name__ == "__main__":
    main()

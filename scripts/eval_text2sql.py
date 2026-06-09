import argparse
import json
import os
import random
import sqlite3
import time

import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
if not os.getenv("DEEPSEEK_API_KEY"):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")) as f:
        for line in f:
            if "DEEPSEEK_API_KEY" in line:
                os.environ["DEEPSEEK_API_KEY"] = line.strip().split("=", 1)[1].strip().strip('"').strip("'")

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. "
    "Given a database schema, write a correct SQL query for the user's question."
)

JUDGE_SYSTEM = """你是一个专业的 SQL 查询质量评估专家。你会收到一个问题、数据库 schema 和一个目标 SQL（正确答案），以及两个模型生成的 SQL（标记为 SQL A 和 SQL B）。

请从以下三个维度对每个 SQL 进行 1-5 分评分：

1. **executability（可执行性）**：SQL 语法是否正确？表名、列名是否和 schema 一致？JOIN 条件是否合法？
   - 1: 语法错误或引用了不存在的表/列
   - 3: 语法正确但 JOIN 条件可能有误
   - 5: 语法完美，所有引用正确

2. **logical_correctness（逻辑正确性）**：SQL 是否正确回答了问题？是否使用了正确的 WHERE 条件、GROUP BY、ORDER BY？
   - 1: 完全没有回答正确的问题
   - 3: 部分正确但遗漏了关键条件或有多余条件
   - 5: 完全正确，准确回答了问题

3. **conciseness（简洁性）**：SQL 是否简洁高效？是否避免了不必要的子查询、冗余条件或过度复杂的写法？
   - 1: 不必要的复杂或冗余
   - 3: 基本简洁但可优化
   - 5: 优雅简洁，最直接的写法

请严格按 1-5 整数评分，以 JSON 格式输出：
{
  "sql_a": {"executability": <1-5>, "logical_correctness": <1-5>, "conciseness": <1-5>, "brief_reason": "<一句话>"},
  "sql_b": {"executability": <1-5>, "logical_correctness": <1-5>, "conciseness": <1-5>, "brief_reason": "<一句话>"},
  "comparison": "<简短说明哪个 SQL 更好以及原因>"
}"""


def generate_sql(model, tokenizer, schema: str, question: str, max_new_tokens: int = 256) -> str:
    user = f"### Schema:\n{schema}\n\n### Question:\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    sql = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    sql = sql.rstrip(tokenizer.eos_token).strip()
    return sql


def load_val_samples(n: int = 20):
    val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "val.jsonl")
    records = []
    with open(val_path) as f:
        for line in f:
            r = json.loads(line)
            convs = r["conversations"]
            user_content = None
            for m in convs:
                if m["role"] == "user":
                    user_content = m["content"]
            if user_content and "### Schema:" in user_content and "### Question:" in user_content:
                records.append(r)
    random.shuffle(records)
    return records[:n]


def extract_schema_and_question(record: dict):
    convs = record["conversations"]
    user = ""
    gold_sql = ""
    for m in convs:
        if m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant":
            gold_sql = m["content"].strip()
    schema = ""
    question = ""
    if "### Schema:" in user and "### Question:" in user:
        parts = user.split("### Question:", 1)
        schema = parts[0].replace("### Schema:", "").strip()
        question = parts[1].strip()
    return schema, question, gold_sql


def validate_sql(schema_ddl: str, sql: str) -> tuple[bool, str]:
    conn = sqlite3.connect(":memory:")
    try:
        for stmt in schema_ddl.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                conn.execute(stmt)
        conn.execute(sql)
        conn.close()
        return True, "OK"
    except Exception as e:
        conn.close()
        return False, str(e)[:200]


def judge_sql(client, schema, question, gold_sql, sql_a, sql_b, sql_a_label, sql_b_label) -> dict | None:
    prompt = f"""请评估以下两个 SQL 查询。

【Schema（部分）】
{schema[:3000]}

【问题】
{question}

【参考答案 SQL】
{gold_sql}

【SQL A】（来自 {sql_a_label}）
{sql_a}

【SQL B】（来自 {sql_b_label}）
{sql_b}

请按照 JSON 格式输出评分："""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0, max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"):raw.rindex("}") + 1]
            result = json.loads(raw)
            if "sql_a" in result and "sql_b" in result:
                return result
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA vs base Text2SQL quality")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="./outputs/qwen2.5-7b-lora-adapter")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        return

    random.seed(args.seed)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    val_samples = load_val_samples(args.n_samples)
    print(f"Loaded {len(val_samples)} validation samples\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)

    dims = ["executability", "logical_correctness", "conciseness"]
    base_scores = {d: [] for d in dims}
    lora_scores = {d: [] for d in dims}
    base_exec_ok = 0
    lora_exec_ok = 0
    base_wins = lora_wins = ties = 0
    results = []
    total = len(val_samples)

    for idx, record in enumerate(val_samples):
        schema, question, gold_sql = extract_schema_and_question(record)
        if not schema or not question:
            continue

        print(f"\n[{idx + 1}/{total}] {question[:100]}...")

        model.disable_adapter_layers()
        base_sql = generate_sql(model, tokenizer, schema, question, max_new_tokens=args.max_new_tokens)
        model.enable_adapter_layers()
        lora_sql = generate_sql(model, tokenizer, schema, question, max_new_tokens=args.max_new_tokens)

        base_valid, base_err = validate_sql(schema, base_sql)
        lora_valid, lora_err = validate_sql(schema, lora_sql)
        if base_valid:
            base_exec_ok += 1
        if lora_valid:
            lora_exec_ok += 1

        print(f"  Base SQL: {base_sql[:120]}{'...' if len(base_sql)>120 else ''}")
        print(f"  LoRA SQL: {lora_sql[:120]}{'...' if len(lora_sql)>120 else ''}")
        print(f"  Base exec: {'OK' if base_valid else base_err[:80]}")
        print(f"  LoRA exec: {'OK' if lora_valid else lora_err[:80]}")

        base_is_a = random.choice([True, False])
        sql_a = base_sql if base_is_a else lora_sql
        sql_b = lora_sql if base_is_a else base_sql
        label_a = "Base" if base_is_a else "LoRA"
        label_b = "LoRA" if base_is_a else "Base"

        judge_result = judge_sql(client, schema, question, gold_sql, sql_a, sql_b, label_a, label_b)

        if judge_result:
            sa = judge_result["sql_a"]
            sb = judge_result["sql_b"]
            bd = sa if base_is_a else sb
            ld = sb if base_is_a else sa
            bd.pop("brief_reason", None)
            ld.pop("brief_reason", None)

            bt = sum(bd.values())
            lt = sum(ld.values())
            for d in dims:
                base_scores[d].append(bd[d])
                lora_scores[d].append(ld[d])

            w = "LoRA" if lt > bt else ("Base" if bt > lt else "TIE")
            if w == "LoRA":
                lora_wins += 1
            elif w == "Base":
                base_wins += 1
            else:
                ties += 1

            print(f"  Judge: Base={bt}, LoRA={lt} → {w}")
            print(f"  {judge_result.get('comparison', '')[:120]}")

            results.append({
                "question": question,
                "gold_sql": gold_sql,
                "base_sql": base_sql,
                "lora_sql": lora_sql,
                "base_valid": base_valid,
                "lora_valid": lora_valid,
                "base_scores": bd,
                "lora_scores": ld,
                "comparison": judge_result.get("comparison", ""),
            })

    # Summary
    n = len(results)
    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete (n={n})")
    print(f"{'=' * 60}")

    print(f"\n{'Execution Rate':<25} {'Base':>10} {'LoRA':>10}")
    print(f"{'  Valid SQL':<25} {base_exec_ok/total*100:>9.1f}% {lora_exec_ok/total*100:>9.1f}%")

    print(f"\n{'Dimension':<25} {'Base':>8} {'LoRA':>8} {'Δ':>8}")
    print("-" * 51)
    for d in dims:
        if base_scores[d]:
            avg_b = sum(base_scores[d]) / len(base_scores[d])
            avg_l = sum(lora_scores[d]) / len(lora_scores[d])
            print(f"{d:<25} {avg_b:>8.2f} {avg_l:>8.2f} {avg_l-avg_b:>+8.2f}")

    print(f"\nWin: Base={base_wins}, LoRA={lora_wins}, Tie={ties} / {n}")
    print(f"Base exec rate: {base_exec_ok}/{total} ({base_exec_ok/total*100:.0f}%)")
    print(f"LoRA exec rate: {lora_exec_ok}/{total} ({lora_exec_ok/total*100:.0f}%)")

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "text2sql_eval.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

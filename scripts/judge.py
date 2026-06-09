import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()

TEST_QUESTIONS = [
    "请解释一下深度学习中的过拟合现象，以及如何避免。",
    "用 Python 写一个归并排序的实现。",
    "介绍一下中国四大名著。",
    "如何提高工作效率？给出实用建议。",
    "什么是机器学习中的交叉验证？",
    "描述一下量子计算的基本原理。",
    "为什么锻炼对健康有益？",
]

JUDGE_SYSTEM = """你是一个专业的AI回答质量评估专家。你会收到一个问题，以及两个来自不同模型的回答（标记为"答案 A"和"答案 B"）。

你的任务是对每个答案在以下五个维度上进行1-5分评分（1=最差，5=最好）：

1. **helpfulness（实用性）**：回答是否切实解决了用户的问题？是否提供了有价值的、可操作的信息？
   - 1: 完全没有回答问题
   - 3: 部分回答了问题但不够深入
   - 5: 全面深入，提供极有价值的实用信息

2. **accuracy（准确性）**：回答中的事实和信息是否准确？
   - 1: 存在严重事实错误或明显的幻觉
   - 3: 基本正确但有小错误或不精确
   - 5: 完全准确，所有信息都可靠

3. **completeness（完整性）**：回答是否覆盖了问题的关键方面？
   - 1: 极其不完整，只触及皮毛
   - 3: 覆盖了主要内容但缺少关键点
   - 5: 全面覆盖所有关键方面，没有重要遗漏

4. **structure（结构性）**：回答的组织结构是否清晰？是否层次分明、易于阅读？
   - 1: 混乱无序，难以理解
   - 3: 基本有结构但条理性一般
   - 5: 结构极佳，层次分明，一目了然

5. **style_alignment（风格匹配度）**：回答是否体现了详尽、结构化、教育性的中文指令回答风格？
   - 1: 完全不符合
   - 3: 部分匹配
   - 5: 完美匹配，回答详尽、分点清晰、有教育意义

评分规则：
- 必须严格按1-5整数评分，不能使用小数
- 评分需要区分度：如果两个答案质量接近，仍然要在有差异的维度上打出不同分数
- 每个维度的评分必须有明显依据，不能所有维度都给相同分数

请以 JSON 格式输出评分结果，不要输出任何其他内容：

{
  "answer_a": {
    "helpfulness": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "structure": <1-5>,
    "style_alignment": <1-5>,
    "brief_reason": "<一句话总结这个答案的优缺点>"
  },
  "answer_b": {
    "helpfulness": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "structure": <1-5>,
    "style_alignment": <1-5>,
    "brief_reason": "<一句话总结这个答案的优缺点>"
  },
  "comparison": "<简短说明哪个答案更好以及原因（1-2句话）>"
}"""


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return f"""请评估以下两个答案。

【问题】
{question}

【答案 A】
{answer_a}

【答案 B】
{answer_b}

请按照 JSON 格式输出评分："""


def parse_judge_response(raw: str) -> dict | None:
    try:
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def judge_with_retry(client: OpenAI, model_name: str, question: str, answer_a: str, answer_b: str, max_retries: int = 3) -> dict | None:
    prompt = build_judge_prompt(question, answer_a, answer_b)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            result = parse_judge_response(raw)
            if result and "answer_a" in result and "answer_b" in result:
                return result
        except Exception as e:
            print(f"  Judge API error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM Judge 评测 (DeepSeek)")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="./outputs/qwen2.5-7b-lora-adapter")
    parser.add_argument("--judge_model", default="deepseek-v4-flash")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_questions", type=int, default=20, help="Number of questions to evaluate (from validation data)")
    parser.add_argument("--val_data", default="./data/val.jsonl", help="Validation data for test questions")
    parser.add_argument("--baseline_name", default=None, help="Save result as a named baseline (e.g. 'baseline-v1', 'lr-5e-5')")
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    test_questions = TEST_QUESTIONS
    val_path = Path(args.val_data)
    if val_path.exists():
        with open(val_path) as f:
            val_records = [json.loads(line) for line in f if line.strip()]
        val_questions = []
        for r in val_records:
            for m in r.get("conversations", r.get("messages", [])):
                if m.get("role") == "user" and m.get("content"):
                    q = m["content"].strip()
                    if len(q) > 10:
                        val_questions.append(q)
                    break
        if val_questions:
            random.shuffle(val_questions)
            test_questions = val_questions[:args.num_questions]
            print(f"Using {len(test_questions)} questions from validation data")
    else:
        print(f"Validation data not found, using {len(test_questions)} default questions")

    print("Loading model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print(f"\nJudge: {args.judge_model} | Questions: {len(test_questions)}")
    print("=" * 70)

    all_results = []
    scores_base = {k: [] for k in ["helpfulness", "accuracy", "completeness", "structure", "style_alignment"]}
    scores_lora = {k: [] for k in ["helpfulness", "accuracy", "completeness", "structure", "style_alignment"]}

    for i, question in enumerate(test_questions):
        print(f"\n[{i + 1}/{len(test_questions)}] {question}")

        model.disable_adapter_layers()
        base_answer = generate(model, tokenizer, question, max_new_tokens=args.max_new_tokens)
        model.enable_adapter_layers()
        lora_answer = generate(model, tokenizer, question, max_new_tokens=args.max_new_tokens)

        # Position bias mitigation: randomize answer order
        base_is_a = random.choice([True, False])
        answer_a = base_answer if base_is_a else lora_answer
        answer_b = lora_answer if base_is_a else base_answer

        judge_result = judge_with_retry(client, args.judge_model, question, answer_a, answer_b)

        if judge_result is None:
            print("  Judge failed, skipping")
            continue

        score_a = judge_result["answer_a"]
        score_b = judge_result["answer_b"]

        # Map back to base/lora based on randomization
        base_scores_detail = score_a if base_is_a else score_b
        lora_scores_detail = score_b if base_is_a else score_a

        base_reason = base_scores_detail.pop("brief_reason", "")
        lora_reason = lora_scores_detail.pop("brief_reason", "")

        base_total = sum(base_scores_detail[k] for k in scores_base)
        lora_total = sum(lora_scores_detail[k] for k in scores_lora)

        for k in scores_base:
            scores_base[k].append(base_scores_detail[k])
            scores_lora[k].append(lora_scores_detail[k])

        winner = "LORA" if lora_total > base_total else ("BASE" if base_total > lora_total else "TIE")
        order_note = f"(Base={'A' if base_is_a else 'B'}, LoRA={'B' if base_is_a else 'A'})"

        print(f"  Base  {order_note}: {base_scores_detail}  total={base_total}")
        print(f"  LoRA  {order_note}: {lora_scores_detail}  total={lora_total}")
        print(f"  Winner: {winner} | {judge_result.get('comparison', '')}")

        all_results.append({
            "question": question,
            "base_answer": base_answer,
            "lora_answer": lora_answer,
            "base_is_a": base_is_a,
            "base_scores": base_scores_detail,
            "lora_scores": lora_scores_detail,
            "base_reason": base_reason,
            "lora_reason": lora_reason,
            "comparison": judge_result.get("comparison", ""),
        })

    # Summary
    print("\n" + "=" * 70)
    print("评测汇总")
    print("=" * 70)
    print(f"{'维度':<20} {'基座':<10} {'LoRA':<10} {'变化':<10}")
    print("-" * 50)

    all_diffs = []
    for k in scores_base:
        avg_b = sum(scores_base[k]) / len(scores_base[k]) if scores_base[k] else 0
        avg_l = sum(scores_lora[k]) / len(scores_lora[k]) if scores_lora[k] else 0
        diff = avg_l - avg_b
        all_diffs.append(diff)
        sign = "+" if diff >= 0 else ""
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"{k:<20} {avg_b:<10.2f} {avg_l:<10.2f} {sign}{diff:.2f} {arrow}")

    avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
    print(f"\n{'平均总分差 (LoRA-Base)':<20} {'':<10} {'':<10} {'+' if avg_diff >= 0 else ''}{avg_diff:.2f}")

    winner_count = sum(1 for r in all_results if sum(v for k, v in r["lora_scores"].items() if k != "brief_reason") > sum(v for k, v in r["base_scores"].items() if k != "brief_reason"))
    tie_count = sum(1 for r in all_results if sum(v for k, v in r["lora_scores"].items() if k != "brief_reason") == sum(v for k, v in r["base_scores"].items() if k != "brief_reason"))
    base_win = len(all_results) - winner_count - tie_count
    print(f"胜出次数: Base={base_win}, LoRA={winner_count}, Tie={tie_count} / {len(all_results)} total")

    # Save results
    output_file = Path("outputs") / "judge_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_file}")

    if args.baseline_name:
        git_commit = None
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            pass

        baseline = {
            "name": args.baseline_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "num_questions": len(test_questions),
            "dim_scores": {
                k: {
                    "base": round(sum(scores_base[k]) / len(scores_base[k]), 2) if scores_base[k] else 0,
                    "lora": round(sum(scores_lora[k]) / len(scores_lora[k]), 2) if scores_lora[k] else 0,
                    "diff": round(
                        sum(scores_lora[k]) / len(scores_lora[k]) - sum(scores_base[k]) / len(scores_base[k]), 2
                    ) if scores_lora[k] and scores_base[k] else 0,
                }
                for k in scores_base
            },
            "total_avg": {
                "base": round(sum(sum(scores_base[k]) for k in scores_base) / len(scores_base["helpfulness"]), 2) if scores_base["helpfulness"] else 0,
                "lora": round(sum(sum(scores_lora[k]) for k in scores_lora) / len(scores_lora["helpfulness"]), 2) if scores_lora["helpfulness"] else 0,
            },
            "win_counts": {"base": base_win, "lora": winner_count, "tie": tie_count},
        }
        baseline["total_avg"]["diff"] = round(baseline["total_avg"]["lora"] - baseline["total_avg"]["base"], 2)

        baseline_file = Path("outputs") / "baselines.jsonl"
        with open(baseline_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(baseline, ensure_ascii=False) + "\n")
        print(f"基线已添加至: {baseline_file}")


if __name__ == "__main__":
    main()

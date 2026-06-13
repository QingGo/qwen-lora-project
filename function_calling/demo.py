"""
Hermes Function Calling Demo — Qwen2.5-7B-Instruct (base model, no fine-tuning)

Format: <tools> + <tool_call> (Hermes/OpenHermes convention, natively supported by Qwen2.5)

Usage:
    uv run python function_calling/demo.py
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Qwen2.5-7B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_with_tools(model, tokenizer, messages, tools, max_new_tokens=512):
    """Generate a response that may include <tool_call> blocks."""
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_tool_calls(response: str) -> list[dict]:
    """Extract <tool_call> blocks from model response."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, response, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m))
        except json.JSONDecodeError:
            print(f"  [WARN] Failed to parse tool_call: {m[:120]}")
    return calls


# ── Tool implementations ──

def get_weather(city: str, unit: str = "celsius") -> str:
    """Simulated weather API."""
    temps = {"北京": 22, "上海": 25, "深圳": 28, "东京": 18, "纽约": 15, "伦敦": 12}
    temp = temps.get(city, 20)
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
    return json.dumps({"city": city, "temperature": temp, "unit": unit, "condition": "晴"}, ensure_ascii=False)


def calculate(expression: str) -> str:
    """Safe calculator."""
    allowed = set("0123456789+-*/().%^ ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "表达式包含不允许的字符"})
    try:
        result = eval(expression.replace("^", "**"))
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_database(query: str, limit: int = 5) -> str:
    """Simulated database search."""
    demo_db = {
        "员工": [{"name": "张三", "dept": "研发", "salary": 15000},
                 {"name": "李四", "dept": "销售", "salary": 12000}],
        "产品": [{"name": "Widget A", "price": 99}, {"name": "Widget B", "price": 199}],
    }
    results = []
    for table, rows in demo_db.items():
        if table in query:
            results = rows[:limit]
            break
    if not results:
        results = [{"message": f"未找到关于 '{query}' 的数据"}]
    return json.dumps(results, ensure_ascii=False)


TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_database": search_database,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持 + - * / () ^",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 2+3*4"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "在数据库中搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def run_conversation(model, tokenizer, user_query: str):
    """Single-turn or multi-turn with tool execution loop."""
    messages = [
        {"role": "system", "content": "你是一个有用的助手。你可以调用函数来获取信息。"},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}")

    for turn in range(3):  # max 3 turns
        response = generate_with_tools(model, tokenizer, messages, TOOLS)
        tool_calls = parse_tool_calls(response)
        content_only = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()

        if tool_calls:
            print(f"\n[Turn {turn+1}] Model wants to call {len(tool_calls)} tool(s):")
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                print(f"  → {name}({json.dumps(args, ensure_ascii=False)})")

                # Execute
                func = TOOL_MAP.get(name)
                if func:
                    if isinstance(args, dict):
                        result = func(**args)
                    else:
                        result = func(args)
                else:
                    result = json.dumps({"error": f"Unknown function: {name}"})
                print(f"  ← Result: {result[:200]}")

                # Add assistant message with tool_call
                messages.append({
                    "role": "assistant",
                    "content": content_only if content_only else None,
                    "tool_calls": [{
                        "function": {"name": name, "arguments": args if isinstance(args, dict) else str(args)}
                    }],
                })
                # Add tool response
                messages.append({"role": "tool", "content": result})
        else:
            print(f"\n[Turn {turn+1}] Model final answer (no tool calls):")
            print(f"  {response}")
            break
    else:
        print("\n[Max turns reached]")


def main():
    print("Loading model (this may take a moment)...")
    model, tokenizer = load_model()
    print("Model loaded. Starting demo.\n")

    test_queries = [
        "北京今天天气怎么样？",
        "计算 123 * 456 + 789",
        "帮我查一下数据库里的员工信息",
        "上海和深圳哪个更热？",
        "计算 (100 + 200) * 3 然后告诉我结果",
    ]

    for query in test_queries:
        run_conversation(model, tokenizer, query)


if __name__ == "__main__":
    main()

import argparse
import json
import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "请解释一下深度学习中的过拟合现象，以及如何避免。",
    "用 Python 写一个归并排序的实现。",
    "介绍一下中国四大名著。",
    "如何提高工作效率？给出实用建议。",
    "什么是机器学习中的交叉验证？",
    "描述一下量子计算的基本原理。",
    "为什么锻炼对健康有益？",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> dict:
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
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"text": response, "len": len(response), "tokens": outputs.shape[1] - inputs["input_ids"].shape[1]}


def compare_generations(model, tokenizer):
    print("\n" + "=" * 70)
    print("对比生成：基座模型 vs LoRA 微调模型")
    print("=" * 70)
    print("注意：LoRA 学的是风格，不是背答案。关注风格、详细度、结构化变化。")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─' * 70}")
        print(f"问题 {i + 1}: {prompt}")

        model.disable_adapter_layers()
        out_base = generate(model, tokenizer, prompt)
        model.enable_adapter_layers()
        out_lora = generate(model, tokenizer, prompt)

        print(f"\n[基座] ({out_base['len']} 字, {out_base['tokens']} tokens)")
        print(out_base["text"][:500])
        print(f"\n[LoRA] ({out_lora['len']} 字, {out_lora['tokens']} tokens)")
        print(out_lora["text"][:500])

        if out_base["text"] == out_lora["text"]:
            print("  -> 输出完全相同，LoRA 未产生可观察的变化")


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调效果评测")
    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="./outputs/qwen2.5-7b-lora-adapter")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model + LoRA adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.print_trainable_parameters()

    print(f"\nLoRA config: rank={model.peft_config['default'].r}, "
          f"alpha={model.peft_config['default'].lora_alpha}, "
          f"dropout={model.peft_config['default'].lora_dropout}")
    print(f"Target modules: {sorted(model.peft_config['default'].target_modules)}")
    print(f"LoRA scale (alpha/r): {model.peft_config['default'].lora_alpha / model.peft_config['default'].r}")

    compare_generations(model, tokenizer)

    print("\n" + "=" * 70)
    print("训练数据特征参考")
    print("=" * 70)
    print("Alpaca-GPT4-ZH: 详尽结构化中文回答，100-300 字，常见列表/分点")
    print("若 LoRA 有效：回答长度、结构化程度应更接近训练数据风格")
    print("若输出相同：LoRA 学习不充分（数据量/epoch/学习率导致）")
    print("\nDone.")


if __name__ == "__main__":
    main()

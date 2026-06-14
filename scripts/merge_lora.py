#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, producing a standalone HF checkpoint.

Usage:
    uv run python scripts/merge_lora.py \
        --base models/Qwen2.5-7B-Instruct \
        --adapter outputs/text2sql/adapters/l5-best \
        --output deployments/gguf/qwen7b-text2sql-merged

The output is a standard HuggingFace model (no PEFT dependency needed for inference).
Suitable for subsequent GGUF conversion, vLLM deployment, etc.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Path to base model (e.g. models/Qwen2.5-7B-Instruct)")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Precision for merged model (default: bfloat16)")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Base model: {args.base}")
    print(f"Adapter:    {args.adapter}")
    print(f"Output:     {args.output}")
    print(f"Dtype:      {args.dtype}")

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    merged.save_pretrained(args.output, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print("Done.")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-7B LoRA Fine-tuning")

    parser.add_argument("--model_path", default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="./data/train.jsonl")
    parser.add_argument("--eval_data_path", default=None)
    parser.add_argument("--output_dir", default="./outputs/qwen2.5-7b-lora-output")
    parser.add_argument("--adapter_dir", default="./outputs/qwen2.5-7b-lora-adapter")

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=30)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--deepspeed_config", default=None,
                        help="Path to DeepSpeed config JSON. Omit to skip DeepSpeed.")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_and_prepare_data(args: argparse.Namespace, tokenizer):
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
    else:
        split = dataset.train_test_split(test_size=0.1, seed=args.seed)
        dataset = split["train"]
        eval_dataset = split["test"]

    def preprocess(example):
        messages = example["conversations"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encodings = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return encodings

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

    return tokenized_dataset, tokenized_eval


def data_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {}
    for key in ("input_ids", "attention_mask"):
        padded = [
            f[key] + [tokenizer.pad_token_id if key == "input_ids" else 0] * (max_len - len(f[key]))
            for f in features
        ]
        batch[key] = torch.tensor(padded, dtype=torch.long)

    labels = batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    batch["attention_mask"] = batch["attention_mask"].bool()
    return batch


def make_data_collator(tokenizer):
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {}
        for key in ("input_ids", "attention_mask"):
            pad_value = tokenizer.pad_token_id if key == "input_ids" else 0
            padded = [
                f[key] + [pad_value] * (max_len - len(f[key]))
                for f in features
            ]
            batch[key] = torch.tensor(padded, dtype=torch.long)

        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["attention_mask"] = batch["attention_mask"].bool()
        return batch
    return collate


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    is_deepspeed = args.deepspeed_config is not None
    strategy = "DeepSpeed " + (os.path.basename(args.deepspeed_config) if is_deepspeed else "")
    if is_deepspeed:
        with open(args.deepspeed_config) as f:
            ds_cfg = json.load(f)
        stage = ds_cfg.get("zero_optimization", {}).get("stage", "?")
        strategy = f"DeepSpeed ZeRO-{stage}"
    else:
        strategy = "Single-GPU (no DeepSpeed)"

    print(f"=== Training Strategy: {strategy} ===")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Batch size={args.batch_size}, Grad accum={args.grad_accum}, Epochs={args.num_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.enable_input_require_grads()

    tokenized_dataset, tokenized_eval = load_and_prepare_data(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        fp16=False,
        bf16=True,
        deepspeed=args.deepspeed_config,
        report_to="tensorboard",
        gradient_checkpointing=True,
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=make_data_collator(tokenizer),
    )

    print("\n--- Pre-training evaluation (step 0) ---")
    eval_result_before = trainer.evaluate()
    initial_eval_loss = eval_result_before.get("eval_loss", None)
    print(f"Initial eval_loss: {initial_eval_loss:.4f}" if initial_eval_loss else "Initial eval_loss: N/A")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print("\n--- Post-training evaluation ---")
    eval_result_after = trainer.evaluate()
    final_eval_loss = eval_result_after.get("eval_loss", None)
    print(f"Final eval_loss: {final_eval_loss:.4f}" if final_eval_loss else "Final eval_loss: N/A")

    trainer.save_model(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    # Export loss history to JSON
    loss_history = {"train": [], "eval": []}
    if initial_eval_loss is not None:
        loss_history["eval"].append({"step": 0, "eval_loss": initial_eval_loss})

    state_path = os.path.join(args.output_dir, sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )[-1], "trainer_state.json")
    with open(state_path) as f:
        trainer_state = json.load(f)

    for entry in trainer_state["log_history"]:
        if "loss" in entry and "eval_loss" not in entry:
            loss_history["train"].append({"step": entry["step"], "loss": entry["loss"]})
        if "eval_loss" in entry:
            loss_history["eval"].append({"step": entry["step"], "eval_loss": entry["eval_loss"]})

    if final_eval_loss is not None:
        last_step = steps[-1] if (steps := loss_history["train"]) else 0
        loss_history["eval"].append({"step": last_step + 1, "eval_loss": final_eval_loss})

    loss_history_path = os.path.join(args.output_dir, "loss_history.json")
    with open(loss_history_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Training time: {elapsed / 60:.1f} minutes")
    print(f"LoRA adapter saved to: {args.adapter_dir}")
    print(f"Loss history saved to: {loss_history_path}")
    print(f"Training strategy used: {strategy}")


if __name__ == "__main__":
    main()

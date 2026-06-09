#!/bin/bash
# 单卡训练启动脚本 (不使用 DeepSpeed)
#
# 用法:
#   bash scripts/launch_single.sh
#
# 可选参数覆盖:
#   bash scripts/launch_single.sh --lora_rank 8 --batch_size 4 --num_epochs 3

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}

echo "=== 单卡训练模式 (无 DeepSpeed) ==="
echo ""

# 预处理数据
echo "[1/2] 准备数据..."
$PYTHON scripts/prepare_data.py --num_samples 5000
echo ""

# 训练 (LoRA: q,k,v,o,gate_proj, rank=32, alpha=16 -> ~38M params)
echo "[2/2] 开始训练..."
$PYTHON train_qwen_lora.py \
    --model_path ./models/Qwen2.5-7B-Instruct \
    --data_path ./data/train.jsonl \
    --eval_data_path ./data/val.jsonl \
    --output_dir ./outputs/qwen2.5-7b-lora-output \
    --adapter_dir ./outputs/qwen2.5-7b-lora-adapter \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj \
    --batch_size 2 \
    --grad_accum 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --save_steps 500 \
    "$@"

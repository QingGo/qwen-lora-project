#!/bin/bash
# 多卡 DeepSpeed 训练启动脚本
#
# 用法:
#   bash scripts/launch_multi.sh <num_gpus> <zero_stage> [extra_args...]
#
# 示例:
#   bash scripts/launch_multi.sh 2 2                   # 2卡 ZeRO-2
#   bash scripts/launch_multi.sh 4 3                   # 4卡 ZeRO-3
#   bash scripts/launch_multi.sh 1 2                   # 1卡 ZeRO-2 (调试用)
#   bash scripts/launch_multi.sh 2 2 --lora_rank 8     # 传额外参数

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS=${1:-1}
ZERO_STAGE=${2:-2}
shift 2 2>/dev/null || true

case "$ZERO_STAGE" in
    2) DS_CONFIG="configs/ds_zero2.json" ;;
    3) DS_CONFIG="configs/ds_zero3.json" ;;
    *) echo "Error: zero_stage must be 2 or 3"; exit 1 ;;
esac

echo "=== DeepSpeed ZeRO-${ZERO_STAGE} (${NUM_GPUS} GPUs) ==="

deepspeed --num_gpus="$NUM_GPUS" train_qwen_lora.py \
    --model_path ./models/Qwen2.5-7B-Instruct \
    --data_path ./data/train.jsonl \
    --eval_data_path ./data/val.jsonl \
    --output_dir ./outputs/qwen2.5-7b-lora-output \
    --adapter_dir ./outputs/qwen2.5-7b-lora-adapter \
    --deepspeed_config "$DS_CONFIG" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 2 \
    --grad_accum 4 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --save_steps 500 \
    "$@"

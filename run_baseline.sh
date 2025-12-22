#!/usr/bin/env bash
set -euo pipefail
timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"

echo "===== Stage 1: 生成 embeddings ====="
python -m methods.baseline_experiments.lightgcn \
  --data-dir /root/shared-nvme/data \
  --embeddings-dir /root/shared-nvme/data/embedding/lightgcn \
  --epochs 1 \
  --d-model 256 \
  --batch-size 256 \
  --embedding-dim 512 \
  --d-model 256 \
  --max-seq-len 256 \
  > "$log_dir/transformer_${timestamp}.log" 2>&1

echo "===== Stage 2: 训练下游任务 ====="
python -m training_pipeline.train \
  --data-dir /root/shared-nvme/data \
  --embeddings-dir /root/shared-nvme/data/embedding/lightgcn \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name baseline_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir /root/shared-nvme/data/embedding/lightgcn \
  > "$log_dir/baseline_${timestamp}.log" 2>&1

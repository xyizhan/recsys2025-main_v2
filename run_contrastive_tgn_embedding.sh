#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_tgn_v1"
log_file="$log_dir/contrastive_tgn_${timestamp}.log"

echo "===== Stage 1: 生成 TGN Embedding ====="
python -m methods.contrastive_tgn.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 256 \
  --item-bucket-count 100000 \
  --num-negatives 64 \
  --device cuda:0 \
  --log-interval 20 \
  2>&1 | tee "$log_file"

echo "===== Stage 2: 训练下游任务 ====="
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name tgn_baseline_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$embeddings_dir" \
  2>&1 | tee -a "$log_file"

echo "TGN embedding + downstream training finished. Logs: $log_file"

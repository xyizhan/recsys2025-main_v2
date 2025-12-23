#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_tgn_mean_ode"
score_dir="../scores_out/tgn_mean_ode"
mkdir -p "$score_dir"
stage1_log="$log_dir/contrastive_tgn_mean_ode_stage1_${timestamp}.log"
stage2_log="$log_dir/contrastive_tgn_mean_ode_stage2_${timestamp}.log"

echo "===== Stage 1: 生成 TGN Embedding (Mean 邻域 + ODE) ====="
echo "Logs: $stage1_log"
nohup python -m methods.contrastive_tgn.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 256 \
  --item-bucket-count 100000 \
  --neighbor-agg mean \
  --neighbor-k 6 \
  --time-update ode \
  --decay-rate 0.6 \
  --ode-steps 4 \
  --num-negatives 64 \
  --device cuda:0 \
  --log-interval 20 \
  > "$stage1_log" 2>&1 &
stage1_pid=$!
wait $stage1_pid

nohup python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name tgn_mean_ode_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir" \
  > "$stage2_log" 2>&1 &
stage2_pid=$!
wait $stage2_pid

echo "TGN mean+ODE experiment launched. Logs: $stage1_log , $stage2_log"

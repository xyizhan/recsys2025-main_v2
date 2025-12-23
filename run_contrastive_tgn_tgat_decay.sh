#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_tgn_tgat_decay"
score_dir="../scores_out/tgn_tgat_decay"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_tgn_tgat_decay_${timestamp}.log"

echo "===== Stage 1: 生成 TGN Embedding (TGAT 邻域 + Decay) ====="
echo "Logs: $log_file"
nohup bash <<EOF > "$log_file" 2>&1 &
set -euo pipefail
echo "[Stage 1] Starting embedding generation..."
python -m methods.contrastive_tgn.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 256 \
  --item-bucket-count 100000 \
  --neighbor-agg tgat \
  --neighbor-k 8 \
  --time-update decay \
  --decay-rate 0.5 \
  --num-negatives 64 \
  --device cuda:0 \
  --log-interval 20

echo "[Stage 2] Starting downstream training..."
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name tgn_tgat_decay_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir"
EOF

echo "TGN TGAT+decay experiment launched. Logs: $log_file"

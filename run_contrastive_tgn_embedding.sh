#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/contrastive_tgn_${timestamp}.log"

nohup python -m methods.contrastive_tgn.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_tgn_v1 \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 256 \
  --item-bucket-count 100000 \
  --num-negatives 64 \
  --device cuda:0 \
  --log-interval 20 \
  > "$log_file" 2>&1 &

echo "Launched TGN embedding job. Logs: $log_file"

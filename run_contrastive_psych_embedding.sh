#!/usr/bin/env bash
set -euo pipefail
timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/contrastive_psych_${timestamp}.log"
nohup python -m methods.contrastive_transformer_psych.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_psych_v1 \
  --epochs 1 \
  --batch-size 512 \
  --embedding-dim 512 \
  --d-model 256 \
  --behavioral-latent-dim 128 \
  --psych-latent-dim 160 \
  --max-seq-len 256 \
  --device cuda:0 \
  --log-interval 20 \
  --contrastive-weight 1.0 \
  --behavior-recon-weight 1.0 \
  --behavior-kl-weight 0.1 \
  --psych-recon-weight 0.5 \
  --psych-kl-weight 0.1 \
  --hierarchy-weight 0.05 \
  --enable-mask-predict \
  --mask-loss-weight 0.5 \
  --mask-prob 0.15 \
  --enable-order-predict \
  --order-loss-weight 0.5 \
  > "$log_file" 2>&1 &
echo "Launched contrastive+Psych dual-VAE embedding job. Logs: $log_file"

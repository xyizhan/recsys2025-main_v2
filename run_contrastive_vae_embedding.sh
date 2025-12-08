#!/usr/bin/env bash
set -euo pipefail
timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/contrastive_vae_${timestamp}.log"
nohup python -m methods.contrastive_transformer_vae.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_vae_v1 \
  --epochs 1 \
  --batch-size 512 \
  --embedding-dim 512 \
  --d-model 256 \
  --latent-dim 128 \
  --max-seq-len 256 \
  --device cuda:0 \
  --log-interval 20 \
  --contrastive-weight 1.0 \
  --recon-weight 1.0 \
  --kl-weight 0.1 \
  > "$log_file" 2>&1 &
echo "Launched contrastive+VAE embedding job. Logs: $log_file"

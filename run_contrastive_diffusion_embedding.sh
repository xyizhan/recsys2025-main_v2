#!/usr/bin/env bash
set -euo pipefail
timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/contrastive_diffusion_${timestamp}.log"
nohup python -m methods.contrastive_transformer_diffusion.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_diffusion_v1 \
  --epochs 1 \
  --batch-size 512 \
  --embedding-dim 512 \
  --latent-dim 256 \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 8 \
  --max-seq-len 256 \
  --device cuda:0 \
  --log-interval 20 \
  --contrastive-weight 1.0 \
  --diffusion-weight 1.0 \
  --diffusion-timesteps 40 \
  --diffusion-beta-start 1e-4 \
  --diffusion-beta-end 0.02 \
  > "$log_file" 2>&1 &
echo "Launched contrastive+diffusion embedding job. Logs: $log_file"

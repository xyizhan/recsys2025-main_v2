#!/usr/bin/env bash
set -euo pipefail
timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/contrastive_transformer_handcraft_${timestamp}.log"
nohup python -m methods.contrastive_transformer_handcraft.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_handcraft_v1 \
  --epochs 1 \
  --batch-size 16 \
  --embedding-dim 512 \
  --d-model 256 \
  --behavioral-latent-dim 128 \
  --psych-latent-dim 160 \
  --max-seq-len 64 \
  --device mps \
  --log-interval 1 \
  --contrastive-weight 1.0 \
  --behavior-recon-weight 1.0 \
  --behavior-kl-weight 0.1 \
  --psych-recon-weight 0.5 \
  --psych-kl-weight 0.1 \
  --hierarchy-weight 0.05 \
  > "$log_file" 2>&1 &
echo "Launched contrastive+handcraft embedding job. Logs: $log_file"
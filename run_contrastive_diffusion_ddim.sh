#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_transformer_diffusion_ddim"
score_dir="../scores_out/diffusion_ddim"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_diffusion_ddim_${timestamp}.log"

echo "===== Stage 1: 生成 Diffusion 对比学习 Embedding (DDIM) ====="
echo "Logs: $log_file"
nohup bash <<EOF > "$log_file" 2>&1 &
set -euo pipefail
echo "[Stage 1] Starting diffusion embedding generation..."
python -m methods.contrastive_transformer_diffusion.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 512 \
  --latent-dim 256 \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 8 \
  --ffn-mult 2.0 \
  --dropout 0.1 \
  --temperature 0.2 \
  --lr 1e-3 \
  --device cuda:0 \
  --log-interval 20 \
  --diffusion-timesteps 50 \
  --diffusion-beta-start 1e-4 \
  --diffusion-beta-end 0.02 \
  --diffusion-hidden-dim 512 \
  --diffusion-time-embed-dim 64 \
  --contrastive-weight 1.0 \
  --diffusion-weight 1.0 \
  --sampling-method ddim \
  --sampling-steps 15

echo "[Stage 2] Starting downstream training..."
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name diffusion_ddim_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir"
EOF

echo "Diffusion+contrastive DDIM experiment launched. Logs: $log_file"

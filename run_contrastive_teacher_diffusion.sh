#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_teacher_diffusion"
score_dir="../scores_out/teacher_diffusion"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_teacher_diffusion_${timestamp}.log"

echo "===== Stage 1: 生成 Conditional Diffusion 统一表征 ====="
echo "Logs: $log_file"
nohup bash <<EOF > "$log_file" 2>&1 &
set -euo pipefail
echo "[Stage 1] Starting teacher-student diffusion embedding generation..."
python -m methods.contrastive_transformer_teacher_diffusion.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 512 \
  --latent-dim 512 \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 8 \
  --dropout 0.1 \
  --ffn-mult 2.0 \
  --temperature 0.2 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --device cuda:0 \
  --log-interval 20 \
  --contrastive-weight 1.0 \
  --diffusion-weight 1.0 \
  --diffusion-timesteps 100 \
  --diffusion-beta-start 1e-4 \
  --diffusion-beta-end 0.02 \
  --diffusion-hidden-dim 512 \
  --diffusion-time-embed-dim 64 \
  --ema-momentum 0.995 \
  --cfg-drop-prob 0.1 \
  --cfg-guidance-weight 1.0 \
  --sampling-method ddim \
  --sampling-steps 20 \
  --num-samples 4 \
  --debug-logging

echo "[Stage 2] Starting downstream training..."
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name teacher_diffusion_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir"
EOF

echo "Teacher-conditioned diffusion experiment launched. Logs: $log_file"

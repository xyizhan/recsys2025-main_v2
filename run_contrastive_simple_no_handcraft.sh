#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_simple_no_handcraft"
score_dir="../scores_out/contrastive_simple_no_handcraft"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_simple_no_handcraft_${timestamp}.log"

echo "===== Stage 1: 单VAE去手工特征消融（仅生成表征） ====="
echo "Logs: $log_file"
nohup bash <<'EOF' > "$log_file" 2>&1 &
set -euo pipefail
echo "[Stage 1] Generating embeddings without handcrafted features..."
python -m methods.contrastive_transformer_simple.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --feat-path methods/contrastive_transformer_psych/user_stats_cache.pkl \
  --epochs 1 \
  --batch-size 512 \
  --max-seq-len 256 \
  --embedding-dim 512 \
  --d-model 256 \
  --latent-dim 160 \
  --n-layers 4 \
  --n-heads 8 \
  --dropout 0.1 \
  --ffn-mult 2.0 \
  --temperature 0.2 \
  --lr 1e-3 \
  --device cuda:0 \
  --log-interval 20 \
  --contrastive-weight 1.0 \
  --recon-weight 1.0 \
  --kl-weight 0.1 \
  --enable-mask-predict \
  --mask-loss-weight 0.3 \
  --mask-prob 0.15 \
  --enable-order-predict \
  --order-loss-weight 0.3 \
  --stats-version concat \
  --stats-dim 46 \
  --disable-handcrafted
EOF

echo "Handcrafted-feature ablation embedding generation launched. Logs: $log_file"
echo "Run ./run_contrastive_simple_no_handcraft_downstream.sh after embeddings are ready to start downstream training."

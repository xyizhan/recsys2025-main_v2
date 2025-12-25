#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_dualcls"
score_dir="../scores_out/contrastive_dualcls"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_dualcls_${timestamp}.log"

echo "===== Stage 1: 生成 Dual-CLS 双VAE 对比学习表征 ====="
echo "Logs: $log_file"
nohup bash <<EOF > "$log_file" 2>&1 &
set -euo pipefail
echo "[Stage 1] Starting dual-CLS embedding generation..."
python -m methods.contrastive_transformer_dualcls.create_embeddings \\
  --data-dir ../autodl-tmp/recsys_new \\
  --embeddings-dir "$embeddings_dir" \\
  --feat-path methods/contrastive_transformer_psych/user_stats_cache.pkl \\
  --epochs 1 \\
  --batch-size 512 \\
  --max-seq-len 256 \\
  --embedding-dim 512 \\
  --d-model 256 \\
  --n-layers 4 \\
  --n-heads 8 \\
  --dropout 0.1 \\
  --ffn-mult 2.0 \\
  --shallow-latent-dim 96 \\
  --deep-latent-dim 160 \\
  --short-window 48 \\
  --temperature 0.2 \\
  --lr 1e-3 \\
  --device cuda:0 \\
  --log-interval 20 \\
  --contrastive-weight 1.0 \\
  --shallow-recon-weight 1.0 \\
  --shallow-kl-weight 0.2 \\
  --deep-recon-weight 0.5 \\
  --deep-kl-weight 0.05 \\
  --structure-weight 0.05 \\
  --enable-mask-predict \\
  --mask-loss-weight 0.3 \\
  --mask-prob 0.15 \\
  --enable-order-predict \\
  --order-loss-weight 0.3 \\
  --stats-dim 46

echo "[Stage 2] Starting downstream training..."
python -m training_pipeline.train \\
  --data-dir ../autodl-tmp/recsys_new \\
  --embeddings-dir "$embeddings_dir" \\
  --tasks churn propensity_category propensity_sku conversion \\
  --log-name dualcls_gpu \\
  --num-workers 10 \\
  --accelerator gpu \\
  --devices 0 \\
  --score-dir "$score_dir"
EOF

echo "Dual-CLS embedding experiment launched. Logs: $log_file"

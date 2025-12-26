#!/usr/bin/env bash
set -euo pipefail

embeddings_dir="../autodl-tmp/embeddings_out/contrastive_simple_mlp"
score_dir="../scores_out/contrastive_simple_mlp"
mkdir -p "$score_dir"

echo "===== Stage 2: MLP 头消融下游训练 ====="
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name simple_mlp_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir" \
  --hidden-logging-mode

echo "Stage 2 finished. Scores stored in $score_dir"

#!/usr/bin/env bash
set -euo pipefail

timestamp=$(date +%m%d_%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
embeddings_dir="../autodl-tmp/embeddings_out/contrastive_simple_no_handcraft"
score_dir="../scores_out/contrastive_simple_no_handcraft"
mkdir -p "$score_dir"
log_file="$log_dir/contrastive_simple_no_handcraft_downstream_${timestamp}.log"

echo "===== Stage 2: 去手工特征消融下游训练 ====="
echo "Logs: $log_file"
nohup bash <<'EOF' > "$log_file" 2>&1 &
set -euo pipefail
python -m training_pipeline.train \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir "$embeddings_dir" \
  --tasks churn propensity_category propensity_sku conversion \
  --log-name simple_no_handcraft_gpu \
  --num-workers 10 \
  --accelerator gpu \
  --devices 0 \
  --score-dir "$score_dir" \
  --hidden-logging-mode
EOF

echo "Stage 2 launched in background. Scores will be stored in $score_dir once finished."

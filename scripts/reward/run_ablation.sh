#!/usr/bin/env bash
# ============================================================================
# Shortcut ablation: does removing the source signature fix within-model accuracy?
#
#   arm A  ablation_within.jsonl        653 pairs, 100% within-model  (treatment)
#   arm B  ablation_mixed_control.jsonl 653 pairs,  32% within-model  (control)
#
# Both arms get identical hyperparameters, so any difference is attributable to
# data composition and not to size or tuning. Anti-overfitting changes vs the
# original run (which hit train acc 1.00 while eval loss rose every epoch):
#   lr 1e-5 -> 5e-6,  weight-decay 1e-6 -> 1e-2,  regularization 1e-3 -> 1e-2
# Every epoch is checkpointed; selection is on eval LOSS, not accuracy.
#
# Usage:  bash scripts/reward/run_ablation.sh
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"
EPOCHS="${EPOCHS:-3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
UID_="$(id -u)"; GID_="$(id -g)"

dock () {  # dock "<shell script>"
  docker run --rm --gpus all --shm-size=10g \
    --entrypoint bash \
    -v "${REPO_DIR}:/data_workspace" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /data_workspace "${IMAGE}" -c "$1"
}

run_arm () {
  local arm="$1" data="$2"
  local out="reward_output/ablation_${arm}_${STAMP}"
  echo "=== arm ${arm}: ${data} -> ${out} ==="
  dock "set -euo pipefail
    python3 scripts/reward/train_reward_model.py \
      --dataset-path ${data} \
      --eval-dataset-path datasets/reward/reward_eval_byprompt.jsonl \
      --output-dir ${out} \
      --epochs ${EPOCHS} \
      --learning-rate 5e-6 \
      --weight-decay 1e-2 \
      --regularization 1e-2 \
      --batch-size 1 --gradient-accumulation-steps 32 \
      --max-length 4096 --pooling last-token \
      --gradient-checkpointing --adafactor \
      --load-in-half --device-map --save-each-epoch
    chown -R ${UID_}:${GID_} ${out}"

  # score every epoch checkpoint; selection happens in compare_ablation.py
  for ck in "${REPO_DIR}/${out}"/epoch*; do
    [ -d "${ck}" ] || continue
    local ep; ep="$(basename "${ck}")"
    dock "python3 scripts/reward/eval_reward.py \
      --checkpoint ${out}/${ep} \
      --eval-file datasets/reward/reward_eval_byprompt.jsonl \
      --output results/reward/ablation_${arm}_${ep}.json
    chown ${UID_}:${GID_} results/reward/ablation_${arm}_${ep}.json"
  done
}

run_arm within  datasets/reward/ablation_within.jsonl
run_arm control datasets/reward/ablation_mixed_control.jsonl

echo
echo "=== done; run: python3 scripts/reward/compare_ablation.py ==="

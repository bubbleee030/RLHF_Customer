#!/usr/bin/env bash
# ============================================================================
# Re-train the Ministral-3-3B-Instruct cost model on GPU, in Docker, WITH the
# fine-tuned backbone saved (fixes the May-18 "backbone not saved" data loss).
#
# Reproducible GPU path:
#   - image  : cost-model-trainer:v2  (pytorch 2.4.0 + CUDA 12.1, transformers 5.7)
#   - GPUs   : 2x Tesla V100 via --gpus all + device_map="auto"
#   - cache  : host ~/.cache/huggingface mounted read-write, offline mode
#
# Prereqs (see docs/2026-06-08_runbook.md):
#   - `nvidia-smi` works on the host (driver module loaded)
#   - `docker run --gpus all ...` can see the GPUs
#
# Hyperparameters are IDENTICAL to the lost May-18 run except max_length (256 -> 512).
#
# Env overrides (used by the smoke test):
#   DATASET, MAX_LENGTH, EPOCHS, RUN_DIR, EXTRA_FLAGS
# Usage:
#   bash scripts/retrain_ministral3b_docker.sh                  # real run
#   nohup bash scripts/retrain_ministral3b_docker.sh > /tmp/retrain.log 2>&1 &
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"
MODEL="${MODEL:-mistralai/Ministral-3-3B-Instruct-2512}"

DATASET="${DATASET:-datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

LORA_R="${LORA_R:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

PEFT_SETUP=""
if [[ "${LORA_R}" -gt 0 ]]; then
    PEFT_SETUP="pip install --quiet peft &&"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# RUN_DIR is relative to /data_workspace (== REPO_DIR) so it resolves both in and out of the container.
RUN_DIR="${RUN_DIR:-cost_output/run_ministral_3b_instruct_${TIMESTAMP}}"

echo "========================================="
echo "Ministral-3-3B-Instruct re-train (Docker/GPU)"
echo "Image      : ${IMAGE}"
echo "Model      : ${MODEL}"
echo "Dataset    : ${DATASET}"
echo "Run dir    : ${REPO_DIR}/${RUN_DIR}"
echo "Max length : ${MAX_LENGTH}"
echo "Batch x acc: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "Epochs     : ${EPOCHS}   LR: ${LR}   Seed: ${SEED}"
echo "Extra      : ${EXTRA_FLAGS:-<none>}"
echo "LoRA       : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "Started    : $(date)"
echo "========================================="

docker run --rm --gpus all --shm-size=10g \
    --entrypoint bash \
    -v "${REPO_DIR}:/data_workspace" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -w /data_workspace \
    "${IMAGE}" -c "
set -euo pipefail
${PEFT_SETUP} python3 scripts/trainer.py \
    --model-name-or-path '${MODEL}' \
    --dataset-path '${DATASET}' \
    --output-dir '${RUN_DIR}' \
    --max-length ${MAX_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --learning-rate ${LR} \
    --weight-decay 1e-6 \
    --regularization 0.001 \
    --epochs ${EPOCHS} \
    --warmup-ratio 0.1 \
    --eval-split-ratio 0.1 \
    --log-steps 10 \
    --seed ${SEED} \
    --pooling last-token \
    --loss-type sequence-wise \
    --load-in-half \
    --device-map \
    --save-eval-predictions \
    --save-backbone \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    ${EXTRA_FLAGS}
"

echo "========================================="
echo "Finished : $(date)"
echo "Run dir  : ${REPO_DIR}/${RUN_DIR}"
echo "========================================="

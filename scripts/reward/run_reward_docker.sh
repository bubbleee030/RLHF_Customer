#!/usr/bin/env bash
# ============================================================================
# Train the Ministral-3-3B-Instruct helpfulness Reward Model on GPU in Docker.
#
#   image  : cost-model-trainer:v2  (transformers 5.7 / torch 2.4, Mistral3 OK)
#   GPUs   : 2x Tesla V100 via --gpus all + device_map="auto"
#   loss   : pure Bradley-Terry (scripts/reward/train_reward_model.py)
#
# Default = by-prompt holdout (honest headline). Override SPLIT=bypair for the
# cost-model-comparable run.
#
# Usage:
#   bash scripts/reward/run_reward_docker.sh                 # by-prompt run
#   SPLIT=bypair bash scripts/reward/run_reward_docker.sh    # by-pair run
#   SMOKE=1 bash scripts/reward/run_reward_docker.sh         # tiny smoke test
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"
MODEL="${MODEL:-mistralai/Ministral-3-3B-Instruct-2512}"

SPLIT="${SPLIT:-byprompt}"
TRAIN="${TRAIN:-datasets/reward/reward_train_${SPLIT}.jsonl}"
EVAL="${EVAL:-datasets/reward/reward_eval_${SPLIT}.jsonl}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-42}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
REGULARIZATION="${REGULARIZATION:-0.001}"
LORA_R="${LORA_R:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

PEFT_SETUP=""
if [[ "${LORA_R}" -gt 0 ]]; then
    PEFT_SETUP="pip install --quiet peft &&"
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    TRAIN="datasets/reward/_smoke_train.jsonl"
    EVAL="datasets/reward/_smoke_eval.jsonl"
    MAX_LENGTH=256; GRAD_ACCUM=2; EPOCHS=1
    RUN_DIR="reward_output/_smoke_$(date +%Y%m%d_%H%M%S)"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-reward_output/run_reward_${SPLIT}_${TIMESTAMP}}"

echo "========================================="
echo "Reward Model training (Docker/GPU)"
echo "Image    : ${IMAGE}"
echo "Model    : ${MODEL}"
echo "Split    : ${SPLIT}"
echo "Train    : ${TRAIN}"
echo "Eval     : ${EVAL}"
echo "Run dir  : ${REPO_DIR}/${RUN_DIR}"
echo "MaxLen   : ${MAX_LENGTH}   Batch x acc: ${BATCH_SIZE} x ${GRAD_ACCUM}"
echo "Epochs   : ${EPOCHS}   LR: ${LR}   Seed: ${SEED}"
echo "LoRA     : r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "EarlyStop: patience=${EARLY_STOPPING_PATIENCE} (eval loss)"
echo "Started  : $(date)"
echo "========================================="

docker run --rm --gpus all --shm-size=10g \
    --entrypoint bash \
    -v "${REPO_DIR}:/data_workspace" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -w /data_workspace \
    "${IMAGE}" -c "
set -euo pipefail
${PEFT_SETUP} python3 scripts/reward/train_reward_model.py \
    --model-name-or-path '${MODEL}' \
    --dataset-path '${TRAIN}' \
    --eval-dataset-path '${EVAL}' \
    --output-dir '${RUN_DIR}' \
    --max-length ${MAX_LENGTH} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --learning-rate ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --regularization ${REGULARIZATION} \
    --epochs ${EPOCHS} \
    --warmup-ratio 0.1 \
    --log-steps 10 \
    --seed ${SEED} \
    --pooling last-token \
    --load-in-half \
    --device-map \
    --save-each-epoch \
    --save-backbone \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --early-stopping-patience ${EARLY_STOPPING_PATIENCE} ${EXTRA_FLAGS}
chown -R ${HOST_UID:-1000}:${HOST_GID:-1000} '${RUN_DIR}'
"

echo "========================================="
echo "Finished : $(date)"
echo "Run dir  : ${REPO_DIR}/${RUN_DIR}"
echo "========================================="

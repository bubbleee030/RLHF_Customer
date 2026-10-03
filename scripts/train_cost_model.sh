#!/usr/bin/env bash
# Cost Model Training Entrypoint (v2 — PKU 3-term loss)
# Usage: ./train_cost_model.sh [MODEL_NAME_OR_PATH] [OUTPUT_DIR] [DATASET_PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME_OR_PATH=${1:-"microsoft/deberta-v3-large"}
OUTPUT_DIR=${2:-"cost_output/run_deberta_v2_$(date +%Y%m%d_%H%M%S)"}
DATA_PATH=${3:-"datasets/cost_dataset_for_safe_rlhf_clean.jsonl"}

MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
REGULARIZATION=${REGULARIZATION:-0.001}
EPOCHS=${EPOCHS:-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
EVAL_SPLIT=${EVAL_SPLIT:-0.1}
LOG_STEPS=${LOG_STEPS:-10}
SEED=${SEED:-42}
POOLING=${POOLING:-mean}
LOSS_TYPE=${LOSS_TYPE:-sequence-wise}
NORMALIZE_SCORE=${NORMALIZE_SCORE:-0}
CONCAT_FORWARD=${CONCAT_FORWARD:-0}
NORMALIZER_MOMENTUM=${NORMALIZER_MOMENTUM:-0.9}
PRECISION=${PRECISION:-"--fp16"}

echo "========================================="
echo "Cost Model Training v2 (PKU 3-term loss)"
echo "========================================="
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Dataset: ${DATA_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Max Length: ${MAX_LENGTH}"
echo "Batch Size: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Regularization: ${REGULARIZATION}"
echo "Epochs: ${EPOCHS}"
echo "Pooling: ${POOLING}"
echo "Loss Type: ${LOSS_TYPE}"
echo "Normalize Score: ${NORMALIZE_SCORE}"
echo "Concat Forward: ${CONCAT_FORWARD}"
echo "Precision: ${PRECISION}"
echo "========================================="

EXTRA_FLAGS=(
    --pooling "${POOLING}"
    --loss-type "${LOSS_TYPE}"
    --normalizer-momentum "${NORMALIZER_MOMENTUM}"
)

if [ "${NORMALIZE_SCORE}" = "1" ]; then
    EXTRA_FLAGS+=(--normalize-score-during-training)
fi

if [ "${CONCAT_FORWARD}" = "1" ]; then
    EXTRA_FLAGS+=(--concat-forward)
fi

python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --dataset-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --regularization "${REGULARIZATION}" \
    --epochs "${EPOCHS}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --eval-split-ratio "${EVAL_SPLIT}" \
    --log-steps "${LOG_STEPS}" \
    --seed "${SEED}" \
    "${EXTRA_FLAGS[@]}" \
    ${PRECISION}

echo "Cost model training complete."

#!/usr/bin/env bash
# Ministral Cost Model Training — Instruct & Base variants
# Uses the same PKU 3-term loss, adapted for decoder architecture.
#
# Key differences from DeBERTa:
#   - --pooling last-token  (decoder models use last non-pad token as sequence repr)
#   - --load-in-half        (load backbone in FP16 to fit 3-8B params in V100 VRAM)
#   - --batch-size 2        (larger model footprint; effective batch stays 32 via accum)
#   - --gradient-accumulation-steps 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="${REPO_DIR}/datasets/cost_dataset_for_safe_rlhf_clean.jsonl"

MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
REGULARIZATION=${REGULARIZATION:-0.001}
EPOCHS=${EPOCHS:-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
EVAL_SPLIT=${EVAL_SPLIT:-0.1}
LOG_STEPS=${LOG_STEPS:-10}
SEED=${SEED:-42}

run_training() {
    local model_name="$1"
    local run_tag="$2"
    local output_dir="${REPO_DIR}/cost_output/${run_tag}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "========================================="
    echo "Ministral Cost Model Training"
    echo "========================================="
    echo "Model    : ${model_name}"
    echo "Run tag  : ${run_tag}"
    echo "Output   : ${output_dir}"
    echo "Dataset  : ${DATA_PATH}"
    echo "Pooling  : last-token"
    echo "Precision: load-in-half (FP16 native)"
    echo "Batch    : ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM)) effective"
    echo "LR       : ${LEARNING_RATE}"
    echo "Epochs   : ${EPOCHS}"
    echo "========================================="

    python3 "${SCRIPT_DIR}/trainer.py" \
        --model-name-or-path "${model_name}" \
        --dataset-path "${DATA_PATH}" \
        --output-dir "${output_dir}" \
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
        --pooling last-token \
        --loss-type sequence-wise \
        --load-in-half

    echo ""
    echo "Run complete: ${output_dir}"
    echo "${output_dir}" >> "${REPO_DIR}/cost_output/ministral_run_dirs.txt"
}

# ── Run 1: Instruct model ──────────────────────────────────────────────────
run_training "mistralai/Ministral-3-8B-Instruct-2512" "run_ministral_instruct"

# ── Run 2: Base model ──────────────────────────────────────────────────────
run_training "mistralai/Ministral-3-8B-Base-2512" "run_ministral_base"

echo ""
echo "All Ministral runs complete."
echo "Run directories logged to: ${REPO_DIR}/cost_output/ministral_run_dirs.txt"

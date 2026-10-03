#!/usr/bin/env bash
# Full pipeline: 8B Instruct + 8B Base (Adafactor) + 3B Instruct (AdamW)
# → plots → MD report.
#
# 3B run uses AdamW (not Adafactor). If it OOMs, the pipeline prints a clear
# message and continues to generate plots/report for whatever completed.
#
# Usage: bash start_training.sh   (or direct: bash scripts/run_all_pipeline.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_PATH="${REPO_DIR}/datasets/cost_dataset_for_safe_rlhf_clean.jsonl"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

INSTRUCT_8B_MODEL="mistralai/Ministral-3-8B-Instruct-2512"
BASE_8B_MODEL="mistralai/Ministral-3-8B-Base-2512"
INSTRUCT_3B_MODEL="mistralai/Ministral-3-3B-Instruct-2512"

INSTRUCT_8B_DIR="${REPO_DIR}/cost_output/run_ministral_8b_instruct_${TIMESTAMP}"
BASE_8B_DIR="${REPO_DIR}/cost_output/run_ministral_8b_base_${TIMESTAMP}"
INSTRUCT_3B_DIR="${REPO_DIR}/cost_output/run_ministral_3b_instruct_${TIMESTAMP}"

PLOT_DIR_8B_INSTRUCT="${REPO_DIR}/outputs/plots/ministral_8b_instruct"
PLOT_DIR_8B_BASE="${REPO_DIR}/outputs/plots/ministral_8b_base"
PLOT_DIR_3B_INSTRUCT="${REPO_DIR}/outputs/plots/ministral_3b_instruct"
REPORT_OUT="${REPO_DIR}/outputs/reports/ministral_report_${TIMESTAMP}.md"

MAX_LENGTH=${MAX_LENGTH:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
REGULARIZATION=${REGULARIZATION:-0.001}
EPOCHS=${EPOCHS:-3}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
EVAL_SPLIT=${EVAL_SPLIT:-0.1}
LOG_STEPS=${LOG_STEPS:-10}
SEED=${SEED:-42}

MINISTRAL_3B_FAILED=0

banner() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

common_args() {
    echo "--dataset-path ${DATA_PATH} \
--max-length ${MAX_LENGTH} \
--batch-size ${BATCH_SIZE} \
--gradient-accumulation-steps ${GRAD_ACCUM} \
--learning-rate ${LEARNING_RATE} \
--weight-decay ${WEIGHT_DECAY} \
--regularization ${REGULARIZATION} \
--epochs ${EPOCHS} \
--warmup-ratio ${WARMUP_RATIO} \
--eval-split-ratio ${EVAL_SPLIT} \
--log-steps ${LOG_STEPS} \
--seed ${SEED} \
--pooling last-token \
--loss-type sequence-wise \
--load-in-half \
--device-map \
--save-eval-predictions"
}

# ── Phase 1: 8B Instruct (Adafactor) ──────────────────────────────────────
banner "Phase 1/5: Training Ministral 8B Instruct (Adafactor)"
echo "Model  : ${INSTRUCT_8B_MODEL}"
echo "Output : ${INSTRUCT_8B_DIR}"
echo "Time   : $(date)"

python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${INSTRUCT_8B_MODEL}" \
    --output-dir "${INSTRUCT_8B_DIR}" \
    $(common_args) \
    --adafactor

echo "8B Instruct done at $(date)"

# ── Phase 2: 8B Base (Adafactor) ──────────────────────────────────────────
banner "Phase 2/5: Training Ministral 8B Base (Adafactor)"
echo "Model  : ${BASE_8B_MODEL}"
echo "Output : ${BASE_8B_DIR}"
echo "Time   : $(date)"

python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${BASE_8B_MODEL}" \
    --output-dir "${BASE_8B_DIR}" \
    $(common_args) \
    --adafactor

echo "8B Base done at $(date)"

# ── Phase 3: 3B Instruct (AdamW — no --adafactor) ─────────────────────────
banner "Phase 3/5: Training Ministral 3B Instruct (AdamW)"
echo "Model  : ${INSTRUCT_3B_MODEL}"
echo "Output : ${INSTRUCT_3B_DIR}"
echo "NOTE   : Using AdamW (not Adafactor). OOM => pipeline will report and continue."
echo "Time   : $(date)"

set +e
python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${INSTRUCT_3B_MODEL}" \
    --output-dir "${INSTRUCT_3B_DIR}" \
    $(common_args)
EXIT_CODE=$?
set -e

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo ""
    echo "====================================================="
    echo "  3B TRAINING FAILED (exit code: ${EXIT_CODE})"
    echo "  Likely cause: CUDA OOM with AdamW on 2×V100."
    echo "  ACTION REQUIRED: Move project to 4-GPU VM and retry."
    echo "  Pipeline will continue to generate plots/report"
    echo "  for 8B models that completed successfully."
    echo "====================================================="
    MINISTRAL_3B_FAILED=1
fi

echo "3B Instruct phase done at $(date) (failed=${MINISTRAL_3B_FAILED})"

# ── Phase 4: Generate plots ────────────────────────────────────────────────
banner "Phase 4/5: Generating plots"

mkdir -p "${PLOT_DIR_8B_INSTRUCT}" "${PLOT_DIR_8B_BASE}"
python3 "${SCRIPT_DIR}/visualize_training.py" \
    --output-dir "${INSTRUCT_8B_DIR}" \
    --save-dir "${PLOT_DIR_8B_INSTRUCT}"

python3 "${SCRIPT_DIR}/visualize_training.py" \
    --output-dir "${BASE_8B_DIR}" \
    --save-dir "${PLOT_DIR_8B_BASE}"

if [ "${MINISTRAL_3B_FAILED}" -eq 0 ]; then
    mkdir -p "${PLOT_DIR_3B_INSTRUCT}"
    python3 "${SCRIPT_DIR}/visualize_training.py" \
        --output-dir "${INSTRUCT_3B_DIR}" \
        --save-dir "${PLOT_DIR_3B_INSTRUCT}"
fi

echo "Plots done at $(date)"

# ── Phase 5: Generate report ───────────────────────────────────────────────
banner "Phase 5/5: Generating report"

mkdir -p "$(dirname "${REPORT_OUT}")"

REPORT_ARGS="--instruct-8b-dir ${INSTRUCT_8B_DIR} \
             --base-8b-dir ${BASE_8B_DIR} \
             --plot-dir-8b-instruct ${PLOT_DIR_8B_INSTRUCT} \
             --plot-dir-8b-base ${PLOT_DIR_8B_BASE} \
             --output ${REPORT_OUT}"

if [ "${MINISTRAL_3B_FAILED}" -eq 0 ]; then
    REPORT_ARGS="${REPORT_ARGS} \
        --instruct-3b-dir ${INSTRUCT_3B_DIR} \
        --plot-dir-3b-instruct ${PLOT_DIR_3B_INSTRUCT}"
fi

python3 "${SCRIPT_DIR}/generate_ministral_report.py" ${REPORT_ARGS}

echo ""
echo "========================================="
echo "PIPELINE COMPLETE at $(date)"
echo "Report: ${REPORT_OUT}"
echo "8B Instruct: ${INSTRUCT_8B_DIR}"
echo "8B Base:     ${BASE_8B_DIR}"
if [ "${MINISTRAL_3B_FAILED}" -eq 0 ]; then
    echo "3B Instruct: ${INSTRUCT_3B_DIR}"
else
    echo "3B Instruct: FAILED (OOM) — move to 4-GPU VM"
fi
echo "========================================="

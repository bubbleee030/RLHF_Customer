#!/usr/bin/env bash
# Full pipeline: train both Ministral models → plots → MD report.
# Designed to run unattended via nohup. All output goes to LOGFILE.
# Usage: nohup bash scripts/run_ministral_pipeline.sh > /tmp/ministral_pipeline.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_PATH="${REPO_DIR}/datasets/cost_dataset_for_safe_rlhf_clean.jsonl"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

INSTRUCT_MODEL="mistralai/Ministral-3-8B-Instruct-2512"
BASE_MODEL="mistralai/Ministral-3-8B-Base-2512"

INSTRUCT_RUN_DIR="${REPO_DIR}/cost_output/run_ministral_instruct_${TIMESTAMP}"
BASE_RUN_DIR="${REPO_DIR}/cost_output/run_ministral_base_${TIMESTAMP}"

PLOT_DIR_INSTRUCT="${REPO_DIR}/outputs/plots/ministral_instruct"
PLOT_DIR_BASE="${REPO_DIR}/outputs/plots/ministral_base"
REPORT_OUT="${REPO_DIR}/outputs/reports/ministral_report_${TIMESTAMP}.md"

MAX_LENGTH=${MAX_LENGTH:-512}
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

banner() { echo ""; echo "========================================"; echo "$1"; echo "========================================"; echo ""; }

# ── Phase 1: Train Instruct model ─────────────────────────────────────────
banner "Phase 1/4: Training Ministral Instruct"
echo "Model  : ${INSTRUCT_MODEL}"
echo "Output : ${INSTRUCT_RUN_DIR}"
echo "Time   : $(date)"

python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${INSTRUCT_MODEL}" \
    --dataset-path "${DATA_PATH}" \
    --output-dir "${INSTRUCT_RUN_DIR}" \
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
    --load-in-half \
    --device-map \
    --adafactor

echo "Instruct training done at $(date)"

# ── Phase 2: Train Base model ──────────────────────────────────────────────
banner "Phase 2/4: Training Ministral Base"
echo "Model  : ${BASE_MODEL}"
echo "Output : ${BASE_RUN_DIR}"
echo "Time   : $(date)"

python3 "${SCRIPT_DIR}/trainer.py" \
    --model-name-or-path "${BASE_MODEL}" \
    --dataset-path "${DATA_PATH}" \
    --output-dir "${BASE_RUN_DIR}" \
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
    --load-in-half \
    --device-map \
    --adafactor

echo "Base training done at $(date)"

# ── Phase 3: Generate plots ────────────────────────────────────────────────
banner "Phase 3/4: Generating plots"

mkdir -p "${PLOT_DIR_INSTRUCT}" "${PLOT_DIR_BASE}"

python3 "${SCRIPT_DIR}/visualize_training.py" \
    --output-dir "${INSTRUCT_RUN_DIR}" \
    --save-dir "${PLOT_DIR_INSTRUCT}"

python3 "${SCRIPT_DIR}/visualize_training.py" \
    --output-dir "${BASE_RUN_DIR}" \
    --save-dir "${PLOT_DIR_BASE}"

echo "Plots done at $(date)"

# ── Phase 4: Generate report ───────────────────────────────────────────────
banner "Phase 4/4: Generating report"

mkdir -p "$(dirname "${REPORT_OUT}")"

python3 "${SCRIPT_DIR}/generate_ministral_report.py" \
    --instruct-dir "${INSTRUCT_RUN_DIR}" \
    --base-dir     "${BASE_RUN_DIR}" \
    --plot-dir-instruct "${PLOT_DIR_INSTRUCT}" \
    --plot-dir-base     "${PLOT_DIR_BASE}" \
    --output "${REPORT_OUT}"

echo ""
echo "========================================="
echo "PIPELINE COMPLETE at $(date)"
echo "Report: ${REPORT_OUT}"
echo "Instruct run: ${INSTRUCT_RUN_DIR}"
echo "Base run:     ${BASE_RUN_DIR}"
echo "========================================="

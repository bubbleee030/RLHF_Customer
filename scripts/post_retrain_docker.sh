#!/usr/bin/env bash
# ============================================================================
# Post-retrain steps for the Ministral-3-3B cost model, in Docker on GPU:
#   1. eval on the original eval set        -> results/eval_original_ministral3b.json
#   2. eval on the Gemma-4 holdout set      -> results/eval_holdout_ministral3b_gemma4.json
#   3. the >512 max-length experiment       -> results/experiment_maxlength.json
#
# Usage:
#   bash scripts/post_retrain_docker.sh <RUN_DIR>
#   # RUN_DIR is relative to the repo root, e.g. cost_output/run_ministral_3b_instruct_<ts>
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"
RUN_DIR="${1:?Usage: post_retrain_docker.sh <RUN_DIR (relative to repo)>}"

run_in_container() {
    docker run --rm --gpus all --entrypoint bash \
        -v "${REPO_DIR}:/data_workspace" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -w /data_workspace \
        "${IMAGE}" -c "$1"
}

echo "### 1/3 Eval on ORIGINAL eval set ###"
run_in_container "python3 scripts/eval_ministral.py \
    --run-dir '${RUN_DIR}' \
    --eval-file datasets/eval_dataset.jsonl \
    --output results/eval_original_ministral3b.json"

echo "### 2/3 Eval on GEMMA-4 HOLDOUT eval set ###"
run_in_container "python3 scripts/eval_ministral.py \
    --run-dir '${RUN_DIR}' \
    --eval-file datasets/eval_dataset_holdout_gemma4.jsonl \
    --output results/eval_holdout_ministral3b_gemma4.json"

echo "### 3/3 Max-length (>512) experiment ###"
run_in_container "python3 scripts/experiment_maxlength.py \
    --run-dir '${RUN_DIR}' \
    --eval-file datasets/eval_dataset.jsonl \
    --output results/experiment_maxlength.json"

echo "### Done. Fixing ownership of new files (docker writes as root) ###"
sudo chown -R "$(id -u):$(id -g)" "${REPO_DIR}/${RUN_DIR}" "${REPO_DIR}/results" 2>/dev/null || true
echo "All post-retrain steps complete."

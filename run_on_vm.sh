#!/usr/bin/env bash
# 這是給 VM 直接跑 Docker 用的腳本，取代原本國網的 job.sh (Slurm/Singularity)

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cost-model-trainer}"
WORKSPACE_DIR="$(pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-microsoft/deberta-v3-large}"
OUTPUT_DIR="${OUTPUT_DIR:-/data_workspace/cost_output/run_deberta_official_$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${DATASET_PATH:-/data_workspace/datasets/cost_dataset_for_safe_rlhf_clean.jsonl}"
MODEL_DIR="${WORKSPACE_DIR}/model"

echo "========================================="
echo "1. 檢查模型對應路徑"
echo "準備掛載的模型路徑: ${MODEL_DIR}"
echo "========================================="
mkdir -p ${MODEL_DIR}

echo "========================================="
echo "2. 建立 Docker Image"
echo "========================================="
# 只有第一次或 Dockerfile 有改時才需要 build
# docker build -f Dockerfile.v2 -t "${IMAGE_NAME}" .

echo "========================================="
echo "3. 啟動 Docker 容器開始訓練"
echo "========================================="
# 使用 --gpus all 啟用 GPU
# 使用 --shm-size=10g 避免 PyTorch 多卡訓練時 shared memory 不足的問題
docker run --gpus all --rm -it \
    --shm-size=10g \
    -e LOSS_TYPE="${LOSS_TYPE:-sequence-wise}" \
    -e CONCAT_FORWARD="${CONCAT_FORWARD:-0}" \
    -e NORMALIZE_SCORE="${NORMALIZE_SCORE:-0}" \
    -v "${MODEL_DIR}:/model" \
    -v "${WORKSPACE_DIR}:/data_workspace" \
    "${IMAGE_NAME}" \
    bash /data_workspace/scripts/train_cost_model.sh "${MODEL_NAME_OR_PATH}" "${OUTPUT_DIR}" "${DATASET_PATH}"

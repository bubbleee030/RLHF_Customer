#!/usr/bin/env bash
# Run a command inside the GPU training image with the repo + HF cache mounted.
# Usage: bash scripts/serve/run_in_docker.sh "python3 scripts/serve/smoke_scorers.py"
#        PORT_ARGS="-p 7860:7860" bash scripts/serve/run_in_docker.sh "..."
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"

RECOVERY_MOUNT_ARGS=()
if [[ -n "${RECOVERY_SHM_DIR:-}" ]]; then
    RECOVERY_SHM_DIR="$(python3 "${REPO_DIR}/scripts/serve/recovery_shm.py" \
        --allow-missing "${RECOVERY_SHM_DIR}")"
    mkdir -p -- "${RECOVERY_SHM_DIR}"
    RECOVERY_SHM_DIR="$(python3 "${REPO_DIR}/scripts/serve/recovery_shm.py" \
        "${RECOVERY_SHM_DIR}")"
    RECOVERY_MOUNT_ARGS=(-v "${RECOVERY_SHM_DIR}:/recovery")
fi

# shellcheck disable=SC2086
docker run --rm --gpus all --shm-size=10g \
    --entrypoint bash ${PORT_ARGS:-} \
    -v "${REPO_DIR}:/data_workspace" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${RECOVERY_MOUNT_ARGS[@]}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -w /data_workspace \
    "${IMAGE}" -c "$*"

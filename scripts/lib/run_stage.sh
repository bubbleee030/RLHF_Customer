#!/usr/bin/env bash
# ============================================================================
# Shared Docker invocation for every training/eval stage.
#
# WHY THIS EXISTS: cost-model-trainer:v2 sets ENTRYPOINT=/bin/bash with no CMD.
# Appending a command after the image name therefore produces "/bin/bash bash"
# or "/bin/bash python3 ...", and the second token is executed as a SCRIPT
# rather than a command — the "cannot execute binary file" error that has been
# rediscovered three times in this project. The fix is to always pass
# --entrypoint explicitly and hand the command to bash -c.
#
# Usage:
#   source scripts/lib/run_stage.sh
#   run_stage "python3 scripts/reward/train_reward_model.py --help"
#
# Env: IMAGE, HF_CACHE, REPO_DIR, GPUS, PORT_ARGS, EXTRA_DOCKER_ARGS, DRY_RUN
# ============================================================================

run_stage() {
    local command="$1"
    local repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    local hf_cache="${HF_CACHE:-${HOME}/.cache/huggingface}"
    local image="${IMAGE:-cost-model-trainer:v2}"
    local gpus="${GPUS:-all}"

    local args=(
        run --rm --gpus "${gpus}" --shm-size=10g
        --entrypoint /bin/bash
        -v "${repo_dir}:/data_workspace"
        -v "${hf_cache}:/root/.cache/huggingface"
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"
        -w /data_workspace
    )

    local -a port_args=()
    local -a extra_docker_args=()
    if [[ -n "${PORT_ARGS:-}" ]]; then
        read -r -a port_args <<< "${PORT_ARGS}"
        args+=("${port_args[@]}")
    fi
    if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
        read -r -a extra_docker_args <<< "${EXTRA_DOCKER_ARGS}"
        args+=("${extra_docker_args[@]}")
    fi

    args+=("${image}" -c "${command}")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'docker'
        printf ' %q' "${args[@]}"
        printf '\n'
        return 0
    fi

    docker "${args[@]}"
}

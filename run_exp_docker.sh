#!/usr/bin/env bash
# Experiment launcher for the PPO diagnosis work (2026-08-29).
#
# Differs from scripts/serve/run_in_docker.sh in one way: it additionally mounts
# the 2026-08-03 vm2 backup READ-ONLY at /backup, so the 113GB customer-8B actor
# and the RM/CM checkpoints can be used without copying them (vm3 has ~45GB free)
# and without any possibility of mutating the backup.
#
# Usage: bash run_exp_docker.sh "python3 scripts/ppo_lag/train_ppo_lag.py"
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Migrated 2026-09-01: vm3's /home/ubuntu/backups/ was NOT carried over (it was a
# duplicate of vm2's live tree). The 8B actor and the original RM came across with
# vm2, so /backup now resolves there. Kept overridable so a future host can point
# BACKUP elsewhere without editing this file.
BACKUP="${BACKUP:-/home/ubuntu/data/twcc-vm2/reward_model}"
if [ ! -d "$BACKUP" ]; then
  echo "run_exp_docker.sh: BACKUP does not exist: $BACKUP" >&2
  echo "  expected the vm2 tree holding model/costomer_model and reward_output/." >&2
  exit 1
fi
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
IMAGE="${IMAGE:-cost-model-trainer:v2}"

# Forward only the knobs train_ppo_lag.py actually reads, so a run's config is
# fully described by the env block in its launch command.
PASS_ENV=(
  DRY_RUN MAX_UPDATES EPOCHS KL_COEFF ACTOR_LR CRITIC_LR
  PROMPT_BATCH_SIZE MICRO_BATCH_SIZE MAX_NEW_TOKENS LORA_R LORA_ALPHA
  EARLY_STOP_KL SAVE_EVERY SAVE_EPOCHS SAVE_CRITICS SEED
  ACTOR_MODEL PROMPT_FORMAT RM_DIR CM_DIR PPO_PROMPT_FILE RUN_DIR
  GRADIENT_CHECKPOINTING
  NCHC_API_KEY NCHC_BASE_URL NCHC_MODEL
  COST_SOURCE POLICY_PROMPT_FILE
  CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
  PPO_RAW_ADAPTER PPO_POLICY_ADAPTER OUT N ACTOR
  MANIFEST POLICIES ZH BI THRESHOLD LAMBDA_MAX CKPT
)
ENV_ARGS=()
for v in "${PASS_ENV[@]}"; do
  if [[ -n "${!v:-}" ]]; then ENV_ARGS+=(-e "${v}=${!v}"); fi
done

# NOTE: image ENTRYPOINT is /bin/bash, so we must pass --entrypoint bash and -c,
# otherwise the command is appended and runs as `/bin/bash bash ...`.
docker run --rm --gpus all --shm-size=10g \
  --entrypoint bash \
  -v "${WS}:/data_workspace" \
  -v "${BACKUP}:/backup:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${ENV_ARGS[@]}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/data_workspace \
  -w /data_workspace \
  "${IMAGE}" -c "$*"

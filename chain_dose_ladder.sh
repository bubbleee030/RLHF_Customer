#!/usr/bin/env bash
# Adversarial-dose ladder -- the secondary experiment of
# results/PREREGISTRATION_replication_and_dose.md.
#
# Runs P and Q differ ONLY in the prompt pool, and that difference is worth
# +0.2557 [+0.141, +0.372]. Everything known about the shape of that curve is a
# single point at 296 adversarial prompts. This walks the dose while holding the
# recipe, RM, CM, threshold and step budget fixed.
#
# The customer half is CONSTANT at 296 at every rung -- cs_within_train.jsonl has
# exactly 296 unique prompts -- so dose and adversarial FRACTION move together
# (0.50 / 0.67 / 0.80 / 0.89). That confound is declared in the pre-registration
# along with the fixed-total sub-ladder that would disambiguate it. Do not read a
# dose response off this ladder as if volume were the only thing varying.
#
# One seed per rung. Rungs are independent, so run whatever subset GPU time
# allows and report a partial ladder as partial.
#
# Usage: bash chain_dose_ladder.sh [dose ...]     (default: all four)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DOSES=("$@")
[ ${#DOSES[@]} -eq 0 ] && DOSES=(296 592 1184 2368)

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}

launch() {
  local name="$1" prompts="$2" gpus="$3"
  env \
    ACTOR_MODEL=/backup/model/costomer_model \
    RM_DIR=reward_output/rmS_safety_aware_20260831/epoch1 \
    CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
    PROMPT_FORMAT=customer_inst \
    PPO_PROMPT_FILE="$prompts" \
    SEED=20260901 \
    LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
    MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
    SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="$gpus" \
    ACTOR_LR=2e-5 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=5.0 \
    RUN_DIR="ppo_output/$name" \
    bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'
}

for dose in "${DOSES[@]}"; do
  POOL="datasets/ppo/ladder_20260901/pool_adv${dose}.jsonl"
  if [ ! -f "$POOL" ]; then
    echo "$(date -Is) SKIP dose $dose -- missing $POOL" >&2
    continue
  fi
  NAME="runDose${dose}_20260901"
  echo "$(date -Is) waiting for a free GPU pair (dose $dose)"
  while ! GPUS=$(pick_pair); do sleep 120; done
  echo "$(date -Is) dose $dose on GPUs $GPUS -> ppo_output/$NAME"
  launch "$NAME" "$POOL" "$GPUS" > "logs/${NAME}.log" 2>&1
  echo "$(date -Is) dose $dose finished rc=$?"
done

echo "$(date -Is) dose ladder complete for: ${DOSES[*]}"

#!/usr/bin/env bash
# Run P replication -- seeds 2 and 3 of the pre-registered primary experiment.
#
# See results/PREREGISTRATION_replication_and_dose.md, written 2026-09-01 on a
# GPU-less host BEFORE any of this could run. The analysis is fixed there; this
# script only produces the adapters.
#
# Why this exists: the project's headline (Run P, comparison #1 point estimate
# above the policy prompt for the first time) rests on ONE seed, and this
# configuration is known to vary run to run -- Run L died at step 311 where Run J
# survived 400 steps on the same settings.
#
# Config is Run P's exactly (actor_lr 2e-5 / lambda_max 5.0 / kl_coeff 0.2 /
# threshold -4.427), the stable recipe: all six 2e-5 runs survived, while 4 of 6
# completed 1e-4 runs diverged on KL regardless of lambda_max.
#
# The prompt pool is REBUILT from source by scripts/augment/build_ppo_pool.py
# rather than reusing mixed_customer_adversarial_20260831.jsonl, so this is a
# replication over prompt sampling as well as over seed. Run P's exact 592
# prompts remain in ppo_output/runP_adversarial_20260831/prompts_used.json if an
# exact-pool replication is wanted instead.
#
# SEED is env-overridable as of 2026-09-01 (it was hardcoded to 42, which would
# have made this experiment impossible); the value lands in each run's
# config.json automatically.
#
# Usage: bash chain_replicate_runP.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

POOL="datasets/ppo/ladder_20260901/pool_adv296.jsonl"
[ -f "$POOL" ] || { echo "missing $POOL -- run scripts/augment/build_ppo_pool.py first" >&2; exit 1; }

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}

launch() {
  local name="$1" seed="$2" gpus="$3"
  env \
    ACTOR_MODEL=/backup/model/costomer_model \
    RM_DIR=reward_output/rmS_safety_aware_20260831/epoch1 \
    CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
    PROMPT_FORMAT=customer_inst \
    PPO_PROMPT_FILE="$POOL" \
    SEED="$seed" \
    LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
    MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
    SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="$gpus" \
    ACTOR_LR=2e-5 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=5.0 \
    RUN_DIR="ppo_output/$name" \
    bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'
}

echo "$(date -Is) waiting for a free GPU pair (seed 2)"
while ! G1=$(pick_pair); do sleep 120; done
launch runP2_seed20260902 20260902 "$G1" > logs/runP2_seed20260902.log 2>&1 &
P1=$!; echo "$(date -Is) Run P2 (seed 20260902) pid $P1 on GPUs $G1"

sleep 60
echo "$(date -Is) waiting for a free GPU pair (seed 3)"
while ! G2=$(pick_pair); do sleep 120; done
launch runP3_seed20260903 20260903 "$G2" > logs/runP3_seed20260903.log 2>&1 &
P2=$!; echo "$(date -Is) Run P3 (seed 20260903) pid $P2 on GPUs $G2"

wait $P1 $P2
echo "$(date -Is) replication training complete"
echo "Next: evaluate with the BASELINE ARMS COPIED, not regenerated --"
echo "  the bar drifts 3.6pt across judging runs and the pre-registration"
echo "  freezes it by reusing results/redteam_heldout_runP_20260831 baselines."

#!/usr/bin/env bash
# Run M -- both fixes together: the safety-aware RM AND an attainable threshold.
#
# The retrained RM reverses the incentive that made every prior run fight itself:
#   old RM  RM(unsafe) - RM(safe_refusal) = +2.121  -> violating PAID; safety needed
#                                                      lambda > 0.560, and a lambda
#                                                      that large diverged Runs H/K
#   new RM                                 = -2.274  -> refusing pays; safety wins
#                                                      at any lambda >= 0
# Helpfulness on the held-out cs_within_test is unchanged within noise (0.7701 vs
# 0.7915, McNemar p=0.314, prompt-clustered CI [-0.076,+0.032]).
#
# THRESHOLD -4.427 is the attainable value derived from the AUGMENTED CM (the one
# this run trains against) on real generations; -6.309 and -5.441 were both below
# its safe mean (-4.751), so lambda could never come off its cap. See chain_runL.
#
# RISK TO WATCH: an RM paying +2.274 to refuse can collapse the policy into
# refusing everything. That scores well on safe_outcome but fails the over-refusal
# and helpfulness checks, so the blinded report's over_refusal / safe_helpful
# columns decide whether this run is real. Do not read success off reward alone.
#
# Waits for Run L to claim its GPUs first so the two do not race for the same pair.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date -Is) waiting for Run L to claim its GPUs"
for i in $(seq 1 120); do
  [ -s ppo_output/runL_attainable_20260831/training_log.jsonl ] && break
  sleep 60
done
sleep 120

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) Run M on GPUs $GPUS"

exec env \
  ACTOR_MODEL=/backup/model/costomer_model \
  RM_DIR=reward_output/rmS_safety_aware_20260831/epoch1 \
  CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
  PROMPT_FORMAT=customer_inst \
  PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
  LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
  MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
  SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=1.0 \
  RUN_DIR=ppo_output/runM_newrm_attainable_20260831 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

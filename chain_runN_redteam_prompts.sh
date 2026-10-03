#!/usr/bin/env bash
# Run N -- the run the I/J evaluation actually pointed at.
#
# Run I is the first VALID bare-PPO measurement with a live constraint, and it is
# statistically tied with the policy-prompt bar overall (0.7891 vs 0.7969, paired
# delta -0.0424, 95% [-0.161,+0.073]) -- an enormous move from the -0.2321 that
# started this investigation. Per track it BEATS the policy prompt everywhere
# except one:
#
#   clean_policy_seeds  0.8571 vs 0.7857     eval_clean      0.8444 vs 0.7556
#   sealed_customer     0.8667 vs 0.8444     eval_all        0.9027 vs 0.8327
#   redteam             0.2857 vs 0.6667   <-- and base_raw is 0.2381
#
# On adversarial prompts PPO bought essentially NOTHING over the untrained model,
# while the policy prompt nearly triples it. That is the entire remaining gap, and
# it is a DISTRIBUTION problem, not the data-VOLUME problem originally hypothesised:
# every PPO run so far sampled from 296 ordinary customer prompts containing zero
# adversarial examples, so there was never a gradient teaching jailbreak refusal.
# The policy prompt generalises to attacks because it states rules; PPO cannot
# generalise to a distribution it never sampled.
#
# Fix: train on 296 customer + 296 adversarial prompts (A1/A2/A3, the pool built
# for the CM augmentation, from two generators). Verified ZERO overlap with the
# evaluation manifest -- and in particular with its 18 redteam prompts -- so this
# adds no leakage. Everything else is Run I/L's configuration.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "$(date -Is) waiting for a free GPU pair"
pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) Run N on GPUs $GPUS"

exec env \
  ACTOR_MODEL=/backup/model/costomer_model \
  RM_DIR=reward_output/rmS_safety_aware_20260831/epoch1 \
  CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
  PROMPT_FORMAT=customer_inst \
  PPO_PROMPT_FILE=datasets/ppo/mixed_customer_adversarial_20260831.jsonl \
  LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
  MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
  SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=1.0 \
  RUN_DIR=ppo_output/runN_redteam_prompts_20260831 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

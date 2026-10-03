#!/usr/bin/env bash
# Run H — the first configuration in which ALL THREE identified failures are
# addressed simultaneously:
#
#   1. CM detection      -> augmented CM (OOD unsafe recall 20.1% -> 63.3%)
#   2. Constraint inert  -> THRESHOLD -6.309 instead of 0.0
#                           With raw CM scores (violation +1.088, safe -6.698) a
#                           batch mean crosses 0.0 only at an ~86% violation
#                           rate, so lambda decayed in EVERY CM-based run ever
#                           done here, including the original 2026-08-03 one.
#                           -6.309 puts the crossover at a 5% violation rate.
#   3. Actor frozen      -> actor_lr 1e-4 (Run G moves 13x more than Run E)
#                           with kl_coeff kept at 0.2 (Run F diverged at 0.05)
#
# Each of those alone was sufficient to disable PPO-Lagrange, which is why no
# single fix has produced safer behaviour yet.
#
# Waits for Run E so it can take GPUs 0,1 without disturbing Run G on 2,3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

while true; do
  n=$(wc -l < ppo_output/runE_augcm_20260830/training_log.jsonl 2>/dev/null || echo 0)
  [ "$n" -ge 400 ] && { echo "$(date -Is) Run E reached $n"; break; }
  [ -d ppo_output/runE_augcm_20260830/final ] && { echo "$(date -Is) Run E final/ at $n"; break; }
  sleep 120
done
sleep 60   # let the container release GPU memory

echo "$(date -Is) launching Run H on GPUs 0,1"
exec env \
  ACTOR_MODEL=/backup/model/costomer_model \
  RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
  CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
  PROMPT_FORMAT=customer_inst \
  PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
  LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
  MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=50 SAVE_EPOCHS=0 \
  SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1 \
  ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-6.309 \
  RUN_DIR=ppo_output/runH_allfixed_20260830 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

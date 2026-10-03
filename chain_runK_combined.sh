#!/usr/bin/env bash
# Run K — the combination none of the nine runs tested.
#
# The two things that independently improved safe_outcome on the blinded
# 163-prompt manifest were:
#   * the policy prompt          base 0.6111 -> 0.7857   (and +PPO -> 0.8016, best measured)
#   * Run G's recipe             base 0.6111 -> 0.7266   (augmented CM + actor_lr 1e-4)
#
# Run D achieved the best number to date WITHOUT any of the fixes: it used the
# leaked-split CM, actor_lr 2e-5, and a constraint that could not engage. Run K
# is Run D's setup with everything since corrected:
#
#   policy prompt (zh)        <- as Run D; bilingual exceeds the ~1100-token
#                                training ceiling on a 32GB V100
#   augmented CM              <- OOD unsafe recall 20.1% -> 63.3%
#   THRESHOLD -6.309          <- constraint engages above a 5% violation rate
#                                instead of the ~86% that threshold 0.0 implied
#   LAMBDA_MAX 1.0            <- 1.79x the measured 0.560 crossover; 5.0 diverged
#                                Run H at step 111, 1.0 survived all 400 in Run J
#   actor_lr 1e-4             <- Run E (2e-5) stayed frozen; Run G moved and gained
#
# Batch 1 because the policy prefix costs ~750 tokens and batch 4 OOMs (Run D's
# lesson); 800 steps keeps episodes at 800, comparable to Run D's own budget.
#
# Compared against base_policy_zh, the baseline matching its training prompt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) Run K on GPUs $GPUS"

exec env \
  ACTOR_MODEL=/backup/model/costomer_model \
  RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
  CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
  PROMPT_FORMAT=customer_inst_policy \
  POLICY_PROMPT_FILE=configs/policy_eval/compiled/system_prompt_zh.txt \
  PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
  LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=128 \
  MAX_UPDATES=800 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
  SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-6.309 LAMBDA_MAX=1.0 \
  RUN_DIR=ppo_output/runK_combined_20260831 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

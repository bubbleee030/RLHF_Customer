#!/usr/bin/env bash
# Run D — policy-prompt + PPO. Waits for Run A to finish, then trains on GPUs 0,1.
#
# Run D differs from Run A in EXACTLY ONE respect: PROMPT_FORMAT is
# customer_inst_policy, so the actor sees the bilingual policy text that the
# winning arm of the 2026-08-18 evaluation received. Same actor, same RM, same CM
# cost source, same hyperparameters, same prompts, same step count.
#
# That isolation is the point. It makes "policy-prompt + PPO vs policy-prompt
# only" a controlled comparison, and Run A vs Run D a clean measurement of what
# the policy text alone contributes inside PPO. Using the judge cost here instead
# would confound the prompt change with a cost-signal change -- Run C2 already
# covers the good-signal arm separately.
#
# Deliberately uses COST_SOURCE=cm (the default): no API calls, so it does not
# compete with corpus generation for the rate-limited endpoint.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUNA_LOG=ppo_output/runA_long_20260829/training_log.jsonl
TARGET_STEPS=1000

while true; do
  steps=$(wc -l < "$RUNA_LOG" 2>/dev/null || echo 0)
  # Run A is done when it reaches its cap OR its container is gone.
  if [ "$steps" -ge "$TARGET_STEPS" ]; then
    echo "$(date -Is) Run A reached $steps steps"; break
  fi
  if [ -d ppo_output/runA_long_20260829/final ]; then
    echo "$(date -Is) Run A wrote final/ at $steps steps"; break
  fi
  sleep 120
done

# Give the container a moment to release GPU memory.
sleep 60
echo "$(date -Is) launching Run D on GPUs 0,1"

export CUDA_VISIBLE_DEVICES=0,1
export ACTOR_MODEL=/backup/model/costomer_model \
       RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
       CM_DIR=/backup/cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss \
       PROMPT_FORMAT=customer_inst_policy \
       POLICY_PROMPT_FILE=configs/policy_eval/compiled/system_prompt_bilingual.txt \
       PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
       KL_COEFF=0.2 ACTOR_LR=2e-5 CRITIC_LR=5e-5 LORA_R=16 LORA_ALPHA=32 \
       PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
       MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 \
       SAVE_EVERY=100 SAVE_EPOCHS=0 SAVE_CRITICS=0 EARLY_STOP_KL=15.0 \
       RUN_DIR=ppo_output/runD_policyprompt_20260829

exec bash run_exp_docker.sh \
  'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

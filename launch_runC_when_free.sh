#!/usr/bin/env bash
# Waits for Run B (CM training on GPUs 2,3) to exit, then launches Run C on
# those GPUs. Run C is the oracle-cost ablation: identical to Run A in every
# respect EXCEPT the cost signal, which comes from the Nemotron policy judge
# instead of the learned CM.
#
# The comparison Run A vs Run C is the whole point: same actor, same RM, same
# hyperparameters, same prompts, same step count -- only the cost signal differs.
# If Run C's lambda holds up and refusal behaviour appears while Run A's lambda
# collapses, the CM is the bottleneck and augmenting its data is justified.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Wait until only Run A's container remains (Run B has exited).
while [ "$(docker ps -q --filter ancestor=cost-model-trainer:v2 | wc -l)" -gt 1 ]; do
  sleep 60
done
echo "$(date -Is) GPUs 2,3 free — launching Run C"

set -a; . /home/ubuntu/.nchc_env; set +a
export NCHC_MODEL="${NCHC_MODEL:-NVIDIA-Nemotron-3-Super-120B-A12B}"
export JUDGE_WORKERS=4
export COST_SOURCE=judge
export CUDA_VISIBLE_DEVICES=2,3   # torch sees these as cuda:0 / cuda:1

export ACTOR_MODEL=/backup/model/costomer_model \
       RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
       CM_DIR=/backup/cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss \
       PROMPT_FORMAT=customer_inst \
       PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
       KL_COEFF=0.2 ACTOR_LR=2e-5 CRITIC_LR=5e-5 \
       LORA_R=16 LORA_ALPHA=32 \
       PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
       MAX_UPDATES=1000 EPOCHS=20 GRADIENT_CHECKPOINTING=1 \
       SAVE_EVERY=100 SAVE_EPOCHS=0 SAVE_CRITICS=0 EARLY_STOP_KL=15.0 \
       RUN_DIR=ppo_output/runC_judgecost_20260829

exec bash run_exp_docker.sh \
  'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

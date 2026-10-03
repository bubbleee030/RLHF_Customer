#!/usr/bin/env bash
# Runs P and Q -- the corrected recipe, on Run I's proven stability settings.
#
# WHY THE RECIPE CHANGED. I previously concluded that lambda_max=5.0 destabilises
# training, from Run H (lambda_max 5.0) diverging at step 111 while Run J
# (lambda_max 1.0) survived 400. That inference was confounded: Run H also ran
# actor_lr=1e-4. Grouping every run by learning rate settles it:
#
#   lambda_max 5.0 + actor_lr 2e-5 : A, C, C2, D, E, I   -> 0 of 6 diverged
#   lambda_max 5.0 + actor_lr 1e-4 : F, G, H             -> 2 of 3 diverged
#   lambda_max 1.0 + actor_lr 1e-4 : J, K, L             -> 2 of 3 diverged
#
# actor_lr is the divergence driver; capping lambda bought no stability and only
# weakened the constraint. Run I -- lambda_max 5.0, actor_lr 2e-5 -- is the single
# best bare-PPO result measured (0.7891, statistically tied with the policy-prompt
# bar) and the only run whose cost_mean genuinely improved: -1.90 -> -4.14 with KL
# never above 6.6. Runs L/M/N were all on the losing recipe.
#
#   Run P = Run I recipe + safety-aware RM + attainable threshold + ADVERSARIAL
#           prompts   -> the goal-critical run (fixes Run I's redteam collapse)
#   Run Q = identical but CUSTOMER prompts only
#           -> isolates the prompt-distribution effect from the new RM's effect
#
# Threshold -4.427 (not Run I's -6.309) so lambda can actually come off its cap:
# Run I's cost reached -4.14, which is within reach of this target.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

launch() {
  local name="$1" prompts="$2" gpus="$3"
  env \
    ACTOR_MODEL=/backup/model/costomer_model \
    RM_DIR=reward_output/rmS_safety_aware_20260831/epoch1 \
    CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
    PROMPT_FORMAT=customer_inst \
    PPO_PROMPT_FILE="$prompts" \
    LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
    MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
    SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="$gpus" \
    ACTOR_LR=2e-5 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=5.0 \
    RUN_DIR="ppo_output/$name" \
    bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'
}

launch runP_adversarial_20260831 datasets/ppo/mixed_customer_adversarial_20260831.jsonl 0,1 \
  > logs/runP_adversarial_20260831.log 2>&1 &
echo "$!" > .runP.pid; echo "Run P (adversarial) pid $(cat .runP.pid) on GPUs 0,1"
sleep 20
launch runQ_customer_20260831 datasets/reward/cs_within_train.jsonl 2,3 \
  > logs/runQ_customer_20260831.log 2>&1 &
echo "$!" > .runQ.pid; echo "Run Q (customer)    pid $(cat .runQ.pid) on GPUs 2,3"
wait

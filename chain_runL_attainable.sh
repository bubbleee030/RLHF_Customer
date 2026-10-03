#!/usr/bin/env bash
# Run L -- the first run whose safety constraint is actually SATISFIABLE.
#
# Defect #2 was `threshold=0.0`, which made the constraint inert (lambda only rose
# above an ~86% violation rate). I corrected it to -6.309 using the CM's means on
# its LABELLED PAIRS: violation +1.088, safe -6.698. Labelled-pair means do not
# transfer to generation, and the constraint stayed unsatisfiable.
#
# Calibrating on real generations requires the CM THIS RUN TRAINS AGAINST. The
# reference eval (fiveway_final_20260830) was scored by the OLD backup CM, so its
# numbers gave -5.441 -- still below the augmented CM's safe mean, i.e. still
# unattainable. The augmented CM's own scores on those same 2,445 responses
# (from the RM evaluation) joined to the blinded judge labels give:
#
#   safe   mean -4.751  (n=3419)      unsafe mean +1.735  (n=1471)
#   threshold for a 5% target = 0.05*(1.735) + 0.95*(-4.751) = -4.427
#
# A perfectly safe policy averages -4.751, which is now BELOW -4.427, so the
# constraint is satisfiable and lambda can come off its cap for the first time.
# Under -6.309 (and -5.441) it never could: lambda pinned at max forever and the
# actor paid a penalty it could not relieve by becoming safer -- precisely the
# C2 / H / J / K signature of lambda at cap, reward collapsing, violations flat.
# The augmented CM also separates far better than the old one (Youden J 0.772
# vs 0.568), which is why its attainable threshold sits a full point higher.
#
# Everything else is Run J, the best-tuned surviving bare config (400/400 steps,
# no divergence). Bare training so the result answers comparison #1 directly and
# can be evaluated in the prompt-MATCHED ppo_raw slot.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) Run L on GPUs $GPUS"

exec env \
  ACTOR_MODEL=/backup/model/costomer_model \
  RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
  CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
  PROMPT_FORMAT=customer_inst \
  PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
  LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
  MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=100 SAVE_EPOCHS=0 \
  SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPUS" \
  ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-4.427 LAMBDA_MAX=1.0 \
  RUN_DIR=ppo_output/runL_attainable_20260831 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py'

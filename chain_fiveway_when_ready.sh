#!/usr/bin/env bash
# Fires the final five-way evaluation once BOTH PPO adapters exist:
#   ppo_raw    <- Run A (bare PPO)
#   ppo_policy <- Run D (policy-prompt PPO)
#
# Produces the two comparisons the goal names:
#   bare PPO            vs policy-prompt-only
#   policy-prompt + PPO vs policy-prompt-only
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Prefer the run's final/ adapter; fall back to the highest step checkpoint so a
# run that hit its cap without writing final/ still yields a usable adapter.
resolve_adapter() {
  local run_dir="$1"
  if [ -d "$run_dir/final/actor_adapter" ]; then
    echo "$run_dir/final/actor_adapter"; return 0
  fi
  local last
  last=$(ls -d "$run_dir"/step*/actor_adapter 2>/dev/null \
         | sed 's/.*step\([0-9]*\)\/actor_adapter/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
  [ -n "$last" ] && { echo "$last"; return 0; }
  return 1
}

wait_for_run() {          # $1=run dir  $2=target steps
  local log="$1/training_log.jsonl"
  while true; do
    local n; n=$(wc -l < "$log" 2>/dev/null || echo 0)
    [ "$n" -ge "$2" ] && { echo "$(date -Is) $1 reached $n steps"; return 0; }
    [ -d "$1/final" ] && { echo "$(date -Is) $1 wrote final/ at $n steps"; return 0; }
    sleep 180
  done
}

wait_for_run ppo_output/runA_long_20260829 1000
wait_for_run ppo_output/runD_policyprompt_20260829 400

RAW=$(resolve_adapter ppo_output/runA_long_20260829) || { echo "no Run A adapter"; exit 1; }
POL=$(resolve_adapter ppo_output/runD_policyprompt_20260829) || { echo "no Run D adapter"; exit 1; }
echo "$(date -Is) ppo_raw=$RAW  ppo_policy=$POL"

# Wait for two free GPUs (the augmented-CM retrain may hold a pair).
pick_free_pair() {
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${used[0]:-99999}" -lt 1000 ] && [ "${used[1]:-99999}" -lt 1000 ]; then echo "0,1"; return 0; fi
  if [ "${used[2]:-99999}" -lt 1000 ] && [ "${used[3]:-99999}" -lt 1000 ]; then echo "2,3"; return 0; fi
  return 1
}
while ! GPUS=$(pick_free_pair); do sleep 180; done
echo "$(date -Is) five-way eval on GPUs $GPUS"

set -a; . /home/ubuntu/.nchc_env; set +a
export PPO_RAW_ADAPTER="$RAW" PPO_POLICY_ADAPTER="$POL"
export OUT=results/fiveway_final_20260830
CUDA_VISIBLE_DEVICES="$GPUS" bash run_fiveway_eval.sh
echo "$(date -Is) five-way eval finished -> $OUT/report"

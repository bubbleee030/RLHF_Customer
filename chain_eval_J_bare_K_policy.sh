#!/usr/bin/env bash
# The prompt-MATCHED evaluation. Every previous PPO evaluation put a bare-trained
# adapter into the "ppo_policy" slot, which prepends the policy system prompt at
# inference -- the exact mismatch generate_responses_fiveway.py warns about
# ("a mismatch here silently invalidates that arm"). Runs E, G, I and J all
# trained PROMPT_FORMAT=customer_inst, i.e. they never saw policy text.
#
# That is why Run G's 0.7266 is not a bare-PPO number: it is a bare-trained
# adapter handed a policy prompt it was never trained on, and it scores BELOW
# the 0.7857 policy-prompt-only bar. The goal was being measured against the
# wrong number. Valid bare-PPO results are base_raw-slot only: 0.6111 (Run A/D)
# and 0.6250 (Run E). The real gap is ~0.16, not 0.059.
#
# This run makes both slots match their training:
#   ppo_raw    <- Run J  (trained customer_inst, served bare)      -> comparison #1
#   ppo_policy <- Run K  (trained customer_inst_policy zh, served zh) -> comparison #2
#
# Run J is the best-tuned surviving bare config (threshold -6.309, lambda_max 1.0,
# actor_lr 1e-4, 400/400 steps). It has never been measured bare.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC=results/fiveway_final_20260830
OUT=results/fiveway_JK_matched_20260831
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
ZH=configs/policy_eval/compiled/system_prompt_zh.txt
BI=configs/policy_eval/compiled/system_prompt_bilingual.txt
ACTOR=/backup/model/costomer_model
RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best
CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss
N=3
K=ppo_output/runK_combined_20260831
J=ppo_output/runJ_lambdacap_20260830

# Wait for Run K: final/, or a stall with a usable checkpoint (it early-stops on KL>15).
while true; do
  [ -d "$K/final" ] && { echo "$(date -Is) Run K wrote final/"; break; }
  n=$(wc -l < "$K/training_log.jsonl" 2>/dev/null || echo 0)
  m=$(stat -c %Y "$K/training_log.jsonl" 2>/dev/null || echo 0)
  if [ $(( $(date +%s) - m )) -gt 1800 ]; then
    echo "$(date -Is) Run K log idle >30min at step $n; using its latest checkpoint"; break
  fi
  sleep 300
done

resolve() {
  [ -d "$1/final/actor_adapter" ] && { echo "$1/final/actor_adapter"; return 0; }
  ls -d "$1"/step*/actor_adapter 2>/dev/null \
    | sed 's/.*step\([0-9]*\)\/actor_adapter/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}
J_AD=$(resolve "$J"); K_AD=$(resolve "$K")
[ -n "$J_AD" ] && [ -n "$K_AD" ] || { echo "FATAL: missing adapter J=$J_AD K=$K_AD"; exit 1; }
echo "$(date -Is) J(bare)=$J_AD  K(policy)=$K_AD"

mkdir -p "$OUT/responses"
cat > "$OUT/VARIANT_MAPPING.json" <<JSON
{
  "note": "PROMPT-MATCHED evaluation. Both PPO slots are served the prompt format they were TRAINED with, unlike fiveway_EF/fiveway_IJ where bare-trained adapters sat in the policy slot.",
  "ppo_raw": "RUN J - trained customer_inst (bare), served bare. threshold -6.309, lambda_max 1.0, actor_lr 1e-4, 400 steps. THIS IS THE COMPARISON-1 NUMBER.",
  "ppo_policy": "RUN K - trained customer_inst_policy (zh), served zh policy. augmented CM, threshold -6.309, lambda_max 1.0, actor_lr 1e-4.",
  "J_adapter": "$J_AD",
  "K_adapter": "$K_AD",
  "bar": "base_policy_zh safe_outcome = 0.7857 (99/126, Wilson [0.7062, 0.8483])"
}
JSON

for V in base_raw base_policy_zh base_policy_bilingual; do
  cp -n "$SRC/responses/${V}.jsonl" "$OUT/responses/${V}.jsonl" 2>/dev/null && echo "copied $V"
done

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) generating on GPUs $GPUS"

set -a; . /home/ubuntu/.nchc_env; set +a
export CUDA_VISIBLE_DEVICES="$GPUS"
export PPO_RAW_ADAPTER="$J_AD" PPO_POLICY_ADAPTER="$K_AD" OUT MANIFEST POLICIES ZH BI ACTOR RM_DIR CM_DIR N

bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; set -e
for pair in "ppo_raw 0" "ppo_policy 1"; do
  set -- $pair
  python3 -m scripts.policy_eval.generate_responses_fiveway \
    --variant "$1" --manifest "$MANIFEST" \
    --compiled-zh "$ZH" --compiled-bilingual "$BI" --actor-model "$ACTOR" \
    --ppo-policy-prompt zh \
    --ppo-raw-adapter-dir "$PPO_RAW_ADAPTER" --ppo-policy-adapter-dir "$PPO_POLICY_ADAPTER" \
    --output "$OUT/responses/$1.jsonl" --device "cuda:$2" --n "$N" &
done
wait
R=("$OUT"/responses/base_raw.jsonl "$OUT"/responses/base_policy_zh.jsonl \
   "$OUT"/responses/base_policy_bilingual.jsonl "$OUT"/responses/ppo_raw.jsonl \
   "$OUT"/responses/ppo_policy.jsonl)
EXP=$(( $(wc -l < "$MANIFEST") * N * 5 ))
python3 -m scripts.policy_eval.score_responses --responses "${R[@]}" \
  --rm-dir "$RM_DIR" --cm-dir "$CM_DIR" --output "$OUT/scores.jsonl" \
  --rm-device cuda:0 --cm-device cuda:1 --expected "$EXP"
python3 -m scripts.policy_eval.judge_policy_fiveway --responses "${R[@]}" \
  --policies "$POLICIES" --output "$OUT/judges.jsonl" \
  --initial-max-tokens 4800 \
  --expected-groups $(( $(wc -l < "$MANIFEST") * N ))
python3 -m scripts.policy_eval.aggregate_report_fiveway --manifest "$MANIFEST" \
  --responses "${R[@]}" --scores "$OUT/scores.jsonl" --judges "$OUT/judges.jsonl" \
  --output-dir "$OUT/report" --expected-prompts "$(wc -l < "$MANIFEST")" \
  --expected-responses "$EXP" --expected-scores "$EXP" \
  --expected-judges $(( $(wc -l < "$MANIFEST") * N * 2 ))
'
echo "$(date -Is) prompt-matched evaluation complete -> $OUT/report"

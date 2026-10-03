#!/usr/bin/env bash
# Blinded evaluation of Runs I and J -- the two configurations that ran with an
# ACTIVE safety constraint and survived.
#
#   variant slot "ppo_raw"    <- Run I  (threshold -6.309, lambda_max 5.0, lr 2e-5)
#   variant slot "ppo_policy" <- Run J  (threshold -6.309, lambda_max 1.0, lr 1e-4)
#
# Slot names come from the evaluator's fixed VARIANTS tuple; the mapping is
# written to VARIANT_MAPPING.json so these are not misread as Run A / Run D.
#
# Every training-log trend this investigation produced turned out to be prompt-
# sampling noise (C2, E and G all showed the same phantom ~0.13 decline). Only
# this fixed 163-prompt blinded evaluation can say whether behaviour changed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC=results/fiveway_final_20260830
OUT=results/fiveway_IJ_20260830
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
ZH=configs/policy_eval/compiled/system_prompt_zh.txt
BI=configs/policy_eval/compiled/system_prompt_bilingual.txt
ACTOR=/backup/model/costomer_model
RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best
CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss
N=3

wait_for() {
  while true; do
    n=$(wc -l < "$1/training_log.jsonl" 2>/dev/null || echo 0)
    [ "$n" -ge "$2" ] && { echo "$(date -Is) $1 at $n steps"; return 0; }
    [ -d "$1/final" ] && { echo "$(date -Is) $1 wrote final/ at $n"; return 0; }
    sleep 180
  done
}
resolve() {
  [ -d "$1/final/actor_adapter" ] && { echo "$1/final/actor_adapter"; return 0; }
  ls -d "$1"/step*/actor_adapter 2>/dev/null \
    | sed 's/.*step\([0-9]*\)\/actor_adapter/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

wait_for ppo_output/runI_active_slow_20260830 400
wait_for ppo_output/runJ_lambdacap_20260830 400
I_AD=$(resolve ppo_output/runI_active_slow_20260830)
J_AD=$(resolve ppo_output/runJ_lambdacap_20260830)
echo "$(date -Is) I=$I_AD  J=$J_AD"

mkdir -p "$OUT/responses"
cat > "$OUT/VARIANT_MAPPING.json" <<JSON
{
  "note": "PPO slots hold Runs I and J, NOT Run A / Run D. Baselines copied from fiveway_final_20260830 (adapter-free, so identical).",
  "ppo_raw": "RUN I - augmented CM, threshold -6.309, lambda_max 5.0, actor_lr 2e-5",
  "ppo_policy": "RUN J - augmented CM, threshold -6.309, lambda_max 1.0, actor_lr 1e-4",
  "I_adapter": "$I_AD",
  "J_adapter": "$J_AD"
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
echo "$(date -Is) generating I/J arms on GPUs $GPUS"

set -a; . /home/ubuntu/.nchc_env; set +a
export CUDA_VISIBLE_DEVICES="$GPUS"
export PPO_RAW_ADAPTER="$I_AD" PPO_POLICY_ADAPTER="$J_AD" OUT MANIFEST POLICIES ZH BI ACTOR RM_DIR CM_DIR N

bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; set -e
for pair in "ppo_raw 0" "ppo_policy 1"; do
  set -- $pair
  python3 -m scripts.policy_eval.generate_responses_fiveway \
    --variant "$1" --manifest "$MANIFEST" \
    --compiled-zh "$ZH" --compiled-bilingual "$BI" --actor-model "$ACTOR" \
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
echo "$(date -Is) I/J evaluation complete -> $OUT/report"

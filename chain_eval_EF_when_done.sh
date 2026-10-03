#!/usr/bin/env bash
# Evaluate Runs E and F once both finish, on the SAME 163-prompt manifest and
# through the SAME blinded judge as the first five-way — otherwise their KL
# movement is uninterpretable (a moving actor may be learning OR reward hacking).
#
# Trick that avoids ~2h of redundant GPU work: the three baseline arms
# (base_raw, base_policy_zh, base_policy_bilingual) contain no PPO adapter, so
# their generations are identical to the first evaluation's. They are copied in
# rather than regenerated. Only the two PPO slots are generated fresh:
#
#     variant slot "ppo_raw"     <- Run E adapter (augmented CM, hyperparams unchanged)
#     variant slot "ppo_policy"  <- Run F adapter (augmented CM, leash released)
#
# The slot NAMES are dictated by the evaluator's fixed VARIANTS tuple; the
# mapping is recorded in VARIANT_MAPPING.json so results are not misread as
# Run A / Run D.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC=results/fiveway_final_20260830
OUT=results/fiveway_EF_20260830
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
ZH=configs/policy_eval/compiled/system_prompt_zh.txt
BI=configs/policy_eval/compiled/system_prompt_bilingual.txt
ACTOR=/backup/model/costomer_model
RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best
CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss
N=3

wait_for() {   # $1 = run dir, $2 = target steps
  while true; do
    n=$(wc -l < "$1/training_log.jsonl" 2>/dev/null || echo 0)
    [ "$n" -ge "$2" ] && { echo "$(date -Is) $1 reached $n steps"; return 0; }
    [ -d "$1/final" ] && { echo "$(date -Is) $1 wrote final/ at $n"; return 0; }
    sleep 180
  done
}
resolve() {
  [ -d "$1/final/actor_adapter" ] && { echo "$1/final/actor_adapter"; return 0; }
  ls -d "$1"/step*/actor_adapter 2>/dev/null \
    | sed 's/.*step\([0-9]*\)\/actor_adapter/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

wait_for ppo_output/runE_augcm_20260830 400
wait_for ppo_output/runG_lr_only_20260830 400
E_AD=$(resolve ppo_output/runE_augcm_20260830)
F_AD=$(resolve ppo_output/runG_lr_only_20260830)
echo "$(date -Is) E=$E_AD  F=$F_AD"

mkdir -p "$OUT/responses"
cat > "$OUT/VARIANT_MAPPING.json" <<JSON
{
  "note": "Slot names come from the evaluator's fixed VARIANTS tuple. The two PPO slots hold DIFFERENT runs than results/fiveway_final_20260830.",
  "base_raw": "base model, no policy prompt (copied from fiveway_final)",
  "base_policy_zh": "base + zh policy (copied)",
  "base_policy_bilingual": "base + bilingual policy (copied)",
  "ppo_raw": "RUN E - augmented CM, actor_lr=2e-5, kl_coeff=0.2 (unchanged)",
  "ppo_policy": "RUN G - augmented CM, actor_lr=1e-4, kl_coeff=0.2 (lr raised, leash KEPT)",
  "E_adapter": "$E_AD",
  "G_adapter": "$F_AD"
}
JSON

# Baselines are adapter-free and therefore identical to the first evaluation.
for V in base_raw base_policy_zh base_policy_bilingual; do
  cp -n "$SRC/responses/${V}.jsonl" "$OUT/responses/${V}.jsonl" 2>/dev/null && echo "copied $V"
done

# Wait for GPUs (the first five-way may still be judging; that needs no GPU).
pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) generating E/F arms on GPUs $GPUS"

set -a; . /home/ubuntu/.nchc_env; set +a
export CUDA_VISIBLE_DEVICES="$GPUS"
export PPO_RAW_ADAPTER="$E_AD" PPO_POLICY_ADAPTER="$F_AD" OUT MANIFEST POLICIES ZH BI ACTOR RM_DIR CM_DIR N

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
  --expected-groups $(( $(wc -l < "$MANIFEST") * N ))
python3 -m scripts.policy_eval.aggregate_report_fiveway --manifest "$MANIFEST" \
  --responses "${R[@]}" --scores "$OUT/scores.jsonl" --judges "$OUT/judges.jsonl" \
  --output-dir "$OUT/report" --expected-prompts "$(wc -l < "$MANIFEST")" \
  --expected-responses "$EXP" --expected-scores "$EXP" \
  --expected-judges $(( $(wc -l < "$MANIFEST") * N * 2 ))
'
echo "$(date -Is) E/F evaluation complete -> $OUT/report"

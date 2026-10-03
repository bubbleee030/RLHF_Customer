#!/usr/bin/env bash
# Parameterised final evaluation:  chain_eval_run.sh <RUN_DIR_NAME> <OUT_TAG> <LOGFILE>
#
# Both PPO slots hold the SAME adapter, served two ways, so one judging budget
# answers both comparisons:
#   ppo_raw    -> served BARE, prompt-MATCHED (these runs train customer_inst).
#                 THIS is the comparison-1 number: bare PPO vs policy-prompt-only.
#   ppo_policy -> served with the zh policy prompt. Deliberately MISMATCHED and
#                 labelled as a COMPOSITION test, not a matched arm.
#
# CHECKPOINT SELECTION -- pre-registered before any result was seen. `final/` is
# written AT the KL>15 early stop, i.e. already past the declared stability limit,
# but hand-picking a healthier earlier checkpoint after the fact is cherry-picking.
#   completed 400 steps -> final/
#   early-stopped       -> LAST periodic checkpoint whose preceding 25-step mean
#                          KL was < 10.0 (a margin below the 15.0 stop)
#   none qualifying     -> final/, stated as such
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUNNAME="$1"; TAG="$2"; TRAINLOG="$3"
RUN="ppo_output/$RUNNAME"
SRC=results/fiveway_final_20260830
OUT="results/fiveway_${TAG}_20260831"
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
ZH=configs/policy_eval/compiled/system_prompt_zh.txt
BI=configs/policy_eval/compiled/system_prompt_bilingual.txt
ACTOR=/backup/model/costomer_model
RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best
CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss
N=3

while true; do
  [ -d "$RUN/final" ] && { echo "$(date -Is) $RUNNAME wrote final/"; break; }
  n=$(wc -l < "$RUN/training_log.jsonl" 2>/dev/null || echo 0)
  m=$(stat -c %Y "$RUN/training_log.jsonl" 2>/dev/null || echo 0)
  if [ $(( $(date +%s) - m )) -gt 1800 ]; then
    echo "$(date -Is) $RUNNAME idle >30min at step $n; using latest checkpoint"; break
  fi
  sleep 300
done

AD=""
if ! grep -q 'EARLY STOP' "$TRAINLOG" 2>/dev/null; then
  [ -d "$RUN/final/actor_adapter" ] && AD="$RUN/final/actor_adapter"
fi
if [ -z "$AD" ]; then
  AD=$(python3 - "$RUN" <<'PYEOF'
import json, os, statistics, sys, glob
run=sys.argv[1]
rows=[json.loads(l) for l in open(os.path.join(run,"training_log.jsonl"))]
best=None
for path in glob.glob(os.path.join(run,"step*","actor_adapter")):
    step=int(os.path.basename(os.path.dirname(path))[4:])
    w=[r["kl"] for r in rows[max(0,step-25):step]]
    if w and statistics.mean(w) < 10.0 and (best is None or step > best[0]):
        best=(step,path)
print(best[1] if best else "")
PYEOF
)
fi
[ -n "$AD" ] || AD="$RUN/final/actor_adapter"
[ -d "$AD" ] || { echo "FATAL: no adapter for $RUNNAME"; exit 1; }
echo "$(date -Is) $RUNNAME adapter = $AD"

mkdir -p "$OUT/responses"
cat > "$OUT/VARIANT_MAPPING.json" <<JSON
{
  "run": "$RUNNAME",
  "ppo_raw": "$RUNNAME served BARE -- prompt-MATCHED. The comparison-1 number.",
  "ppo_policy": "$RUNNAME served with the zh policy prompt -- COMPOSITION test, NOT a matched arm.",
  "adapter": "$AD",
  "recipe": "actor_lr 2e-5, lambda_max 5.0, threshold -4.427, safety-aware RM, augmented CM",
  "bar": "base_policy_zh: 0.7857 (fiveway_final) / 0.7969 (fiveway_IJ); judge re-run noise ~1.1pt",
  "decisive_track": "redteam (21 responses): Run I 0.2857 vs policy prompt 0.6667"
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
export PPO_RAW_ADAPTER="$AD" PPO_POLICY_ADAPTER="$AD" OUT MANIFEST POLICIES ZH BI ACTOR RM_DIR CM_DIR N

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
echo "$(date -Is) $RUNNAME evaluation complete -> $OUT/report"

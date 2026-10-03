#!/usr/bin/env bash
# Final five-way blinded evaluation — the two comparisons the goal requires:
#   1. bare PPO            (ppo_raw)    vs policy-prompt-only (base_policy_bilingual)
#   2. policy-prompt + PPO (ppo_policy) vs policy-prompt-only (base_policy_bilingual)
#
# Uses the SAME 163-prompt manifest as the 2026-08-18 four-way evaluation
# (reconstructed from its own response files), so the new numbers are directly
# comparable to the published ones rather than merely similar.
#
# Stages: generate (GPU, per variant) -> score (GPU) -> judge (API) -> aggregate.
# Every stage is a tested module; this only sequences them.
#
# Usage:
#   PPO_RAW_ADAPTER=... PPO_POLICY_ADAPTER=... bash run_fiveway_eval.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BACKUP=/backup
OUT="${OUT:-results/fiveway_$(date +%Y%m%d_%H%M%S)}"
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
ZH=configs/policy_eval/compiled/system_prompt_zh.txt
BI=configs/policy_eval/compiled/system_prompt_bilingual.txt
ACTOR=${ACTOR:-$BACKUP/model/costomer_model}
RM_DIR=${RM_DIR:-$BACKUP/reward_output/run_reward_cs_within_20260803/best}
CM_DIR=${CM_DIR:-$BACKUP/cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss}
N=${N:-3}

: "${PPO_RAW_ADAPTER:?set PPO_RAW_ADAPTER (Run A final actor_adapter dir)}"
: "${PPO_POLICY_ADAPTER:?set PPO_POLICY_ADAPTER (Run D final actor_adapter dir)}"

mkdir -p "$OUT/responses"
echo "=== five-way eval -> $OUT ==="
echo "manifest : $MANIFEST ($(wc -l < "$MANIFEST") prompts, n=$N samples)"
echo "ppo_raw  : $PPO_RAW_ADAPTER"
echo "ppo_polcy: $PPO_POLICY_ADAPTER"

# --- Stage 1: generation, TWO variants at a time (one per visible GPU) ---
# Generation is batch-1 and ~10s/sample; 5 variants x 489 samples is ~6.8h
# serially while the second GPU sits idle. Pairing them halves it.
#
# Deliberately NOT skipping non-empty output files: the worker resumes from its
# own checkpoint (job_key), so an interrupted variant must be RE-ENTERED to
# finish. Skipping "already has rows" would silently ship a partial variant --
# which would have happened here, since base_raw held 7 of 489 rows.
VARIANT_LIST=(base_raw base_policy_zh base_policy_bilingual ppo_raw ppo_policy)
gen_one() {
  local variant="$1" dev="$2"
  python3 -m scripts.policy_eval.generate_responses_fiveway \
    --variant "$variant" \
    --manifest "$MANIFEST" \
    --compiled-zh "$ZH" --compiled-bilingual "$BI" \
    --actor-model "$ACTOR" \
    --ppo-raw-adapter-dir "$PPO_RAW_ADAPTER" \
    --ppo-policy-adapter-dir "$PPO_POLICY_ADAPTER" \
    --output "$OUT/responses/${variant}.jsonl" --device "cuda:${dev}" --n "$N"
}
idx=0
while [ "$idx" -lt "${#VARIANT_LIST[@]}" ]; do
  pids=()
  for dev in 0 1; do
    [ "$idx" -lt "${#VARIANT_LIST[@]}" ] || break
    V="${VARIANT_LIST[$idx]}"
    echo "--- generating $V on cuda:${dev} ---"
    gen_one "$V" "$dev" &
    pids+=("$!")
    idx=$((idx + 1))
  done
  for p in "${pids[@]}"; do
    wait "$p" || { echo "generation failed (pid $p)"; exit 1; }
  done
done

RESPS=("$OUT"/responses/base_raw.jsonl "$OUT"/responses/base_policy_zh.jsonl \
       "$OUT"/responses/base_policy_bilingual.jsonl "$OUT"/responses/ppo_raw.jsonl \
       "$OUT"/responses/ppo_policy.jsonl)
EXPECTED_RESP=$(( $(wc -l < "$MANIFEST") * N * 5 ))

# --- Stage 2: RM/CM scoring ---
if [ ! -s "$OUT/scores.jsonl" ]; then
  echo "--- scoring ---"
  python3 -m scripts.policy_eval.score_responses \
    --responses "${RESPS[@]}" --rm-dir "$RM_DIR" --cm-dir "$CM_DIR" \
    --output "$OUT/scores.jsonl" --rm-device cuda:0 --cm-device cuda:1 \
    --expected "$EXPECTED_RESP"
fi

# --- Stage 3: blinded judging (API; paced by the module's own backoff) ---
if [ ! -s "$OUT/judges.jsonl" ]; then
  echo "--- judging ---"
  python3 -m scripts.policy_eval.judge_policy_fiveway \
    --responses "${RESPS[@]}" --policies "$POLICIES" \
    --output "$OUT/judges.jsonl" \
    --expected-groups $(( $(wc -l < "$MANIFEST") * N ))
fi

# --- Stage 4: aggregation ---
echo "--- aggregating ---"
python3 -m scripts.policy_eval.aggregate_report_fiveway \
  --manifest "$MANIFEST" --responses "${RESPS[@]}" \
  --scores "$OUT/scores.jsonl" --judges "$OUT/judges.jsonl" \
  --output-dir "$OUT/report" \
  --expected-prompts "$(wc -l < "$MANIFEST")" \
  --expected-responses "$EXPECTED_RESP" \
  --expected-scores "$EXPECTED_RESP" \
  --expected-judges $(( $(wc -l < "$MANIFEST") * N * 2 ))

echo "=== DONE -> $OUT/report ==="

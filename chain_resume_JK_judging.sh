#!/usr/bin/env bash
# Resume the matched J-bare/K-policy evaluation at the JUDGING stage only.
# Generation (5 x 489) and scoring (2445) are already complete on disk; the judge
# resumes from judges.jsonl via load_checkpoint, so nothing already judged is redone.
#
# Restarted to pick up the retry-policy fix: parse/validation failures were being
# retried on the 429 backoff ladder (up to 300s a try), which cost the I/J run
# 8,510 seconds -- 2.4 hours -- of pure sleeping across 123 parse retries. Waiting
# cannot fix malformed JSON; only a resample can. Rate-limit backoff is unchanged.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=results/fiveway_JK_matched_20260831
MANIFEST=configs/policy_eval/fiveway_manifest_163.jsonl
POLICIES=configs/policy_eval/example_policy.jsonl
N=3
set -a; . /home/ubuntu/.nchc_env; set +a
export OUT MANIFEST POLICIES N
bash run_exp_docker.sh 'set -e
R=("$OUT"/responses/base_raw.jsonl "$OUT"/responses/base_policy_zh.jsonl \
   "$OUT"/responses/base_policy_bilingual.jsonl "$OUT"/responses/ppo_raw.jsonl \
   "$OUT"/responses/ppo_policy.jsonl)
EXP=$(( $(wc -l < "$MANIFEST") * N * 5 ))
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
echo "$(date -Is) matched JK evaluation complete -> $OUT/report"

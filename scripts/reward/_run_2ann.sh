#!/usr/bin/env bash
# 2-annotator experiment: agreed-only then union, both eval'd on the SAME human
# by-prompt holdout (reward_eval_byprompt.jsonl) for comparison to the 0.60 baseline.
set -uo pipefail
cd /home/ubuntu/reward_model
export EVAL=datasets/reward/reward_eval_byprompt.jsonl
echo "=== AGREED run starting $(date) ==="
SPLIT=agreed bash scripts/reward/run_reward_docker.sh > reward_output/train_agreed.log 2>&1
echo "=== AGREED exit=$? $(date) ==="
echo "=== UNION run starting $(date) ==="
SPLIT=union bash scripts/reward/run_reward_docker.sh > reward_output/train_union.log 2>&1
echo "=== UNION exit=$? $(date) ==="
echo "=== ALL DONE $(date) ==="

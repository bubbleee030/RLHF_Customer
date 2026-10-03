#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/reward_model
echo "=== BY-PROMPT run starting $(date) ==="
SPLIT=byprompt bash scripts/reward/run_reward_docker.sh > reward_output/train_byprompt.log 2>&1
echo "=== BY-PROMPT exit=$? $(date) ==="
echo "=== BY-PAIR run starting $(date) ==="
SPLIT=bypair bash scripts/reward/run_reward_docker.sh > reward_output/train_bypair.log 2>&1
echo "=== BY-PAIR exit=$? $(date) ==="
echo "=== ALL DONE $(date) ==="

#!/usr/bin/env bash
# Detached watcher: wait for the currently-running by-prompt container to finish,
# then launch the by-pair run. Started via `setsid nohup` so it survives SSH drop.
set -uo pipefail
cd /home/ubuntu/reward_model
CID="${1:-}"
echo "=== [watcher] $(date) waiting for by-prompt container ${CID} ==="
if [ -n "${CID}" ]; then
    docker wait "${CID}" 2>/dev/null || echo "[watcher] container ${CID} already gone"
fi
echo "=== [watcher] $(date) by-prompt done; launching by-pair ==="
SPLIT=bypair bash scripts/reward/run_reward_docker.sh > reward_output/train_bypair.log 2>&1
echo "=== [watcher] $(date) ALL DONE (exit=$?) ==="

#!/usr/bin/env bash
# 一鍵啟動完整 Ministral 訓練 pipeline（完全 detached，關 Mac 後繼續跑）
# 模型：8B Instruct (Adafactor) + 8B Base (Adafactor) + 3B Instruct (AdamW)
# 用法：bash start_training.sh
cd "$(dirname "$0")"
nohup bash scripts/run_all_pipeline.sh > /tmp/ministral_pipeline_full.log 2>&1 &
echo "Pipeline 已啟動，PID=$!"
echo "Log: /tmp/ministral_pipeline_full.log"
echo "查看進度：tail -f /tmp/ministral_pipeline_full.log"

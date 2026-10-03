#!/bin/bash
set -euo pipefail

echo "======================================"
echo " HPC Migration Packager"
echo "======================================"
echo ""
echo "This script will:"
echo "1. Prompt for your HuggingFace token"
echo "2. Download google/gemma-3-12b-pt locally to ./model"
echo "3. Package the necessary files into hpc_upload.tar.gz"
echo ""

read -p "Enter your HuggingFace Token (hf_...): " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Token cannot be empty."
    exit 1
fi

HF_CLI="huggingface-cli"
HF_LOGIN_CMD="login"
if command -v hf &> /dev/null; then
    HF_CLI="hf"
    HF_LOGIN_CMD="auth login"
elif ! command -v huggingface-cli &> /dev/null; then
    if [ -f "$HOME/.local/bin/huggingface-cli" ]; then
        HF_CLI="$HOME/.local/bin/huggingface-cli"
    else
        HF_CLI="python3 -m huggingface_hub.commands.huggingface_cli"
    fi
fi

echo "Logging into HuggingFace..."
$HF_CLI $HF_LOGIN_CMD --token $HF_TOKEN

export HF_HOME=$(pwd)/model
echo "Downloading google/gemma-3-12b-pt to $HF_HOME (this may take a while)..."
$HF_CLI download google/gemma-3-12b-pt --token $HF_TOKEN

echo "Packaging files into hpc_upload.tar.gz..."
tar czf hpc_upload.tar.gz \
    cost-model-trainer_v2.sif \
    model/hub/models--google--gemma-3-12b-pt/ \
    datasets/cost_dataset_for_safe_rlhf_clean.jsonl \
    scripts/train_cost_model_v2.py \
    scripts/train_cost_model.sh \
    job.sh

echo "Done! You can now transfer hpc_upload.tar.gz to your HPC."

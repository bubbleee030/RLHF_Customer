#!/bin/bash
#SBATCH -A GOV113021
#SBATCH -N 1
#SBATCH -J train_cost_model
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -p gp1d
#SBATCH --cpus-per-task=8

# 如果國網計算節點需要 proxy 連網 (例如回傳 wandb 數據)，請取消註解
#export HTTP_PROXY=socks5h://un-ln01:9000
#export HTTPS_PROXY=socks5h://un-ln01:9000

# 1. 您的 Singularity SIF 映像檔路徑
export SIF_PATH=/work/u7616835/cost-model-trainer_v2.sif

# 2. 定義外部 (國網) 實際的存放路徑
export HOST_MODEL_DIR=/work/u7616835/model
export HOST_WORKSPACE=/work/u7616835/reward_model

# 3. Weights & Biases API KEY (若腳本中有 --log_type wandb 建議設定)
# export WANDB_API_KEY="your-wandb-api-key"

echo "=========================================================="
echo "Job Start Time: $(date)"
echo "Node List: $SLURM_NODELIST"
echo "=========================================================="

# 透過 srun 搭配 singularity 執行
# 設定 HF_HOME 到 /model 並開啟離線模式
srun singularity exec --nv \
    -B ${HOST_MODEL_DIR}:/model \
    -B ${HOST_WORKSPACE}:/data_workspace \
    --env HF_HOME=/model \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PRECISION="--bf16 --load-in-half" \
    ${SIF_PATH} bash /data_workspace/scripts/train_cost_model.sh \
        google/gemma-3-12b-pt \
        /data_workspace/cost_output/run_gemma3_12b \
        /data_workspace/datasets/cost_dataset_for_safe_rlhf_clean.jsonl

echo "=========================================================="
echo "Job End Time: $(date)"
echo "=========================================================="

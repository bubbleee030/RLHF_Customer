FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 變更為 noninteractive 避免編譯時卡在使用者提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝基本環境相依套件
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone safe-rlhf repo (原作者程式)
RUN git clone https://github.com/PKU-Alignment/safe-rlhf.git

# 根據 safe-rlhf 專案內容，安裝相依賴套件
WORKDIR /workspace/safe-rlhf
RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    accelerate==0.27.2 \
    deepspeed==0.14.4 \
    datasets>=2.17.1 \
    torchvision \
    torchaudio \
    wandb \
    numpy \
    evaluate \
    scipy

# 安裝 safe-rlhf 套件（使用可編輯模式，方便追蹤與引用）
RUN pip install -e .

# 預設執行點將會是我們的訓練腳本
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]

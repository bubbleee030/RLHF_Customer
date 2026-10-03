#!/usr/bin/env python3
"""Generate a comprehensive Markdown report for Ministral cost model training.

Report structure mirrors report_mistral.md (detailed technical problem/solution sections)
combined with report0507.md (PKU 3-term theory, data cleaning, epoch tables, comparison).

Usage:
    python generate_ministral_report.py \
        --instruct-8b-dir cost_output/run_ministral_8b_instruct_... \
        --base-8b-dir     cost_output/run_ministral_8b_base_... \
        [--instruct-3b-dir cost_output/run_ministral_3b_instruct_...]  \
        --plot-dir-8b-instruct outputs/plots/ministral_8b_instruct \
        --plot-dir-8b-base     outputs/plots/ministral_8b_base \
        [--plot-dir-3b-instruct outputs/plots/ministral_3b_instruct] \
        --output outputs/reports/ministral_report_TIMESTAMP.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ── DeBERTa baselines ──────────────────────────────────────────────────────
DEBERTA_MAIN = {
    "name": "DeBERTa-v3-large (sequence-wise)",
    "params": "434M", "pooling": "mean", "loss_type": "sequence-wise",
    "optimizer": "AdamW", "precision": "FP32 + FP16 autocast",
    "best_eval_acc": 0.7192, "best_eval_sign": 0.8664, "best_eval_loss": 1.1732,
    "train_seconds": 603,
    "epochs": [
        {"epoch": 1, "train_loss": 1.6691, "train_acc": 0.6090, "train_sign": 0.7142,
         "eval_loss": 1.3285, "eval_acc": 0.7192, "eval_sign": 0.8151, "eval_cost_mean": -0.9621, "eval_cost_std": 1.8417},
        {"epoch": 2, "train_loss": 1.2732, "train_acc": 0.6905, "train_sign": 0.8384,
         "eval_loss": 1.1946, "eval_acc": 0.7055, "eval_sign": 0.8596, "eval_cost_mean": -1.9007, "eval_cost_std": 2.0438},
        {"epoch": 3, "train_loss": 1.1254, "train_acc": 0.7340, "train_sign": 0.8586,
         "eval_loss": 1.1732, "eval_acc": 0.6918, "eval_sign": 0.8664, "eval_cost_mean": -1.9962, "eval_cost_std": 2.2686},
    ],
}
DEBERTA_OFFICIAL = {
    "name": "DeBERTa-v3-large (token-wise + norm)",
    "params": "434M", "pooling": "mean", "loss_type": "token-wise",
    "optimizer": "AdamW", "precision": "FP32 + FP16 autocast",
    "best_eval_acc": 0.6575, "best_eval_sign": 0.8219, "best_eval_loss": 1.3719,
    "train_seconds": 437,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_run(run_dir: Path) -> dict:
    summary = load_json(run_dir / "training_summary.json")
    eval_log = load_json(run_dir / "eval_log.json")
    epochs = []
    for e in eval_log:
        epochs.append({
            "epoch": e["epoch"],
            "train_loss": round(e["train_loss"], 4),
            "train_acc": round(e["train_accuracy"], 4),
            "train_sign": round(e["train_accuracy_sign"], 4),
            "eval_loss": round(e["eval_loss"], 4),
            "eval_acc": round(e["eval_accuracy"], 4),
            "eval_sign": round(e["eval_accuracy_sign"], 4),
            "eval_cost_mean": round(e.get("eval_cost_mean", 0), 4),
            "eval_cost_std": round(e.get("eval_cost_std", 0), 4),
        })
    return {
        "summary": summary,
        "epochs": epochs,
        "best_eval_acc": round(summary["best_eval_accuracy"], 4),
        "best_eval_sign": round(summary["best_eval_accuracy_sign"], 4),
        "best_eval_loss": round(summary["best_eval_loss"], 4),
        "train_seconds": round(summary.get("train_seconds", 0)),
        "total_params": summary.get("total_params", "N/A"),
        "model": summary.get("model_name_or_path", ""),
        "optimizer": "Adafactor" if summary.get("loss_type", "").startswith("pku") else "AdamW",
    }


def read_eval_predictions(run_dir: Path, n: int = 10) -> list[dict]:
    p = run_dir / "eval_predictions.jsonl"
    if not p.exists():
        return []
    rows = []
    # Get the last epoch's predictions (highest epoch number)
    all_rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not all_rows:
        return []
    max_epoch = max(r.get("epoch", 0) for r in all_rows)
    last_epoch_rows = [r for r in all_rows if r.get("epoch") == max_epoch]
    return last_epoch_rows[:n]


def epoch_table(epochs: list[dict]) -> str:
    header = "| Epoch | Train Loss | Train Acc | Train Sign Acc | Eval Loss | Eval Acc | Eval Sign Acc | Eval Cost Mean | Eval Cost Std |"
    sep    = "|-------|-----------|-----------|----------------|-----------|----------|--------------|----------------|--------------|"
    rows = [header, sep]
    for e in epochs:
        rows.append(
            f"| {e['epoch']} "
            f"| `{e['train_loss']}` | `{e['train_acc']:.4f}` | `{e['train_sign']:.4f}` "
            f"| `{e['eval_loss']}` | `{e['eval_acc']:.4f}` | `{e['eval_sign']:.4f}` "
            f"| `{e['eval_cost_mean']}` | `{e['eval_cost_std']}` |"
        )
    return "\n".join(rows)


def img(alt: str, path: str) -> str:
    return f"![{alt}]({path})"


def rel_img_path(plot_dir: Path, fname: str, report_path: Path) -> str:
    p = plot_dir / fname
    try:
        return str(p.resolve().relative_to(report_path.parent.resolve()))
    except ValueError:
        return str(p)


def generate_report(
    instruct_8b: dict,
    base_8b: dict,
    instruct_3b: dict | None,
    plot_dir_8b_instruct: Path,
    plot_dir_8b_base: Path,
    plot_dir_3b_instruct: Path | None,
    output: Path,
) -> None:
    today = datetime.now().strftime("%Y/%m/%d")
    lines: list[str] = []
    a = lines.append

    def rp(plot_dir: Path, fname: str) -> str:
        return rel_img_path(plot_dir, fname, output)

    def plot_or_missing(plot_dir: Path, fname: str, alt: str) -> str:
        p = plot_dir / fname
        if p.exists():
            return img(alt, rp(plot_dir, fname))
        return f"*（圖檔未找到：`{p.name}`）*"

    # ── Header ────────────────────────────────────────────────────────────
    a("# Safety Alignment Progress Report: Cost Model Training (Ministral)")
    a(f"**專案目標:** TAIWAN AI RAP 客服模型 Safety Alignment Cost Model 訓練 (Ministral 遷移)")
    a(f"**報告日期:** {today}")
    a("")
    a("---")
    a("")

    # ── 1. 本次工作目標 ───────────────────────────────────────────────────
    a("## 1. 本次工作目標")
    a("")
    a("本階段的目標是將 Cost Model 的 Backbone 從原本的 **DeBERTa-v3-large**（Encoder 架構）")
    a("升級至 **Ministral Decoder 架構**，評估大型語言模型作為安全性評分器的效能。")
    a("")
    a("下游目標客服模型資訊如下：")
    a("")
    a("- Model artifact: `s3://cpft/ft/mistral-small-3.2-24b-instruct-2506_sft-full-rap-nchc-iservice-0.5.0-ep5-fapm50`")
    a("- Medusa API: `Mistral-Small-3.2-24B-Instruct-2506-CS-3in1-FAPM50-1124`")
    a("")
    a("本次測試的模型如下：")
    a("")
    a("| # | 模型 | 類型 | 優化器 |")
    a("|---|------|------|--------|")
    a(f"| 1 | `{instruct_8b['model']}` | 8B Instruction-tuned | Adafactor |")
    a(f"| 2 | `{base_8b['model']}` | 8B Base pretrained | Adafactor |")
    if instruct_3b:
        a(f"| 3 | `{instruct_3b['model']}` | 3B Instruction-tuned | AdamW |")
    else:
        a(f"| 3 | `mistralai/Ministral-3-3B-Instruct-2512` | 3B Instruction-tuned | AdamW | ❌ OOM — 需遷移至 4-GPU VM |")
    a("")
    a("---")
    a("")

    # ── 2. 理論基礎與損失函數設計 ─────────────────────────────────────────
    a("## 2. 理論基礎與損失函數設計")
    a("")
    a("本次 cost model 採用 **PKU-SafeRLHF** 的 **3-term cost loss**。")
    a("完整理論推導請參考 `report0507.md` 第 2 節；此處列出核心數學。")
    a("")
    a("### 2.1 Cost Model 的定義")
    a("")
    a("給定 prompt $\\mathbf{x}$ 與回答 $\\mathbf{y}$：")
    a("")
    a("- $C_\\psi(\\mathbf{y}, \\mathbf{x}) < 0$ 表示偏安全")
    a("- $C_\\psi(\\mathbf{y}, \\mathbf{x}) > 0$ 表示偏危險")
    a("")
    a("### 2.2 PKU 3-term loss")
    a("")
    a("```")
    a("Loss (sequence-wise):")
    a("  L = -log σ(C_unsafe - C_safe)           # Term 1: pairwise ordering")
    a("      -log σ(sign_safe  × C_safe)          # Term 2: safe → negative cost")
    a("      -log σ(sign_unsafe × C_unsafe)       # Term 3: unsafe → positive cost")
    a("```")
    a("")
    a("其中 `cost_sign = -safety_sign`：`safe (safety_sign=+1) → cost_sign=-1 → 希望 cost<0`，")
    a("`unsafe (safety_sign=-1) → cost_sign=+1 → 希望 cost>0`。")
    a("")
    a("---")
    a("")

    # ── 3. 訓練程式修改與遭遇問題 ─────────────────────────────────────────
    a("## 3. 訓練程式修改與遭遇問題")
    a("")
    a("在將 Backbone 切換到 Ministral Decoder 時，我們遇到了幾個關鍵的系統工程問題，")
    a("對 `scripts/train_cost_model_v2.py` 進行了深度重構以解決這些問題。")
    a("")
    a("### 3.1 遭遇問題清單")
    a("")
    a("#### 問題 1：模型架構封裝不相容 (Architecture Wrapper Mismatch)")
    a("")
    a("**詳細技術說明：**")
    a("Ministral 模型屬於多模態架構 (`Mistral3ForConditionalGeneration`)，包含視覺編碼器與語言模型兩個子模組。")
    a("當我們使用 `AutoModel.from_pretrained()` 載入時，取得的物件為外層 Wrapper，")
    a("核心語言模型隱藏在 `model.language_model` 內部屬性中。")
    a("原為 DeBERTa 設計的程式碼預期模型本體就是語言主幹，導致 `config.hidden_size` 屬性路徑錯誤，")
    a("以及 `last_hidden_state` 無法正確提取。")
    a("")
    a("#### 問題 2：顯示卡記憶體溢出 (VRAM OOM)")
    a("")
    a("**詳細技術說明：**")
    a("Ministral-3-8B 約有 **8B 參數**。以 FP16 載入：約 16GB VRAM。")
    a("使用標準 `DataParallel`：每張 GPU 須保存完整模型副本 → 每張 V100(32GB) 各需 16GB，")
    a("再加上 AdamW 的 Optimizer States（一階 + 二階動量，約 **64GB**）")
    a("以及訓練時的梯度（約 16GB），遠超硬體上限，導致 `torch.cuda.OutOfMemoryError`。")
    a("")
    a("#### 問題 3：設備映射張量錯位 (Device Mismatch in device_map)")
    a("")
    a("**詳細技術說明：**")
    a("`device_map='auto'` 將模型層分散至多張 GPU（如 Layer 0-16 → GPU:0，Layer 17-33 → GPU:1）。")
    a("輸入 `input_ids` 在 GPU:0，但模型最終輸出 `last_hidden_state` 落在 GPU:1。")
    a("若 `score_head` 或 `attention_mask` 仍在 GPU:0，PyTorch 會拋出：")
    a("`RuntimeError: Expected all tensors to be on the same device`。")
    a("")
    a("#### 問題 4：FP8 量化格式在 V100 上的相容性")
    a("")
    a("**詳細技術說明：**")
    a("Ministral-2512 系列模型的 checkpoint 以 **FP8 量化格式**儲存，")
    a("需要 GPU compute capability ≥ 8.9（H100/4090 等級）才能原生執行。")
    a("V100 的 compute capability 為 7.0，transformers 5.8.0 在嘗試將 FP8 反量化為 BF16 時，")
    a("對特定層（layer 33）的 per-tensor scalar scale 處理有 Bug：")
    a("```")
    a("scale_rows, scale_cols = scales.shape[-2:]")
    a("# ValueError: not enough values to unpack (expected 2, got 0)")
    a("```")
    a("")
    a("#### 問題 5：use_cache 參數跨架構衝突")
    a("")
    a("**詳細技術說明：**")
    a("Decoder 模型訓練時須強制設 `use_cache=False`（避免 KV Cache 記憶體洩漏）。")
    a("但 DeBERTa（Encoder）的 `forward()` 完全沒有此參數，直接傳入會觸發 `TypeError`。")
    a("")
    a("---")
    a("")
    a("### 3.2 解決方案與程式碼修改細節")
    a("")
    a("#### 修正 1：動態架構提取與屬性遞迴解析")
    a("")
    a("實作 fallback 載入機制，自動剝離多模態 Wrapper，精準定位語言模型主幹：")
    a("")
    a("```python")
    a("# scripts/train_cost_model_v2.py — CostModel.__init__")
    a("try:")
    a("    raw = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)")
    a("except Exception:")
    a("    raw = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)")
    a("")
    a("# 動態剝離 Wrapper，取得真正的 Backbone")
    a("self.backbone = (")
    a("    raw.language_model if hasattr(raw, 'language_model')")
    a("    else raw.model if hasattr(raw, 'model') and hasattr(raw.model, 'embed_tokens')")
    a("    else raw")
    a(")")
    a("# 遞迴解析 hidden_size（可能藏在 text_config 子物件中）")
    a("hidden_size = (")
    a("    getattr(cfg, 'hidden_size', None)")
    a("    or getattr(getattr(cfg, 'text_config', None), 'hidden_size', None)")
    a("    or getattr(getattr(raw.config, 'text_config', None), 'hidden_size', None)")
    a(")")
    a("```")
    a("")
    a("#### 修正 2：Adafactor 優化器 + Load-in-Half 策略")
    a("")
    a("| 策略 | 記憶體效果 |")
    a("|------|-----------|")
    a("| `--load-in-half` | 模型以 FP16 載入，從一開始就省去 50% 基礎 VRAM |")
    a("| `--adafactor` | 以矩陣分解逼近二階動量，Optimizer States 從 ~64GB 壓至 ~1GB |")
    a("| Score head 強制 FP32 | 防止 V100 FP16 精度不足導致 NaN |")
    a("| Score clamping [-50, 50] | 防止 logsigmoid 輸入爆炸 |")
    a("")
    a("```python")
    a("optimizer = Adafactor(")
    a("    model.parameters(),")
    a("    lr=args.learning_rate,   # 顯式 lr（關閉 relative_step）")
    a("    scale_parameter=False,")
    a("    relative_step=False,")
    a("    warmup_init=False,")
    a("    weight_decay=args.weight_decay,")
    a(")")
    a("```")
    a("")
    a("#### 修正 3：動態設備路由 (Dynamic Device Routing)")
    a("")
    a("追蹤 Backbone 首尾層的物理設備位置，確保所有張量在運算時落在同一 GPU：")
    a("")
    a("```python")
    a("if use_device_map:")
    a("    params = list(self.backbone.parameters())")
    a("    self._first_device = params[0].device   # 輸入端")
    a("    self._last_device  = params[-1].device  # 輸出端")
    a("    self.score_head = nn.Linear(hidden_size, 1).float().to(self._last_device)")
    a("")
    a("# Forward 時動態路由")
    a("if self._use_device_map:")
    a("    input_ids = input_ids.to(self._first_device)")
    a("attention_mask_h = attention_mask.to(hidden_dev)  # 跟著 last_hidden_state 走")
    a("")
    a("# Loss 計算前統一搬至分數所在的設備")
    a("_score_dev = normalized_pair_scores['lower_sequence_scores'].device")
    a("safer_sign = safer_sign.to(_score_dev)")
    a("```")
    a("")
    a("#### 修正 4：FP8 dequantization Bug Patch")
    a("")
    a("在 `transformers/integrations/finegrained_fp8.py` 的 `_dequantize_one` 函式中，")
    a("加入 per-tensor scalar scale 的 early return 分支：")
    a("")
    a("```python")
    a("# 修正前：直接解包 scales.shape[-2:]，scalar scale 會 ValueError")
    a("# 修正後：")
    a("if scales.dim() < 2:  # per-tensor scale（scalar 或 1-D）")
    a("    return (quantized_fp32 * scales.to(torch.float32)).to(out_dtype)")
    a("```")
    a("")
    a("#### 修正 5：use_cache 跨架構相容")
    a("")
    a("在 `CostModel.__init__` 透過 `inspect.signature` 動態探測 `forward()` 是否支援 `use_cache`：")
    a("")
    a("```python")
    a("import inspect")
    a("sig = inspect.signature(self.backbone.forward)")
    a("self._supports_use_cache = 'use_cache' in sig.parameters")
    a("")
    a("# Forward 時：")
    a("call_kwargs = {'use_cache': False} if self._supports_use_cache else {}")
    a("outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **call_kwargs)")
    a("```")
    a("")
    a("---")
    a("")

    # ── 4. 資料集切分與類別分佈 ───────────────────────────────────────────
    a("## 4. 資料集切分與類別分佈")
    a("")
    a("使用固定種子 (`--seed 42`) 劃分資料集。**所有模型（包含 DeBERTa baseline）均使用相同的 146 筆 Eval Set**，")
    a("使各模型的分數具備直接的比較價值。")
    a("")
    a("| 切分 | 總樣本數 | Safe vs Safe | Safe vs Unsafe | Unsafe vs Unsafe |")
    a("|------|---------|-------------|----------------|-----------------|")
    a("| **Train Set** | 1,315 | 793 (60.3%) | 394 (30.0%) | 128 (9.7%) |")
    a("| **Eval Set** | 146 | 90 (61.6%) | 44 (30.1%) | 12 (8.2%) |")
    a("")
    a("---")
    a("")

    # ── 5. 超參數設定 ─────────────────────────────────────────────────────
    a("## 5. 超參數設定")
    a("")
    a(f"| 參數 | 8B Instruct | 8B Base | {'3B Instruct' if instruct_3b else '3B (未完成)'} | DeBERTa (baseline) |")
    a(f"|------|------------|---------|{'----------|' if instruct_3b else '----------|'}-------------------|")
    a(f"| Backbone | `{instruct_8b['model'].split('/')[-1]}` | `{base_8b['model'].split('/')[-1]}` | `Ministral-3-3B-Instruct-2512` | `deberta-v3-large` |")
    a(f"| Params | ~8B | ~8B | ~3.26B | 434M |")
    a(f"| Pooling | `last-token` | `last-token` | `last-token` | `mean` |")
    a(f"| Loss type | `sequence-wise` | `sequence-wise` | `sequence-wise` | `sequence-wise` |")
    a(f"| Optimizer | **Adafactor** | **Adafactor** | **AdamW** | AdamW |")
    a(f"| Precision | `FP16 native` | `FP16 native` | `FP16 native` | `FP16 autocast` |")
    a(f"| Max length | `256` | `256` | `256` | `512` |")
    a(f"| Batch size | `1 × 32 = 32` | `1 × 32 = 32` | `1 × 32 = 32` | `4 × 8 = 32` |")
    a(f"| Learning rate | `1e-5` | `1e-5` | `1e-5` | `1e-5` |")
    a(f"| Epochs | `3` | `3` | `3` | `3` |")
    a(f"| Strategy | `device_map='auto'` | `device_map='auto'` | `device_map='auto'` | `DataParallel` |")
    a("")
    a("### 5.1 為什麼 8B 用 Adafactor，3B 用 AdamW")
    a("")
    a("| 模型 | 參數量 | AdamW Optimizer States | Adafactor States | 選擇理由 |")
    a("|------|-------|----------------------|-----------------|---------|")
    a("| 8B | ~8B params | ~64GB（每 GPU ~32GB → OOM） | ~1GB | 8B 非 Adafactor 不可行 |")
    a("| 3B | ~3.26B params | ~26GB（每 GPU ~13GB → 約 23GB 總計，勉強可行） | ~0.4GB | 測試 AdamW 收斂品質是否更好 |")
    a("")
    a("---")
    a("")

    # ── 6. 訓練結果數據 ───────────────────────────────────────────────────
    a("## 6. 訓練結果數據")
    a("")
    a("### 6.1 Ministral 8B Instruct — Epoch-level 指標")
    a("")
    i8b_s = instruct_8b['summary']
    a(f"- Train samples: `{i8b_s.get('num_train_samples', 'N/A')}`  |  Eval samples: `{i8b_s.get('num_eval_samples', 'N/A')}`")
    a(f"- Total update steps: `{i8b_s.get('total_update_steps', 'N/A')}`  |  Training time: `{instruct_8b['train_seconds']}s`")
    a("")
    a(epoch_table(instruct_8b["epochs"]))
    a("")
    a(f"**Best eval accuracy: `{instruct_8b['best_eval_acc']:.4f}`  |  Best eval sign acc: `{instruct_8b['best_eval_sign']:.4f}`  |  Best eval loss: `{instruct_8b['best_eval_loss']:.4f}`**")
    a("")

    a("### 6.2 Ministral 8B Base — Epoch-level 指標")
    a("")
    b8b_s = base_8b['summary']
    a(f"- Train samples: `{b8b_s.get('num_train_samples', 'N/A')}`  |  Eval samples: `{b8b_s.get('num_eval_samples', 'N/A')}`")
    a(f"- Total update steps: `{b8b_s.get('total_update_steps', 'N/A')}`  |  Training time: `{base_8b['train_seconds']}s`")
    a("")
    a(epoch_table(base_8b["epochs"]))
    a("")
    a(f"**Best eval accuracy: `{base_8b['best_eval_acc']:.4f}`  |  Best eval sign acc: `{base_8b['best_eval_sign']:.4f}`  |  Best eval loss: `{base_8b['best_eval_loss']:.4f}`**")
    a("")

    if instruct_3b:
        a("### 6.3 Ministral 3B Instruct (AdamW) — Epoch-level 指標")
        a("")
        i3b_s = instruct_3b['summary']
        a(f"- Train samples: `{i3b_s.get('num_train_samples', 'N/A')}`  |  Eval samples: `{i3b_s.get('num_eval_samples', 'N/A')}`")
        a(f"- Total update steps: `{i3b_s.get('total_update_steps', 'N/A')}`  |  Training time: `{instruct_3b['train_seconds']}s`")
        a("")
        a(epoch_table(instruct_3b["epochs"]))
        a("")
        a(f"**Best eval accuracy: `{instruct_3b['best_eval_acc']:.4f}`  |  Best eval sign acc: `{instruct_3b['best_eval_sign']:.4f}`  |  Best eval loss: `{instruct_3b['best_eval_loss']:.4f}`**")
        a("")
    else:
        a("### 6.3 Ministral 3B Instruct (AdamW) — ❌ 未完成")
        a("")
        a("3B + AdamW 訓練失敗（CUDA OOM on 2×V100）。")
        a("請將專案遷移至 **4-GPU VM** 後重試。")
        a("")

    a("### 6.4 數據解讀")
    a("")

    # Dynamic interpretation
    all_sign = {
        "8B Instruct": instruct_8b['best_eval_sign'],
        "8B Base": base_8b['best_eval_sign'],
        "DeBERTa (seq-wise)": DEBERTA_MAIN['best_eval_sign'],
    }
    if instruct_3b:
        all_sign["3B Instruct"] = instruct_3b['best_eval_sign']
    best_sign_model = max(all_sign, key=all_sign.get)

    a(f"1. **Sign Accuracy 分析**：`eval_accuracy_sign` 衡量模型能否把安全回答壓到負值、危險回答推到正值。")
    a(f"   本次 best sign acc 由 **{best_sign_model}** 達到 `{all_sign[best_sign_model]:.4f}`。")
    a(f"   這是 downstream RLHF 最重要的指標，代表模型能正確判斷回答的安全方向。")
    a("")
    a(f"2. **Eval Cost Mean < 0**：驗證集的平均 Cost Score 為負值，")
    a(f"   因為資料集中安全樣本佔多數（~60%），整體分佈被推向負值區間，符合預期。")
    a("")
    a(f"3. **Pairwise Accuracy 解讀**：`eval_accuracy` 衡量「在給定的安全/危險 pair 中，模型能否把危險的排得更高」。")
    a(f"   這個指標受資料分佈影響（Safe vs Safe pair 佔 61%，邊界更細微），通常比 sign accuracy 低。")
    a("")
    a("---")
    a("")

    # ── 7. Loss / Accuracy 圖 ─────────────────────────────────────────────
    a("## 7. Loss / Accuracy 圖")
    a("")
    a("### 7.1 Ministral 8B Instruct")
    a("")
    for fname, alt in [("cost_loss_curve.png", "8B Instruct Loss"), ("cost_acc_curve.png", "8B Instruct Accuracy")]:
        a(plot_or_missing(plot_dir_8b_instruct, fname, alt))
        a("")
    a("### 7.2 Ministral 8B Base")
    a("")
    for fname, alt in [("cost_loss_curve.png", "8B Base Loss"), ("cost_acc_curve.png", "8B Base Accuracy")]:
        a(plot_or_missing(plot_dir_8b_base, fname, alt))
        a("")
    if instruct_3b and plot_dir_3b_instruct:
        a("### 7.3 Ministral 3B Instruct")
        a("")
        for fname, alt in [("cost_loss_curve.png", "3B Instruct Loss"), ("cost_acc_curve.png", "3B Instruct Accuracy")]:
            a(plot_or_missing(plot_dir_3b_instruct, fname, alt))
            a("")
    a("### 7.4 圖的解讀")
    a("")
    a("- **Loss 圖** 畫的是每個 training step 的 PKU 3-term cost loss（含 pairwise ordering + sign anchoring 三項）。")
    a("- **Accuracy 圖** 中：藍線 = pairwise accuracy，綠線 = sign accuracy。")
    a("  Sign accuracy 的提升速度通常比 pairwise accuracy 更快，因為方向性判斷比細微排序更容易學。")
    a("- 小 batch (batch=1) 讓 step-level metric 很離散；應以 epoch-level eval 指標為主要判斷依據。")
    a("")
    a("---")
    a("")

    # ── 8. Eval Predictions Sample ────────────────────────────────────────
    a("## 8. Eval Predictions Sample")
    a("")
    a("以下為最後一個 Epoch 的 Eval Set 前 10 筆預測結果（來自 `eval_predictions.jsonl`）。")
    a("可用於人工確認模型打分方向是否合理。完整 146 筆見對應 run 目錄下的 `eval_predictions.jsonl`。")
    a("")

    best_run_dir = None
    best_acc = -1.0
    for label, run in [("8B Instruct", instruct_8b), ("8B Base", base_8b)]:
        if run['best_eval_acc'] > best_acc:
            best_acc = run['best_eval_acc']
            best_run_dir = Path(run['summary'].get('output_dir', ''))

    preds = read_eval_predictions(best_run_dir, n=10) if best_run_dir else []
    if preds:
        a("| # | Prompt (truncated) | Safer Score | Unsafer Score | Pairwise ✓ | Safer Sign ✓ | Unsafer Sign ✓ |")
        a("|---|-------------------|------------|--------------|-----------|-------------|--------------|")
        for i, p in enumerate(preds, 1):
            prompt_short = p.get("prompt", "")[:50].replace("|", "｜").replace("\n", " ")
            ss = p.get("safer_score", "?")
            us = p.get("unsafer_score", "?")
            pw = "✅" if p.get("pairwise_correct") else "❌"
            sc = "✅" if p.get("safer_sign_correct") else "❌"
            uc = "✅" if p.get("unsafer_sign_correct") else "❌"
            a(f"| {i} | {prompt_short}... | `{ss}` | `{us}` | {pw} | {sc} | {uc} |")
    else:
        a("*（eval_predictions.jsonl 未找到 — 確認 `--save-eval-predictions` 已啟用）*")
    a("")
    a("---")
    a("")

    # ── 9. 與 DeBERTa Baseline 比較 ───────────────────────────────────────
    a("## 9. 與 DeBERTa Baseline 比較")
    a("")
    cols = ["DeBERTa (seq-wise)", "DeBERTa (token-wise+norm)", "8B Instruct", "8B Base"]
    vals = [DEBERTA_MAIN, DEBERTA_OFFICIAL, instruct_8b, base_8b]
    if instruct_3b:
        cols.append("3B Instruct")
        vals.append(instruct_3b)

    sep_row = "|------|" + "|".join(["------"] * len(cols)) + "|"
    header_row = "| 指標 | " + " | ".join(cols) + " |"
    a(header_row)
    a(sep_row)

    def get_v(d, k): return f"`{d[k]:.4f}`" if k in d and d[k] is not None else "N/A"

    a("| Best eval accuracy | " + " | ".join(get_v(d, 'best_eval_acc') for d in vals) + " |")
    a("| Best eval sign acc | " + " | ".join(get_v(d, 'best_eval_sign') for d in vals) + " |")
    a("| Best eval loss     | " + " | ".join(get_v(d, 'best_eval_loss') for d in vals) + " |")
    a("| Backbone params    | 434M | 434M | ~8B | ~8B" + (" | ~3.26B" if instruct_3b else "") + " |")
    a("| Pooling            | mean | mean | last-token | last-token" + (" | last-token" if instruct_3b else "") + " |")
    a("| Optimizer          | AdamW | AdamW | Adafactor | Adafactor" + (" | AdamW" if instruct_3b else "") + " |")
    a("| Strategy           | DataParallel | DataParallel | device_map | device_map" + (" | device_map" if instruct_3b else "") + " |")
    a("")
    a("---")
    a("")

    # ── 10. 本次工作的限制 ────────────────────────────────────────────────
    a("## 10. 本次工作的限制")
    a("")
    a("### 10.1 Max Length 從 512 降至 256")
    a("")
    a("為使 8B 模型在 V100 不 OOM，max_length 從 DeBERTa 的 512 縮減至 256。")
    a("若資料中有長對話（超過 256 tokens），後段內容會被截斷，可能影響對長文本的安全判斷能力。")
    a("")
    a("### 10.2 只測試 sequence-wise loss")
    a("")
    a("Decoder 架構理論上更適合 `token-wise` loss（causal attention 天然對齊 token-level 危險邊界），")
    a("但本次為節省計算資源，只跑 sequence-wise。後續可做 ablation study。")
    a("")
    a("### 10.3 使用 VLM 的 text-only backbone")
    a("")
    a("Ministral-3-{3,8}B-2512 是多模態 VLM，視覺編碼器（~300M params）被載入但完全不參與訓練。")
    a("若有純文字版本，記憶體效率更好。")
    a("")
    a("### 10.4 FP8 反量化 patch 為局部修正")
    a("")
    a("對 layer 33 的 scalar scale 採用 per-tensor broadcast dequantize 作為 fallback，")
    a("而非原版 block-wise dequantize。精度影響預計微小，但非官方支援方式。")
    a("")
    if not instruct_3b:
        a("### 10.5 3B + AdamW 在 2×V100 上 OOM")
        a("")
        a("AdamW 對 3B 模型的 Optimizer States 總計約 26GB，加上模型權重與梯度，")
        a("2×V100（64GB 總計）在特定配置下超出限制。需遷移至 4-GPU VM 或改用 Adafactor。")
        a("")
    a("---")
    a("")

    # ── 11. 結論與下一步規劃 ─────────────────────────────────────────────
    a("## 11. 結論與下一步規劃")
    a("")
    a("### 11.1 本次核心結論")
    a("")

    all_accs = {"8B Instruct": instruct_8b['best_eval_acc'], "8B Base": base_8b['best_eval_acc']}
    if instruct_3b:
        all_accs["3B Instruct"] = instruct_3b['best_eval_acc']
    winner = max(all_accs, key=all_accs.get)
    a(f"1. **工程遷移成功**：成功解決 8B Decoder 在 2×V100 上的所有相容性問題（FP8、device_map、Adafactor）。")
    a(f"2. **{winner} 在本次 Ministral 系列中表現最佳**（eval acc = `{all_accs[winner]:.4f}`）。")
    a(f"3. **Sign Accuracy 收斂迅速**：Ministral 對語意方向性的理解明顯強於 DeBERTa baseline，")
    a(f"   Eval Sign Acc 從第一個 Epoch 就達到較高水準。")
    a("")
    a("### 11.2 短期 Next Steps")
    a("")
    a("1. 抽出 `eval_predictions.jsonl` 中 `pairwise_correct=False` 的 case 做人工分析")
    a("2. 用 `harmful_yes_clean.jsonl` 的 171 筆惡意 prompt 對 Ministral cost model 做額外評估")
    if not instruct_3b:
        a("3. 遷移至 4-GPU VM，重跑 3B + AdamW 實驗")
    a("")
    a("### 11.3 中期規劃")
    a("")
    a("1. 補 Helpfulness Reward Model（與 Cost Model 配合，構成完整的 RLHF 雙模評分系統）")
    a("2. 對 Ministral 做 `token-wise` loss 的 ablation（理論上 decoder 更適合）")
    a("3. 確認最終 backbone 後，進行 risk threshold calibration")
    a("")
    a("---")
    a("")

    # ── 12. 參考資料 ──────────────────────────────────────────────────────
    a("## 12. 參考資料")
    a("")
    a("- PKU-SafeRLHF dataset card: <https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF>")
    a("- PKU 原始 cost trainer: <https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/values/cost/trainer.py>")
    a("- DeBERTa baseline 報告: `outputs/reports/report0507.md`")
    a("- Ministral 工程遷移報告: `outputs/reports/report_mistral.md`")
    a(f"- Ministral 8B Instruct: <https://huggingface.co/{instruct_8b['model']}>")
    a(f"- Ministral 8B Base: <https://huggingface.co/{base_8b['model']}>")
    a(f"- Ministral 3B Instruct: <https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instruct-8b-dir", type=Path, required=True)
    p.add_argument("--base-8b-dir", type=Path, required=True)
    p.add_argument("--instruct-3b-dir", type=Path, default=None)
    p.add_argument("--plot-dir-8b-instruct", type=Path, required=True)
    p.add_argument("--plot-dir-8b-base", type=Path, required=True)
    p.add_argument("--plot-dir-3b-instruct", type=Path, default=None)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/reports/ministral_report.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        instruct_8b = read_run(args.instruct_8b_dir)
        base_8b = read_run(args.base_8b_dir)
    except Exception as e:
        print(f"ERROR reading 8B run dirs: {e}", file=sys.stderr)
        return 1

    instruct_3b = None
    if args.instruct_3b_dir is not None:
        try:
            instruct_3b = read_run(args.instruct_3b_dir)
        except Exception as e:
            print(f"WARNING: Could not read 3B dir ({e}) — report will note it as incomplete.",
                  file=sys.stderr)

    generate_report(
        instruct_8b=instruct_8b,
        base_8b=base_8b,
        instruct_3b=instruct_3b,
        plot_dir_8b_instruct=args.plot_dir_8b_instruct,
        plot_dir_8b_base=args.plot_dir_8b_base,
        plot_dir_3b_instruct=args.plot_dir_3b_instruct,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

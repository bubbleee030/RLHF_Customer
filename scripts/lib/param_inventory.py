#!/usr/bin/env python3
"""Regenerate the CM/RM parameter tables in docs/parameters/ from source.

Parses each trainer's argparse block with `ast` (no torch import needed) and joins it
against the shell wrapper values and each historical run's arguments.json. Run this after
changing either trainer's argparse block so the inventory cannot drift from the code.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DESCRIPTIONS = {
    "model_name_or_path": "HuggingFace id or local path of the backbone to attach the score head to.",
    "dataset_path": "Training data (JSONL). Must be train-only when --eval-dataset-path is also given; the trainer does not subtract eval rows.",
    "eval_dataset_path": "External evaluation split. When set, the trainer skips its own internal split.",
    "output_dir": "Run directory: checkpoints, arguments.json, training_log.json, run_manifest.json.",
    "max_length": "Tokenizer truncation length. Dominates VRAM; the single biggest cost knob.",
    "batch_size": "Micro-batch per optimizer step. Effective batch = batch_size x gradient_accumulation_steps.",
    "gradient_accumulation_steps": "Steps accumulated before an optimizer update.",
    "learning_rate": "Peak LR after warmup.",
    "weight_decay": "AdamW weight decay.",
    "epochs": "Full passes over the training set.",
    "warmup_ratio": "Fraction of total steps spent warming the LR up from 0.",
    "regularization": "L2 penalty on raw score magnitude, keeping outputs from drifting.",
    "eval_split_ratio": "Fallback internal holdout fraction, used only when --eval-dataset-path is absent.",
    "log_steps": "Optimizer steps between log lines.",
    "seed": "Seed for shuffling, split, and init.",
    "pooling": "How token states collapse to one score. last-token is what production uses.",
    "loss_type": "sequence-wise scores the whole sequence once; token-wise averages per-token scores.",
    "concat_forward": "Run the chosen and rejected halves of a pair in one forward pass.",
    "normalize_score_during_training": "Track a running mean/var and normalize the emitted score.",
    "normalizer_momentum": "EMA momentum for that running normalizer.",
    "gradient_checkpointing": "Trade compute for VRAM by recomputing activations in the backward pass.",
    "fp16": "Half-precision autocast. CM only; not present on the RM trainer.",
    "bf16": "bfloat16 autocast. Preferred over fp16 where supported.",
    "load_in_half": "Load backbone weights already in half precision, halving load-time RAM.",
    "device_map": "Let accelerate shard the model across visible GPUs (device_map=auto).",
    "adafactor": "Use Adafactor instead of AdamW to cut optimizer state memory.",
    "save_eval_predictions": "Write per-example eval scores for error analysis.",
    "save_backbone": "Persist full backbone weights. --no-save-backbone keeps only the adapter + score head.",
    "save_best": "Comma-separated metrics to track a best-* checkpoint for (CM).",
    "save_best_only": "Keep only the best checkpoint instead of every epoch (RM).",
    "save_each_epoch": "Write a checkpoint after every epoch (RM).",
    "lora_r": "LoRA rank. 0 disables LoRA entirely and the run becomes full fine-tuning.",
    "lora_alpha": "LoRA scaling factor; effective scale is alpha/r.",
    "lora_dropout": "Dropout on the LoRA path.",
    "early_stopping_patience": "Epochs without improvement before stopping. 0 disables early stopping.",
    "early_stopping_min_delta": "Minimum improvement that counts as progress for early stopping.",
}

DESCRIPTIONS_ZH = {
    "model_name_or_path": "作為骨幹（backbone）的 HuggingFace 模型 id 或本地路徑，score head 會接在其上。",
    "dataset_path": "訓練資料（JSONL）。同時給定 --eval-dataset-path 時，此檔必須只含訓練資料；trainer 不會自動扣除 eval 列。",
    "eval_dataset_path": "外部評估集。設定後 trainer 會跳過自身的內部切分。",
    "output_dir": "執行目錄：checkpoints、arguments.json、training_log.json、run_manifest.json。",
    "max_length": "tokenizer 截斷長度。主導 VRAM 用量，是成本影響最大的單一參數。",
    "batch_size": "每個 optimizer step 的 micro-batch。有效批次 = batch_size x gradient_accumulation_steps。",
    "gradient_accumulation_steps": "累積多少 step 後才更新一次 optimizer。",
    "learning_rate": "warmup 之後的峰值學習率。",
    "weight_decay": "AdamW 的 weight decay。",
    "epochs": "完整走過訓練集的次數。",
    "warmup_ratio": "總步數中用於將學習率自 0 暖身的比例。",
    "regularization": "對原始分數量值的 L2 懲罰，避免輸出漂移。",
    "eval_split_ratio": "後備的內部保留比例，僅在未提供 --eval-dataset-path 時生效。",
    "log_steps": "每隔幾個 optimizer step 輸出一行紀錄。",
    "seed": "shuffle、切分與初始化所用的亂數種子。",
    "pooling": "如何將 token 狀態壓成單一分數。production 使用 last-token。",
    "loss_type": "sequence-wise 對整段序列給一次分數；token-wise 則平均每個 token 的分數。",
    "concat_forward": "將 pair 的 chosen 與 rejected 兩半併在同一次 forward 計算。",
    "normalize_score_during_training": "追蹤 running mean/var 並對輸出分數做正規化。",
    "normalizer_momentum": "上述 running normalizer 的 EMA 動量。",
    "gradient_checkpointing": "在 backward 時重算 activation，以計算量換取 VRAM。",
    "fp16": "半精度 autocast。僅 CM 有，RM trainer 沒有此參數。",
    "bf16": "bfloat16 autocast。在支援的硬體上優於 fp16。",
    "load_in_half": "直接以半精度載入骨幹權重，載入時記憶體需求減半。",
    "device_map": "交由 accelerate 將模型切分到可見的多張 GPU（device_map=auto）。",
    "adafactor": "改用 Adafactor 取代 AdamW，以降低 optimizer state 記憶體。",
    "save_eval_predictions": "輸出每筆樣本的 eval 分數，供錯誤分析。",
    "save_backbone": "保存完整骨幹權重。--no-save-backbone 則只留 adapter 與 score head。",
    "save_best": "以逗號分隔的指標清單，據以追蹤 best-* checkpoint（CM）。",
    "save_best_only": "只保留最佳 checkpoint，而非每個 epoch 都存（RM）。",
    "save_each_epoch": "每個 epoch 結束後寫出一個 checkpoint（RM）。",
    "lora_r": "LoRA rank。設為 0 會完全停用 LoRA，該次執行即成為 full fine-tuning。",
    "lora_alpha": "LoRA 縮放係數；實際縮放為 alpha/r。",
    "lora_dropout": "LoRA 路徑上的 dropout。",
    "early_stopping_patience": "容忍幾個 epoch 沒有進步才停止。0 表示停用 early stopping。",
    "early_stopping_min_delta": "要被視為有進步所需的最小改善幅度。",
}

CM = {
    "source": "scripts/train_cost_model_v2.py",
    "wrapper": {
        "model_name_or_path": "$MODEL", "dataset_path": "$DATASET", "output_dir": "$RUN_DIR",
        "max_length": "$MAX_LENGTH", "batch_size": "$BATCH_SIZE",
        "gradient_accumulation_steps": "$GRAD_ACCUM", "learning_rate": "$LR",
        "weight_decay": "1e-6", "regularization": "0.001", "epochs": "$EPOCHS",
        "warmup_ratio": "0.1", "eval_split_ratio": "0.1", "log_steps": "10", "seed": "$SEED",
        "pooling": "last-token", "loss_type": "sequence-wise", "load_in_half": "set",
        "device_map": "set", "save_eval_predictions": "set", "save_backbone": "set",
        "lora_r": "$LORA_R", "lora_alpha": "$LORA_ALPHA", "lora_dropout": "$LORA_DROPOUT",
    },
    "runs": {
        "run 20260625 (best)":
            "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/arguments.json",
    },
}

RM = {
    "source": "scripts/reward/train_reward_model.py",
    "wrapper": {
        "model_name_or_path": "$MODEL", "dataset_path": "$TRAIN", "eval_dataset_path": "$EVAL",
        "output_dir": "$RUN_DIR", "max_length": "$MAX_LENGTH", "batch_size": "$BATCH_SIZE",
        "gradient_accumulation_steps": "$GRAD_ACCUM", "learning_rate": "$LR",
        "weight_decay": "$WEIGHT_DECAY", "regularization": "$REGULARIZATION",
        "epochs": "$EPOCHS", "warmup_ratio": "0.1", "log_steps": "10", "seed": "$SEED",
        "pooling": "last-token", "load_in_half": "set", "device_map": "set",
        "save_each_epoch": "set", "save_backbone": "set", "lora_r": "$LORA_R",
        "lora_alpha": "$LORA_ALPHA", "lora_dropout": "$LORA_DROPOUT",
        "early_stopping_patience": "$EARLY_STOPPING_PATIENCE",
    },
    "runs": {
        "June (helpfulness)":
            "reward_output/run_reward_byprompt_20260622_104203/arguments.json",
        "Aug (customer CS)":
            "reward_output/run_reward_cs_within_20260803/arguments.json",
    },
}


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def extract_arguments(path: Path) -> list[dict]:
    """Pull every add_argument() call out of a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        flags = [v for v in (_literal(a) for a in node.args) if isinstance(v, str)]
        long = next((f for f in flags if f.startswith("--")), None)
        if long is None:
            continue
        kwargs = {kw.arg: _literal(kw.value) for kw in node.keywords}
        found.append({
            "flag": long,
            "attr": kwargs.get("dest") or long[2:].replace("-", "_"),
            "type": kwargs.get("type", ""),
            "default": kwargs.get("default"),
            "required": kwargs.get("required", False),
            "action": kwargs.get("action", ""),
            "choices": kwargs.get("choices"),
            "help": " ".join(str(kwargs.get("help", "")).split()),
        })
    return found


def _type_label(arg: dict) -> str:
    if arg["action"] == "store_true":
        return "flag"
    if "BooleanOptionalAction" in str(arg["action"]):
        return "flag (--x/--no-x)"
    return {"int": "int", "float": "float", "str": "str", "Path": "path"}.get(
        str(arg["type"]), str(arg["type"]) or "str"
    )


def _default_label(arg: dict) -> str:
    if arg["required"]:
        return "**required**"
    if arg["action"] == "store_true":
        return "`False`"
    return f"`{arg['default']}`"


def render_table(spec: dict, lang: str = "en") -> str:
    args = extract_arguments(REPO_ROOT / spec["source"])
    runs = {}
    for name, rel in spec["runs"].items():
        try:
            runs[name] = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runs[name] = {}
    run_names = list(spec["runs"])

    descriptions = DESCRIPTIONS_ZH if lang == "zh" else DESCRIPTIONS
    labels = (["參數", "型別", "argparse 預設", "wrapper 設定"] + run_names + ["用途"]) if lang == "zh" \
        else (["Flag", "Type", "argparse default", "Wrapper sets"] + run_names + ["Purpose"])
    header = labels
    rows = ["| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|"]
    for arg in args:
        cells = [f"`{arg['flag']}`", _type_label(arg), _default_label(arg)]
        wrapper = spec["wrapper"].get(arg["attr"], "—")
        cells.append(f"`{wrapper}`" if wrapper != "—" else "—")
        for name in run_names:
            value = runs[name].get(arg["attr"], "—")
            cells.append(f"`{value}`" if value != "—" else "—")
        # EN keeps the trainer's own help= text when present; zh always uses the
        # translated description, since help= is English.
        purpose = (descriptions.get(arg["attr"]) or arg["help"]) if lang == "zh" \
            else (arg["help"] or descriptions.get(arg["attr"], ""))
        if arg["choices"]:
            purpose = f"{purpose} Choices: {arg['choices']}.".strip()
        cells.append(purpose.replace("|", "\\|") or "—")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["cm", "rm", "both"], default="both")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    for name, spec in (("cm", CM), ("rm", RM)):
        if args.which in (name, "both"):
            print(f"### {name.upper()} — {spec['source']}\n")
            print(render_table(spec, args.lang))
            print()


if __name__ == "__main__":
    main()

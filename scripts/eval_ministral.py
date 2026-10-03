#!/usr/bin/env python3
"""
Run Ministral 3B cost model eval (pre-trained backbone + trained score_head).

Note: the fine-tuned backbone was not saved; this uses the original HuggingFace
pre-trained backbone + the trained score_head.pt. Expected to show degraded
performance vs. training-time metrics (backbone hidden states not aligned).

Usage:
    cd /home/ubuntu/reward_model
    python3 scripts/eval_ministral.py \
        --run-dir cost_output/run_ministral_3b_instruct_20260518_120434 \
        --eval-file datasets/eval_dataset.jsonl \
        --output results/eval_original_ministral3b_pretrained.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_model(run_dir: Path):
    args_path = run_dir / "arguments.json"
    with open(args_path) as f:
        args = json.load(f)

    model_id = args["model_name_or_path"]  # mistralai/Ministral-3-3B-Instruct-2512
    pooling = args.get("pooling", "last-token")
    max_length = int(args.get("max_length", 256))

    print(f"Loading tokenizer from {run_dir}...")
    tok = AutoTokenizer.from_pretrained(str(run_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Prefer the fine-tuned backbone saved in the run dir (after the save-bug fix).
    # Fall back to the pretrained HF backbone only if no saved weights exist.
    saved_backbone = (run_dir / "config.json").exists() and (
        (run_dir / "model.safetensors").exists()
        or (run_dir / "model.safetensors.index.json").exists()
        or (run_dir / "pytorch_model.bin").exists()
    )
    if saved_backbone:
        print(f"Loading FINE-TUNED backbone from {run_dir} (device={device})...")
        # Trained/saved in bf16 (the FP8 cache dequantizes to bf16); load bf16 to match.
        backbone = AutoModel.from_pretrained(str(run_dir), dtype=torch.bfloat16)
    else:
        print(f"Loading pre-trained backbone from HF cache: {model_id} (device={device})...")
        print("  (⚠️  no fine-tuned backbone saved — using original pre-trained weights)")
        backbone = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
    backbone = backbone.to(device).eval()

    # Ministral config nests hidden_size under text_config
    cfg = backbone.config
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    )
    if not hidden_size:
        raise ValueError("Cannot determine hidden_size from model config.")
    print(f"  hidden_size={hidden_size}, pooling={pooling}")

    score_head_path = run_dir / "score_head.pt"
    print(f"Loading score head from {score_head_path}...")
    score_head = nn.Linear(hidden_size, 1).float()
    state = torch.load(score_head_path, map_location="cpu", weights_only=True)
    score_head.load_state_dict(state)
    score_head.eval()

    status = "fine-tuned backbone" if saved_backbone else "⚠️  pre-trained backbone — expect degraded performance"
    print(f"Model ready ({status}).\n")
    return backbone, score_head, tok, pooling, max_length


@torch.no_grad()
def score_pair(backbone, score_head, tok, prompt: str, response: str,
               pooling: str, max_length: int, truncate: bool = True) -> float:
    text = f"User: {prompt}\nAssistant: {response}"
    if truncate:
        enc = tok(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
    else:
        # Feed the full input (no truncation). The backbone supports max_position_embeddings
        # far beyond any sample here, so this is safe.
        enc = tok(text, truncation=False, padding=False, return_tensors="pt")
    dev = next(backbone.parameters()).device
    out = backbone(input_ids=enc["input_ids"].to(dev),
                   attention_mask=enc["attention_mask"].to(dev), use_cache=False)
    hidden = out.last_hidden_state.float().cpu()
    enc = {k: v.cpu() for k, v in enc.items()}

    if pooling == "last-token":
        idx = enc["attention_mask"].long().sum() - 1
        pooled = hidden[0, idx.item()]
    else:
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1)
        pooled = pooled[0]

    return score_head(pooled.float()).item()


def eval_dataset(backbone, score_head, tok, pooling: str, max_length: int,
                 eval_file: Path, truncate: bool = True) -> dict:
    with open(eval_file) as f:
        rows = [json.loads(l) for l in f]

    print(f"Truncation: {('ON (max_length=%d)' % max_length) if truncate else 'OFF (full input, no truncation)'}")

    correct_pair = correct_sign_safer = correct_sign_unsafer = total = 0
    predictions = []
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        prompt = row["input"]
        ans, other = row["answer"], row["other_answer"]
        is_safe_ans = row.get("is_safe", True)
        is_safe_other = row.get("is_other_safe", True)

        if row.get("safer", True):
            safer, unsafer = ans, other
            is_safe_safer, is_safe_unsafer = is_safe_ans, is_safe_other
        else:
            safer, unsafer = other, ans
            is_safe_safer, is_safe_unsafer = is_safe_other, is_safe_ans

        s_safer = score_pair(backbone, score_head, tok, prompt, safer, pooling, max_length, truncate)
        s_unsafer = score_pair(backbone, score_head, tok, prompt, unsafer, pooling, max_length, truncate)

        pair_ok = s_safer < s_unsafer
        cost_sign_safer = -1 if is_safe_safer else +1
        cost_sign_unsafer = -1 if is_safe_unsafer else +1
        sign_safer_ok = (cost_sign_safer * s_safer) > 0
        sign_unsafer_ok = (cost_sign_unsafer * s_unsafer) > 0

        correct_pair += int(pair_ok)
        correct_sign_safer += int(sign_safer_ok)
        correct_sign_unsafer += int(sign_unsafer_ok)
        total += 1

        elapsed = time.time() - t0
        eta = elapsed / i * (len(rows) - i)
        marker = "✅" if pair_ok else "❌"
        print(f"[{i:3d}/{len(rows)}] {marker}  safer={s_safer:+.3f}  unsafer={s_unsafer:+.3f}  "
              f"gap={s_unsafer-s_safer:+.3f}  ETA={eta:.0f}s  {prompt[:40]}...")

        predictions.append({
            "_eval_idx": row.get("_eval_idx", i - 1),
            "s_safer": s_safer, "s_unsafer": s_unsafer,
            "pair_ok": pair_ok,
            "sign_safer_ok": sign_safer_ok, "sign_unsafer_ok": sign_unsafer_ok,
        })

    pairwise_acc = correct_pair / total
    sign_avg = (correct_sign_safer + correct_sign_unsafer) / (2 * total)

    print(f"\n{'='*60}")
    print(f"總樣本數:           {total}")
    print(f"Pairwise Acc:       {correct_pair}/{total} = {pairwise_acc:.4f}")
    print(f"Sign Acc (safer):   {correct_sign_safer}/{total} = {correct_sign_safer/total:.4f}")
    print(f"Sign Acc (unsafer): {correct_sign_unsafer}/{total} = {correct_sign_unsafer/total:.4f}")
    print(f"Sign Acc (avg):     {sign_avg:.4f}")

    return {
        "total": total, "correct_pair": correct_pair,
        "pairwise_acc": pairwise_acc,
        "sign_acc_safer": correct_sign_safer / total,
        "sign_acc_unsafer": correct_sign_unsafer / total,
        "sign_acc_avg": sign_avg,
        "truncation": truncate,
        "max_length": max_length,
        "eval_seconds": time.time() - t0,
        "predictions": predictions,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--truncate", action=argparse.BooleanOptionalAction, default=True,
                   help="Truncate inputs to max_length (default: on). Use --no-truncate to feed the full input.")
    return p.parse_args()


def main():
    args = parse_args()
    backbone, score_head, tok, pooling, max_length = load_model(args.run_dir)
    results = eval_dataset(backbone, score_head, tok, pooling, max_length, args.eval_file, args.truncate)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至: {args.output}")


if __name__ == "__main__":
    main()

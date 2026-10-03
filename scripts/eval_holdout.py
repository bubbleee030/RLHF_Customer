#!/usr/bin/env python3
"""
Run DeBERTa cost model eval on a JSONL eval file and return metrics.

Usage:
    cd /home/ubuntu/reward_model
    python3 scripts/eval_holdout.py \
        --run-dir cost_output/run_deberta_v2_20260504_050830 \
        --eval-file datasets/eval_dataset_holdout_gemma4.jsonl \
        --output results/eval_holdout_deberta_gemma4.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CostModel(nn.Module):
    def __init__(self, backbone, hidden_size: int, pooling: str = "mean"):
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(hidden_size, 1)
        self.pooling = pooling

    @torch.no_grad()
    def score(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, E)
        if self.pooling == "last-token":
            idx = attention_mask.long().sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), idx]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        return self.score_head(pooled).squeeze(-1)


def load_model(run_dir: Path):
    # Load arguments to determine model config
    args_path = run_dir / "arguments.json"
    with open(args_path) as f:
        args = json.load(f)

    pooling = args.get("pooling", "mean")

    print(f"Loading tokenizer from {run_dir}...")
    tok = AutoTokenizer.from_pretrained(str(run_dir))

    print(f"Loading backbone from {run_dir}...")
    backbone = AutoModel.from_pretrained(str(run_dir), torch_dtype=torch.float32)
    hidden_size = backbone.config.hidden_size
    print(f"  hidden_size={hidden_size}, pooling={pooling}")

    model = CostModel(backbone, hidden_size, pooling)

    score_head_path = run_dir / "score_head.pt"
    print(f"Loading score head from {score_head_path}...")
    state = torch.load(score_head_path, map_location="cpu", weights_only=True)
    model.score_head.load_state_dict(state)

    model.eval()
    print("Model ready.\n")
    return model, tok, int(args.get("max_length", 512))


@torch.no_grad()
def score_pair(model, tok, prompt: str, response: str, max_length: int) -> float:
    text = f"User: {prompt}\nAssistant: {response}"
    enc = tok(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
    s = model.score(enc["input_ids"], enc["attention_mask"])
    return s.item()


def eval_dataset(model, tok, max_length: int, eval_file: Path) -> dict:
    with open(eval_file) as f:
        rows = [json.loads(l) for l in f]

    correct_pair = 0
    correct_sign_safer = 0
    correct_sign_unsafer = 0
    total = 0
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

        s_safer = score_pair(model, tok, prompt, safer, max_length)
        s_unsafer = score_pair(model, tok, prompt, unsafer, max_length)

        pair_ok = s_safer < s_unsafer
        cost_sign_safer = -1 if is_safe_safer else +1
        cost_sign_unsafer = -1 if is_safe_unsafer else +1
        sign_safer_ok = (cost_sign_safer * s_safer) > 0
        sign_unsafer_ok = (cost_sign_unsafer * s_unsafer) > 0

        correct_pair += int(pair_ok)
        correct_sign_safer += int(sign_safer_ok)
        correct_sign_unsafer += int(sign_unsafer_ok)
        total += 1

        marker = "✅" if pair_ok else "❌"
        elapsed = time.time() - t0
        eta = elapsed / i * (len(rows) - i)
        print(f"[{i:3d}/{len(rows)}] {marker}  safer={s_safer:+.3f}  unsafer={s_unsafer:+.3f}  "
              f"gap={s_unsafer-s_safer:+.3f}  ETA={eta:.0f}s  {prompt[:40]}...")

        predictions.append({
            "_eval_idx": row.get("_eval_idx", i - 1),
            "s_safer": s_safer,
            "s_unsafer": s_unsafer,
            "pair_ok": pair_ok,
            "sign_safer_ok": sign_safer_ok,
            "sign_unsafer_ok": sign_unsafer_ok,
            "_paraphrased": row.get("_paraphrased", None),
        })

    pairwise_acc = correct_pair / total
    sign_safer_acc = correct_sign_safer / total
    sign_unsafer_acc = correct_sign_unsafer / total
    sign_acc_avg = (correct_sign_safer + correct_sign_unsafer) / (2 * total)

    print(f"\n{'='*60}")
    print(f"總樣本數:           {total}")
    print(f"Pairwise Acc:       {correct_pair}/{total} = {pairwise_acc:.4f}")
    print(f"Sign Acc (safer):   {correct_sign_safer}/{total} = {sign_safer_acc:.4f}")
    print(f"Sign Acc (unsafer): {correct_sign_unsafer}/{total} = {sign_unsafer_acc:.4f}")
    print(f"Sign Acc (avg):     {sign_acc_avg:.4f}")

    return {
        "total": total,
        "correct_pair": correct_pair,
        "pairwise_acc": pairwise_acc,
        "sign_acc_safer": sign_safer_acc,
        "sign_acc_unsafer": sign_unsafer_acc,
        "sign_acc_avg": sign_acc_avg,
        "eval_seconds": time.time() - t0,
        "predictions": predictions,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model, tok, max_length = load_model(args.run_dir)
    results = eval_dataset(model, tok, max_length, args.eval_file)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至: {args.output}")


if __name__ == "__main__":
    main()

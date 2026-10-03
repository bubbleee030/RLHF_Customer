#!/usr/bin/env python3
"""Evaluate a trained reward-model checkpoint on a preference-pair eval file.

Loads a saved epoch checkpoint (backbone via save_pretrained + score_head.pt,
same convention as scripts/eval_ministral.py) and scores each (chosen, rejected)
pair. Reports pairwise accuracy, score margin stats, and per-pair predictions.

Usage (inside cost-model-trainer:v2):
    python3 scripts/reward/eval_reward.py \
        --checkpoint reward_output/run_reward_byprompt_<ts>/epoch2 \
        --eval-file datasets/reward/reward_eval_byprompt.jsonl \
        --output results/reward/eval_byprompt.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def checkpoint_load_spec(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "reward_model_config.json"
    if not metadata_path.exists():
        return {
            "is_lora": False,
            "base_model_name_or_path": str(checkpoint),
            "pooling": "last-token",
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "is_lora": bool(metadata.get("lora", {}).get("enabled")),
        "base_model_name_or_path": str(metadata.get("base_model_name_or_path") or checkpoint),
        "pooling": str(metadata.get("pooling", "last-token")),
    }


def row_prompt_id(row: dict) -> str | None:
    return row.get("prompt_id") or row.get("prompt_fingerprint")


def _extract_backbone(raw):
    return (
        raw.language_model if hasattr(raw, "language_model")
        else raw.model.language_model
        if hasattr(raw, "model") and hasattr(raw.model, "language_model")
        else raw.model if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens")
        else raw
    )


def load_model(checkpoint: Path, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = checkpoint_load_spec(checkpoint)
    if spec["is_lora"]:
        from peft import PeftModel
        raw = AutoModel.from_pretrained(spec["base_model_name_or_path"], dtype=dtype)
        backbone = PeftModel.from_pretrained(_extract_backbone(raw), str(checkpoint))
    else:
        backbone = AutoModel.from_pretrained(str(checkpoint), dtype=dtype)
    backbone = backbone.to(device).eval()
    cfg = backbone.config
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    if not hidden_size:
        raise ValueError("Cannot determine hidden_size.")
    score_head = nn.Linear(hidden_size, 1).float()
    score_head.load_state_dict(torch.load(checkpoint / "score_head.pt", map_location="cpu", weights_only=True))
    score_head = score_head.to(device).eval()
    return backbone, score_head, tok, device


@torch.no_grad()
def score(backbone, score_head, tok, device, text: str, max_length: int, pooling: str = "last-token") -> float:
    enc = tok(text, max_length=max_length, truncation=True, return_tensors="pt").to(device)
    out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    h = out.last_hidden_state  # (1, L, E)
    mask = enc["attention_mask"]
    if pooling == "last-token":
        idx = mask.long().sum(dim=1).sub(1).clamp(min=0)
        pooled = h[torch.arange(h.size(0), device=h.device), idx]
    elif pooling == "mean":
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    else:  # cls
        pooled = h[:, 0, :]
    return score_head(pooled.float()).squeeze(-1).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--pooling", type=str, default=None)
    args = ap.parse_args()

    backbone, score_head, tok, device = load_model(args.checkpoint)
    pooling = args.pooling or checkpoint_load_spec(args.checkpoint)["pooling"]
    rows = [json.loads(l) for l in args.eval_file.open(encoding="utf-8") if l.strip()]

    preds, correct, margins = [], 0, []
    per_prompt = defaultdict(lambda: [0, 0])  # prompt_id -> [correct, total]
    for r in rows:
        cs = score(backbone, score_head, tok, device, f"User: {r['input']}\nAssistant: {r['chosen']}",
                   args.max_length, pooling)
        rs = score(backbone, score_head, tok, device, f"User: {r['input']}\nAssistant: {r['rejected']}",
                   args.max_length, pooling)
        ok = cs > rs
        correct += int(ok)
        margins.append(cs - rs)
        pid = row_prompt_id(r)
        per_prompt[pid][0] += int(ok)
        per_prompt[pid][1] += 1
        preds.append({"prompt_id": pid, "pair": r.get("pair"),
                      "chosen_score": round(cs, 4), "rejected_score": round(rs, 4),
                      "margin": round(cs - rs, 4), "correct": bool(ok)})

    n = len(rows)
    m = torch.tensor(margins)
    prompt_accuracies = {
        str(prompt_id): prompt_correct / prompt_total
        for prompt_id, (prompt_correct, prompt_total) in per_prompt.items()
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "eval_file": str(args.eval_file),
        "num_pairs": n,
        "pairwise_accuracy": correct / max(1, n),
        "prompt_macro_accuracy": (
            sum(prompt_accuracies.values()) / len(prompt_accuracies)
            if prompt_accuracies else 0.0
        ),
        "margin_mean": m.mean().item(),
        "margin_std": m.std().item(),
        "num_prompts": len(per_prompt),
        "per_prompt_accuracy": prompt_accuracies,
        "predictions": preds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"pairwise_accuracy={result['pairwise_accuracy']:.4f}, "
          f"prompt_macro_accuracy={result['prompt_macro_accuracy']:.4f} "
          f"on {n} pairs ({len(per_prompt)} prompts), "
          f"margin_mean={result['margin_mean']:.3f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

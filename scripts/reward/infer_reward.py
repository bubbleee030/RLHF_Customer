#!/usr/bin/env python3
"""Score (prompt, response) pairs with a trained helpfulness reward model.

A reward model maps ONE prompt+response to a scalar (higher = more helpful).
Loads a saved epoch checkpoint (backbone via save_pretrained + score_head.pt).

Usage (inside cost-model-trainer:v2, GPU):
  # single pair on the CLI
  python3 scripts/reward/infer_reward.py \
      --checkpoint reward_output/run_reward_byprompt_20260622_104203/epoch1 \
      --prompt "有沒有提供資料視覺化的功能？" \
      --response "是的，我們支援多種圖表..."

  # batch: a JSONL file with {"prompt":..., "response":...} per line -> adds "reward"
  python3 scripts/reward/infer_reward.py --checkpoint <ckpt> \
      --input-file my_inputs.jsonl --output-file scored.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load(checkpoint: Path):
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = AutoModel.from_pretrained(str(checkpoint), dtype=torch.bfloat16).to(dev).eval()
    cfg = backbone.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    head = nn.Linear(hidden, 1).float()
    head.load_state_dict(torch.load(checkpoint / "score_head.pt", map_location="cpu", weights_only=True))
    return backbone, head.to(dev).eval(), tok, dev


@torch.no_grad()
def reward(backbone, head, tok, dev, prompt: str, response: str, max_length: int = 4096) -> float:
    text = f"User: {prompt}\nAssistant: {response}"  # MUST match training format
    enc = tok(text, max_length=max_length, truncation=True, return_tensors="pt").to(dev)
    h = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).last_hidden_state
    last = enc["attention_mask"].long().sum(1).sub(1).clamp(min=0)        # last non-pad token
    pooled = h[torch.arange(h.size(0), device=h.device), last]
    return head(pooled.float()).squeeze(-1).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompt", type=str)
    ap.add_argument("--response", type=str)
    ap.add_argument("--input-file", type=Path, help="JSONL with prompt/response per line")
    ap.add_argument("--output-file", type=Path)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()

    backbone, head, tok, dev = load(args.checkpoint)

    if args.input_file:
        rows = [json.loads(l) for l in args.input_file.open(encoding="utf-8") if l.strip()]
        for r in rows:
            r["reward"] = round(reward(backbone, head, tok, dev, r["prompt"], r["response"], args.max_length), 4)
        out = args.output_file or args.input_file.with_suffix(".scored.jsonl")
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"scored {len(rows)} rows -> {out}")
    else:
        if not args.prompt or args.response is None:
            raise SystemExit("provide --prompt and --response, or --input-file")
        print(round(reward(backbone, head, tok, dev, args.prompt, args.response, args.max_length), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnostic: compare one LoRA RM checkpoint under fp16 and bf16 reload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reward import load_model, score  # noqa: E402


def main() -> None:
    checkpoint = Path(sys.argv[1])
    pair = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()[0])
    for dtype in (torch.float16, torch.bfloat16):
        backbone, head, tokenizer, device = load_model(checkpoint, dtype=dtype)
        chosen = score(
            backbone, head, tokenizer, device,
            f"User: {pair['input']}\nAssistant: {pair['chosen']}", 256,
        )
        rejected = score(
            backbone, head, tokenizer, device,
            f"User: {pair['input']}\nAssistant: {pair['rejected']}", 256,
        )
        first_parameter = next(backbone.parameters())
        print(
            f"requested={dtype} actual={first_parameter.dtype} "
            f"chosen={chosen:.6f} rejected={rejected:.6f} margin={chosen-rejected:.6f}"
        )
        del backbone, head
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

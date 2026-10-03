#!/usr/bin/env python3
"""Standalone train/eval split for the cost-model dataset.

Extracted from scripts/train_cost_model_v2.py:769-776, where the split was an
in-trainer side effect with no inspectable output. Two strategies:

  by_pair   - the historical algorithm, reproduced exactly. Shuffles PAIR
              indices, so responses to the same prompt land on both sides.
              Retained only to reproduce historical runs.
  by_prompt - groups by prompt first, so no prompt appears in both splits.
              This is the correct default.

Usage:
    python3 scripts/cost/split_cost_dataset.py \
        --dataset datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl \
        --out-dir datasets/cost/splits/by_prompt_seed42 \
        --strategy by_prompt --seed 42 --eval-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROMPT_KEY = "input"


def split_indices(n_total: int, seed: int, eval_ratio: float) -> tuple[list[int], list[int]]:
    """Historical pair-level split. Mirrors train_cost_model_v2.py:769-776."""
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_eval = max(1, int(n_total * eval_ratio))
    return indices[n_eval:], indices[:n_eval]


def split_by_prompt(rows: list[dict], seed: int, eval_ratio: float,
                    prompt_key: str = PROMPT_KEY) -> tuple[list[int], list[int]]:
    """Prompt-disjoint split: every row of a prompt lands on the same side."""
    by_prompt: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_prompt.setdefault(str(row.get(prompt_key, "")), []).append(i)

    prompts = sorted(by_prompt)
    random.Random(seed).shuffle(prompts)
    n_eval_prompts = max(1, int(len(prompts) * eval_ratio))

    eval_prompts = prompts[:n_eval_prompts]
    train_prompts = prompts[n_eval_prompts:]

    eval_idx = sorted(i for p in eval_prompts for i in by_prompt[p])
    train_idx = sorted(i for p in train_prompts for i in by_prompt[p])
    return train_idx, eval_idx


def build_manifest(rows: list[dict], train_idx: list[int], eval_idx: list[int],
                   strategy: str, seed: int, eval_ratio: float,
                   prompt_key: str = PROMPT_KEY) -> dict:
    """Summarize a split, including the prompt overlap that by_pair hides."""
    train_prompts = {str(rows[i].get(prompt_key, "")) for i in train_idx}
    eval_prompts = {str(rows[i].get(prompt_key, "")) for i in eval_idx}
    return {
        "strategy": strategy,
        "seed": seed,
        "eval_ratio": eval_ratio,
        "n_total": len(rows),
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "n_prompts_total": len({str(r.get(prompt_key, "")) for r in rows}),
        "n_prompts_train": len(train_prompts),
        "n_prompts_eval": len(eval_prompts),
        "prompt_overlap": len(train_prompts & eval_prompts),
    }


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict], indices: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i in indices:
            handle.write(json.dumps(rows[i], ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split the cost-model dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=["by_prompt", "by_pair"], default="by_prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if not rows:
        raise ValueError(f"No rows loaded from {args.dataset}")

    if args.strategy == "by_pair":
        train_idx, eval_idx = split_indices(len(rows), args.seed, args.eval_ratio)
    else:
        train_idx, eval_idx = split_by_prompt(rows, args.seed, args.eval_ratio)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "train.jsonl", rows, train_idx)
    write_rows(args.out_dir / "eval.jsonl", rows, eval_idx)

    manifest = build_manifest(rows, train_idx, eval_idx,
                              args.strategy, args.seed, args.eval_ratio)
    manifest["dataset"] = str(args.dataset)
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest["prompt_overlap"] > 0:
        print(f"WARNING: {manifest['prompt_overlap']} prompts appear in BOTH splits "
              f"(expected for strategy=by_pair; this is the leakage defect).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

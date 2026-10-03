#!/usr/bin/env python3
"""Deterministic three-way prompt-disjoint split for reward-model pairs.

The shipped 296/37/36 split was produced ad hoc in two manual stages and could
not be re-derived. This reproduces the same shape deterministically: split
prompts into train and holdout, then divide the holdout evenly into validation
and sealed test.

Grouping key is `prompt_fingerprint`, so all pairs from one prompt stay together.

Usage:
    python3 scripts/reward/split_reward_dataset.py \
        --pairs datasets/reward/cs_pairs_all.jsonl \
        --out-dir datasets/reward/splits/seed42 \
        --seed 42 --train-ratio 0.8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FINGERPRINT_KEY = "prompt_fingerprint"


def split_three_way(rows: list[dict], seed: int, train_ratio: float,
                    fingerprint_key: str = FINGERPRINT_KEY,
                    ) -> tuple[list[int], list[int], list[int]]:
    """Return (train_idx, validation_idx, test_idx), disjoint by prompt."""
    by_fp: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_fp.setdefault(str(row.get(fingerprint_key, "")), []).append(i)

    fingerprints = sorted(by_fp)
    random.Random(seed).shuffle(fingerprints)

    n_train = int(len(fingerprints) * train_ratio)
    train_fps = fingerprints[:n_train]
    holdout = fingerprints[n_train:]

    # Divide the holdout as evenly as possible; validation takes the extra one.
    n_validation = (len(holdout) + 1) // 2
    validation_fps = holdout[:n_validation]
    test_fps = holdout[n_validation:]

    def collect(fps: list[str]) -> list[int]:
        return sorted(i for fp in fps for i in by_fp[fp])

    return collect(train_fps), collect(validation_fps), collect(test_fps)


def build_manifest(rows: list[dict], train_idx: list[int], validation_idx: list[int],
                   test_idx: list[int], seed: int, train_ratio: float,
                   fingerprint_key: str = FINGERPRINT_KEY) -> dict:
    def fps(idx: list[int]) -> set[str]:
        return {str(rows[i].get(fingerprint_key, "")) for i in idx}

    train_fps, validation_fps, test_fps = fps(train_idx), fps(validation_idx), fps(test_idx)
    return {
        "seed": seed,
        "train_ratio": train_ratio,
        "n_pairs_total": len(rows),
        "n_pairs_train": len(train_idx),
        "n_pairs_validation": len(validation_idx),
        "n_pairs_test": len(test_idx),
        "n_prompts_total": len({str(r.get(fingerprint_key, "")) for r in rows}),
        "n_prompts_train": len(train_fps),
        "n_prompts_validation": len(validation_fps),
        "n_prompts_test": len(test_fps),
        "overlaps": {
            "train_validation": len(train_fps & validation_fps),
            "train_test": len(train_fps & test_fps),
            "validation_test": len(validation_fps & test_fps),
        },
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
    parser = argparse.ArgumentParser(description="Three-way split for reward-model pairs")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    rows = load_rows(args.pairs)
    if not rows:
        raise ValueError(f"No rows loaded from {args.pairs}")

    train_idx, validation_idx, test_idx = split_three_way(
        rows, args.seed, args.train_ratio)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "train.jsonl", rows, train_idx)
    write_rows(args.out_dir / "validation.jsonl", rows, validation_idx)
    write_rows(args.out_dir / "test.jsonl", rows, test_idx)

    manifest = build_manifest(rows, train_idx, validation_idx, test_idx,
                              args.seed, args.train_ratio)
    manifest["pairs_file"] = str(args.pairs)
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    total_overlap = sum(manifest["overlaps"].values())
    if total_overlap:
        raise SystemExit(f"FATAL: prompt overlap detected across splits: {manifest['overlaps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

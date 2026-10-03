#!/usr/bin/env python3
"""Pure scheduling/preflight helpers for the final five-way evaluator."""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from scripts.policy_eval.variants import VARIANTS


def generation_batches(
    variants: Iterable[str], max_parallel_gpus: int
) -> tuple[tuple[str, ...], ...]:
    """Partition variants into deterministic batches no wider than the GPU pool."""
    variants = tuple(variants)
    if max_parallel_gpus < 1:
        raise ValueError("max_parallel_gpus must be at least one")
    if not variants:
        raise ValueError("variants must not be empty")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    return tuple(
        variants[index : index + max_parallel_gpus]
        for index in range(0, len(variants), max_parallel_gpus)
    )


def expected_counts(
    *, n_prompts: int, n_samples: int, variants: int
) -> dict[str, int]:
    if n_prompts < 1 or n_samples < 1 or variants < 1:
        raise ValueError("prompt, sample, and variant counts must be positive")
    responses = n_prompts * n_samples * variants
    return {
        "responses": responses,
        "scores": responses,
        "judge_runs": n_prompts * n_samples * 2,
    }


def plan(
    n_prompts: int, n_samples: int, max_parallel_gpus: int
) -> dict[str, object]:
    batches = generation_batches(VARIANTS, max_parallel_gpus)
    return {
        "variants": list(VARIANTS),
        "generation_batches": [list(batch) for batch in batches],
        "expected": expected_counts(
            n_prompts=n_prompts, n_samples=n_samples, variants=len(VARIANTS)
        ),
        "judge_groups": n_prompts * n_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the safe batch/count plan for a five-way evaluation"
    )
    parser.add_argument("--n-prompts", type=int, default=163)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-parallel-gpus", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            plan(args.n_prompts, args.n_samples, args.max_parallel_gpus),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

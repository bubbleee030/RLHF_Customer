#!/usr/bin/env python3
"""Fail-closed coverage gate for generated CM response pairs.

The ordinary pair audit can only summarize prompt IDs present in a generated
corpus.  This module also reads the immutable source inventories, so a source
prompt with zero accepted pairs cannot be mistaken for complete coverage.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


UNSAFE_LABELS = {"partial_unsafe_compliance", "full_unsafe_compliance"}


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _required_text(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record has invalid {field}")
    return value.strip()


def _source_prompt_ids(path: str | Path, role: str) -> set[str]:
    prompt_ids: set[str] = set()
    for row in _read_jsonl(path):
        if row.get("role") != role:
            raise ValueError(f"source prompt role does not match {role!r}")
        prompt_id = _required_text(row, "prompt_id")
        _required_text(row, "text")
        if prompt_id in prompt_ids:
            raise ValueError(f"source prompt IDs are not unique for {role}")
        prompt_ids.add(prompt_id)
    if not prompt_ids:
        raise ValueError(f"source prompt inventory is empty for {role}")
    return prompt_ids


def _accepted_counts(
    path: str | Path,
    role: str,
    source_prompt_ids: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen_pair_indexes: set[tuple[str, int]] = set()
    seen_triples: set[tuple[str, str, str]] = set()
    for row in _read_jsonl(path):
        if row.get("role") != role:
            raise ValueError(f"generated pair role does not match {role!r}")
        if row.get("accepted") is not True:
            continue
        prompt_id = _required_text(row, "prompt_id")
        if prompt_id not in source_prompt_ids:
            raise ValueError(f"accepted {role} pair uses an unknown source prompt")
        if row.get("safe_label") != "safe_refusal" or row.get("unsafe_label") not in UNSAFE_LABELS:
            raise ValueError("accepted pair has an invalid policy-label pair")
        pair_index = row.get("pair_index")
        if not isinstance(pair_index, int) or pair_index < 0:
            raise ValueError("accepted pair has an invalid pair_index")
        safe_response = _required_text(row, "answer")
        unsafe_response = _required_text(row, "other_answer")
        input_text = _required_text(row, "input")
        if safe_response == unsafe_response:
            raise ValueError("accepted pair has identical response text")
        pair_key = (prompt_id, pair_index)
        if pair_key in seen_pair_indexes:
            raise ValueError("multiple accepted pairs occupy one prompt pair_index")
        triple = (input_text, safe_response, unsafe_response)
        if triple in seen_triples:
            raise ValueError("duplicate accepted response triple")
        seen_pair_indexes.add(pair_key)
        seen_triples.add(triple)
        counts[prompt_id] += 1
    return counts


def _coverage_summary(counts: Counter[str], prompt_ids: set[str]) -> dict[str, int | float]:
    values = [counts[prompt_id] for prompt_id in prompt_ids]
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def require_minimum_pair_coverage(
    train_prompts_path: str | Path,
    val_prompts_path: str | Path,
    train_pairs_path: str | Path,
    val_pairs_path: str | Path,
    *,
    min_pairs: int = 3,
) -> dict:
    """Return safe aggregate coverage evidence or raise for an incomplete corpus."""
    if min_pairs < 1:
        raise ValueError("min_pairs must be at least one")
    sources = {
        "train": _source_prompt_ids(train_prompts_path, "train"),
        "val": _source_prompt_ids(val_prompts_path, "val"),
    }
    counts = {
        "train": _accepted_counts(train_pairs_path, "train", sources["train"]),
        "val": _accepted_counts(val_pairs_path, "val", sources["val"]),
    }
    underfilled = {
        role: sorted(prompt_id for prompt_id in sources[role] if counts[role][prompt_id] < min_pairs)
        for role in ("train", "val")
    }
    for role in ("train", "val"):
        if underfilled[role]:
            preview = ", ".join(underfilled[role][:10])
            raise ValueError(
                f"{role} source prompts fewer than {min_pairs} accepted pairs: {preview}"
            )
    return {
        "min_pairs": min_pairs,
        "source_prompts": {role: len(sources[role]) for role in sources},
        "accepted_pairs": {role: sum(counts[role].values()) for role in counts},
        "accepted_pairs_per_source_prompt": {
            role: _coverage_summary(counts[role], sources[role]) for role in counts
        },
        "underfilled_source_prompts": {role: 0 for role in sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed unless every source CM prompt has enough accepted pairs"
    )
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--val-prompts", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--val-pairs", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            require_minimum_pair_coverage(
                args.train_prompts,
                args.val_prompts,
                args.train_pairs,
                args.val_pairs,
                min_pairs=args.min_pairs,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

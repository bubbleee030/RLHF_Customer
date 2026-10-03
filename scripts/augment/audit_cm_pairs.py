#!/usr/bin/env python3
"""Audit generated CM response-pair records without printing answer text."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


UNSAFE_LABELS = {
    "partial_unsafe_compliance",
    "full_unsafe_compliance",
}


def _read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            records.append(record)
    return records


def _text(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"accepted record has invalid {field}")
    return value.strip()


def _validate_record(record: dict, expected_role: str) -> tuple[bool, tuple[str, str, str] | None]:
    prompt_id = record.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("record has invalid prompt_id")
    if record.get("role") != expected_role:
        raise ValueError(f"record role does not match {expected_role!r} input")
    generator = record.get("generator")
    if not isinstance(generator, str) or not generator.strip():
        raise ValueError("record has invalid generator")
    accepted = record.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("record accepted field must be bool")
    if not accepted:
        return False, None

    safe_label = record.get("safe_label")
    unsafe_label = record.get("unsafe_label")
    if safe_label != "safe_refusal" or unsafe_label not in UNSAFE_LABELS:
        raise ValueError("accepted record has an invalid policy-label pair")
    triple = (
        _text(record, "input"),
        _text(record, "answer"),
        _text(record, "other_answer"),
    )
    if triple[1] == triple[2]:
        raise ValueError("accepted record has identical response text")
    return True, triple


def _median_summary(counts: Iterable[int]) -> dict[str, int | float | None]:
    values = list(counts)
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def audit_pair_files(train_path: str | Path, val_path: str | Path) -> dict:
    """Return aggregate provenance/integrity evidence, never response content."""
    by_role = {
        "train": _read_jsonl(train_path),
        "val": _read_jsonl(val_path),
    }
    accepted = Counter()
    rejected = Counter()
    prompt_ids: dict[str, set[str]] = defaultdict(set)
    generators: dict[str, set[str]] = defaultdict(set)
    label_pairs = Counter()
    per_prompt = Counter()
    seen_triples: set[tuple[str, str, str]] = set()
    duplicate_triples = 0

    for role, records in by_role.items():
        for record in records:
            is_accepted, triple = _validate_record(record, role)
            prompt_id = record["prompt_id"].strip()
            generator = record["generator"].strip()
            prompt_ids[role].add(prompt_id)
            generators[role].add(generator)
            if not is_accepted:
                rejected[role] += 1
                continue
            accepted[role] += 1
            per_prompt[prompt_id] += 1
            label_pairs[
                f"{record['safe_label']}|{record['unsafe_label']}"
            ] += 1
            assert triple is not None
            if triple in seen_triples:
                duplicate_triples += 1
            else:
                seen_triples.add(triple)

    overlap = sorted(generators["train"] & generators["val"])
    if overlap:
        raise ValueError("train and validation generator pools overlap")

    return {
        "records": {role: len(records) for role, records in by_role.items()},
        "accepted": {role: accepted[role] for role in by_role},
        "rejected": {role: rejected[role] for role in by_role},
        "unique_prompt_ids": {role: len(prompt_ids[role]) for role in by_role},
        "label_pairs": dict(sorted(label_pairs.items())),
        "generators": {role: sorted(generators[role]) for role in by_role},
        "generator_overlap": overlap,
        "accepted_pairs_per_prompt": _median_summary(per_prompt.values()),
        "duplicate_triples": duplicate_triples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit generated CM pairs without emitting response text"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_pair_files(args.train, args.val), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a prompt-disjoint augmented dataset for cost-model retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


PAIR_FIELDS = (
    "input",
    "answer",
    "other_answer",
    "safer",
    "is_safe",
    "is_other_safe",
)
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
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _required_text(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _validate_pair(row: dict) -> dict:
    normalized = {
        "input": _required_text(row, "input"),
        "answer": _required_text(row, "answer"),
        "other_answer": _required_text(row, "other_answer"),
    }
    for field in ("safer", "is_safe", "is_other_safe"):
        value = row.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be bool")
        normalized[field] = value
    if normalized["answer"] == normalized["other_answer"]:
        raise ValueError("answer and other_answer must differ")

    if normalized["safer"]:
        safer_sign = int(normalized["is_safe"])
        unsafer_sign = int(normalized["is_other_safe"])
    else:
        safer_sign = int(normalized["is_other_safe"])
        unsafer_sign = int(normalized["is_safe"])
    if safer_sign < unsafer_sign:
        raise ValueError("safer response has lower safety sign")
    return normalized


def _is_verified_generated_pair(row: dict, expected_role: str) -> bool:
    role = row.get("role")
    if role != expected_role:
        raise ValueError(
            f"generated row role {role!r} cannot be read from {expected_role!r} split"
        )
    if row.get("accepted") is not True:
        return False
    return (
        row.get("safe_label") == "safe_refusal"
        and row.get("unsafe_label") in UNSAFE_LABELS
    )


def _normalize_generated_pair(row: dict) -> dict:
    return _validate_pair(
        {
            "input": row.get("input"),
            "answer": row.get("answer"),
            "other_answer": row.get("other_answer"),
            "safer": True,
            "is_safe": True,
            "is_other_safe": False,
        }
    )


def _triple(row: dict) -> tuple[str, str, str]:
    return row["input"], row["answer"], row["other_answer"]


def _deduplicate(rows: list[dict], seen: set[tuple[str, str, str]]) -> tuple[list[dict], int]:
    result: list[dict] = []
    dropped = 0
    for row in rows:
        key = _triple(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        result.append(row)
    return result, dropped


def _generated_rows(rows: list[dict], expected_role: str) -> tuple[list[tuple[dict, str]], int]:
    accepted: list[tuple[dict, str]] = []
    rejected = 0
    for row in rows:
        if not _is_verified_generated_pair(row, expected_role):
            rejected += 1
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("accepted generated row requires prompt_id")
        accepted.append((_normalize_generated_pair(row), prompt_id))
    return accepted, rejected


def _histogram(values: list[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def build_dataset(
    base_train_path: str | Path,
    base_eval_path: str | Path,
    generated_train_path: str | Path,
    generated_eval_path: str | Path,
    *,
    min_pairs: int,
) -> tuple[list[dict], list[dict], dict]:
    """Return validated train/eval pairs and a provenance summary.

    Base rows are retained first. Generated rows are accepted only from their
    designated side, then exact triples are deduplicated across both sides.
    """
    if min_pairs < 1:
        raise ValueError("min_pairs must be at least one")

    base_train = [_validate_pair(row) for row in _read_jsonl(base_train_path)]
    base_eval = [_validate_pair(row) for row in _read_jsonl(base_eval_path)]
    generated_train, rejected_train = _generated_rows(
        _read_jsonl(generated_train_path), "train"
    )
    generated_eval, rejected_eval = _generated_rows(
        _read_jsonl(generated_eval_path), "val"
    )

    seen: set[tuple[str, str, str]] = set()
    train, dropped_base_train = _deduplicate(base_train, seen)
    train_generated, dropped_generated_train = _deduplicate(
        [row for row, _ in generated_train], seen
    )
    train.extend(train_generated)
    evaluation, dropped_base_eval = _deduplicate(base_eval, seen)
    eval_generated, dropped_generated_eval = _deduplicate(
        [row for row, _ in generated_eval], seen
    )
    evaluation.extend(eval_generated)

    train_prompts = {row["input"] for row in train}
    eval_prompts = {row["input"] for row in evaluation}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise ValueError(f"prompt overlap across train/eval: {len(overlap)}")

    accepted_by_prompt: dict[str, int] = defaultdict(int)
    for (_, prompt_id), row in zip(generated_train, train_generated):
        if row in train_generated:
            accepted_by_prompt[prompt_id] += 1
    for (_, prompt_id), row in zip(generated_eval, eval_generated):
        if row in eval_generated:
            accepted_by_prompt[prompt_id] += 1
    insufficient = sorted(
        prompt_id
        for prompt_id, count in accepted_by_prompt.items()
        if count < min_pairs
    )
    if insufficient:
        raise ValueError(
            f"{len(insufficient)} generated prompts have fewer than {min_pairs} pairs"
        )

    generators = sorted(
        {
            str(row["generator"])
            for row in _read_jsonl(generated_train_path)
            + _read_jsonl(generated_eval_path)
            if row.get("accepted") is True and isinstance(row.get("generator"), str)
        }
    )
    manifest = {
        "n_base_train": len(base_train),
        "n_base_eval": len(base_eval),
        "n_generated_train": len(train_generated),
        "n_generated_eval": len(eval_generated),
        "n_train": len(train),
        "n_eval": len(evaluation),
        "n_prompts_train": len(train_prompts),
        "n_prompts_eval": len(eval_prompts),
        "prompt_overlap": len(overlap),
        "duplicate_triples_dropped": (
            dropped_base_train
            + dropped_generated_train
            + dropped_base_eval
            + dropped_generated_eval
        ),
        "rejected_unverified_pairs": rejected_train + rejected_eval,
        "generator_pools": generators,
        "accepted_pairs_per_prompt_histogram": _histogram(
            list(accepted_by_prompt.values())
        ),
    }
    return train, evaluation, manifest


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a prompt-disjoint augmented CM train/eval dataset"
    )
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--pairs-train", type=Path, required=True)
    parser.add_argument("--pairs-val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-accepted-per-prompt", type=int, default=3)
    args = parser.parse_args()

    train, evaluation, manifest = build_dataset(
        args.base_train,
        args.base_eval,
        args.pairs_train,
        args.pairs_val,
        min_pairs=args.min_accepted_per_prompt,
    )
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    _write_jsonl_atomic(train_path, train)
    _write_jsonl_atomic(eval_path, evaluation)
    manifest.update(
        {
            "base_train": str(args.base_train),
            "base_eval": str(args.base_eval),
            "pairs_train": str(args.pairs_train),
            "pairs_val": str(args.pairs_val),
            "train_sha256": _sha256(train_path),
            "eval_sha256": _sha256(eval_path),
        }
    )
    manifest_path = args.out_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Behavior tests for the reward-model three-way prompt-disjoint split."""

from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reward.split_reward_dataset import (  # noqa: E402
    build_manifest,
    split_three_way,
)

REPO = Path(__file__).resolve().parents[1]


def _rows(n_prompts: int, per_prompt: int) -> list[dict]:
    rows = []
    for p in range(n_prompts):
        for k in range(per_prompt):
            rows.append({
                "input": f"prompt-{p}",
                "chosen": f"good-{p}-{k}",
                "rejected": f"bad-{p}-{k}",
                "prompt_fingerprint": f"fp-{p}",
            })
    return rows


def test_three_splits_have_zero_prompt_overlap() -> None:
    rows = _rows(n_prompts=100, per_prompt=5)
    train, validation, test = split_three_way(rows, seed=42, train_ratio=0.8)

    def fps(idx):
        return {rows[i]["prompt_fingerprint"] for i in idx}

    assert fps(train) & fps(validation) == set()
    assert fps(train) & fps(test) == set()
    assert fps(validation) & fps(test) == set()


def test_every_row_lands_in_exactly_one_split() -> None:
    rows = _rows(n_prompts=100, per_prompt=5)
    train, validation, test = split_three_way(rows, seed=42, train_ratio=0.8)
    assert sorted(train + validation + test) == list(range(len(rows)))


def test_holdout_is_divided_as_evenly_as_possible() -> None:
    """369 prompts at 0.8 -> 295 train, 74 holdout -> 37 validation / 37 test."""
    rows = _rows(n_prompts=369, per_prompt=1)
    train, validation, test = split_three_way(rows, seed=42, train_ratio=0.8)
    assert len(train) == 295
    assert abs(len(validation) - len(test)) <= 1
    assert len(validation) + len(test) == 74


def test_split_is_deterministic_for_a_seed() -> None:
    rows = _rows(n_prompts=60, per_prompt=3)
    a = split_three_way(rows, seed=13, train_ratio=0.8)
    b = split_three_way(rows, seed=13, train_ratio=0.8)
    c = split_three_way(rows, seed=14, train_ratio=0.8)
    assert a == b
    assert a != c


def test_manifest_reports_zero_overlaps() -> None:
    rows = _rows(n_prompts=60, per_prompt=3)
    train, validation, test = split_three_way(rows, seed=42, train_ratio=0.8)
    manifest = build_manifest(rows, train, validation, test, 42, 0.8)
    assert manifest["overlaps"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert manifest["n_prompts_total"] == 60


@pytest.mark.skipif(not (REPO / "datasets").is_dir(), reason="datasets/ is not part of the public release")
def test_existing_shipped_splits_are_prompt_disjoint() -> None:
    """Guards the property the shipped 296/37/36 files already have."""
    def fps(name: str) -> set[str]:
        path = REPO / "datasets" / "reward" / f"cs_within_{name}.jsonl"
        return {json.loads(l)["prompt_fingerprint"]
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}

    train, validation, test = fps("train"), fps("validation"), fps("test")
    assert (len(train), len(validation), len(test)) == (296, 37, 36)
    assert train & validation == set()
    assert train & test == set()
    assert validation & test == set()
    assert len(train | validation | test) == 369

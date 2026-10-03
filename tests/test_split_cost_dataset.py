#!/usr/bin/env python3
"""Behavior tests for the standalone cost-model dataset split."""

from __future__ import annotations

import json
import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost.split_cost_dataset import (  # noqa: E402
    build_manifest,
    split_by_prompt,
    split_indices,
)

REPO = Path(__file__).resolve().parents[1]
CLEAN = REPO / "datasets" / "cost" / "cost_dataset_for_safe_rlhf_clean.jsonl"


def _rows(n_prompts: int, per_prompt: int) -> list[dict]:
    rows = []
    for p in range(n_prompts):
        for k in range(per_prompt):
            rows.append({
                "input": f"prompt-{p}",
                "answer": f"safe-{p}-{k}",
                "other_answer": f"unsafe-{p}-{k}",
                "safer": True,
                "is_safe": True,
                "is_other_safe": False,
            })
    return rows


def test_by_pair_reproduces_historical_algorithm_exactly() -> None:
    """by_pair must match train_cost_model_v2.py:769-776 bit for bit."""
    n_total = 1461
    train_idx, eval_idx = split_indices(n_total, seed=42, eval_ratio=0.1)

    expected = list(range(n_total))
    random.Random(42).shuffle(expected)
    n_eval = max(1, int(n_total * 0.1))

    assert eval_idx == expected[:n_eval]
    assert train_idx == expected[n_eval:]
    assert len(train_idx) == 1315
    assert len(eval_idx) == 146


@pytest.mark.skipif(not (REPO / "datasets").is_dir(), reason="datasets/ is not part of the public release")
def test_by_pair_on_real_dataset_gives_1315_146() -> None:
    rows = [json.loads(l) for l in CLEAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1461
    train_idx, eval_idx = split_indices(len(rows), seed=42, eval_ratio=0.1)
    assert (len(train_idx), len(eval_idx)) == (1315, 146)


def test_by_prompt_produces_zero_prompt_overlap() -> None:
    rows = _rows(n_prompts=50, per_prompt=4)
    train_idx, eval_idx = split_by_prompt(rows, seed=42, eval_ratio=0.2)

    train_prompts = {rows[i]["input"] for i in train_idx}
    eval_prompts = {rows[i]["input"] for i in eval_idx}
    assert train_prompts & eval_prompts == set()
    assert train_prompts | eval_prompts == {r["input"] for r in rows}


def test_by_prompt_keeps_every_row() -> None:
    rows = _rows(n_prompts=50, per_prompt=4)
    train_idx, eval_idx = split_by_prompt(rows, seed=42, eval_ratio=0.2)
    assert sorted(train_idx + eval_idx) == list(range(len(rows)))


def test_by_prompt_is_deterministic_for_a_seed() -> None:
    rows = _rows(n_prompts=50, per_prompt=4)
    a = split_by_prompt(rows, seed=7, eval_ratio=0.2)
    b = split_by_prompt(rows, seed=7, eval_ratio=0.2)
    c = split_by_prompt(rows, seed=8, eval_ratio=0.2)
    assert a == b
    assert a != c


def test_by_pair_leaks_prompts_and_manifest_records_it() -> None:
    """The defect this task exists to expose: by_pair shares prompts across splits."""
    rows = _rows(n_prompts=50, per_prompt=4)
    train_idx, eval_idx = split_indices(len(rows), seed=42, eval_ratio=0.2)
    manifest = build_manifest(rows, train_idx, eval_idx, "by_pair", 42, 0.2)
    assert manifest["prompt_overlap"] > 0

    train_idx, eval_idx = split_by_prompt(rows, seed=42, eval_ratio=0.2)
    manifest = build_manifest(rows, train_idx, eval_idx, "by_prompt", 42, 0.2)
    assert manifest["prompt_overlap"] == 0


def test_manifest_has_required_keys() -> None:
    rows = _rows(n_prompts=10, per_prompt=2)
    train_idx, eval_idx = split_by_prompt(rows, seed=1, eval_ratio=0.2)
    manifest = build_manifest(rows, train_idx, eval_idx, "by_prompt", 1, 0.2)
    for key in ("strategy", "seed", "eval_ratio", "n_total", "n_train", "n_eval",
                "n_prompts_total", "n_prompts_train", "n_prompts_eval", "prompt_overlap"):
        assert key in manifest, f"missing manifest key: {key}"

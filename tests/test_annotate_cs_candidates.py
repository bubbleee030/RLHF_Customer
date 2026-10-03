#!/usr/bin/env python3
"""Behavior tests for the customer-candidate annotation pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reward.annotate_cs_candidates import (  # noqa: E402
    load_checkpoint,
    consensus_pairs,
    load_candidates,
    parse_ranking,
    run_annotation_jobs,
    split_prompt_rows,
)


def test_parse_ranking_accepts_exact_six_label_json() -> None:
    got = parse_ranking('analysis\n{"ranking":["F","A","C","B","D","E"]}', "ABCDEF")
    assert got == ["F", "A", "C", "B", "D", "E"]


def test_parse_ranking_rejects_duplicates_or_missing_labels() -> None:
    assert parse_ranking('{"ranking":["A","B","C","D","E","E"]}', "ABCDEF") is None
    assert parse_ranking('{"ranking":["A","B","C","D","E"]}', "ABCDEF") is None


def _record(prompt: str = "p0") -> dict:
    return {
        "prompt": prompt,
        "candidates": [
            {"temp": 0.3 + i / 10, "answer": f"answer-{i}", "has_think": True}
            for i in range(6)
        ],
    }


def test_consensus_pairs_drops_only_the_disagreed_direction() -> None:
    rows = consensus_pairs(
        _record(),
        rankings=[
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5, 4],
        ],
        judge_models=["judge-a", "judge-a"],
    )
    assert len(rows) == 14
    assert not any({r["chosen_index"], r["rejected_index"]} == {4, 5} for r in rows)
    first = next(r for r in rows if r["chosen_index"] == 0 and r["rejected_index"] == 1)
    assert first["input"] == "p0"
    assert first["chosen"] == "answer-0"
    assert first["rejected"] == "answer-1"
    assert first["agreement_count"] == 2


def test_prompt_split_is_deterministic_and_disjoint() -> None:
    rows = []
    for p in range(4):
        for pair in range(2):
            rows.append({"prompt_fingerprint": f"p{p}", "pair": pair})
    train_a, eval_a = split_prompt_rows(rows, eval_ratio=0.25, seed=42)
    train_b, eval_b = split_prompt_rows(list(reversed(rows)), eval_ratio=0.25, seed=42)
    assert train_a == train_b
    assert eval_a == eval_b
    train_prompts = {r["prompt_fingerprint"] for r in train_a}
    eval_prompts = {r["prompt_fingerprint"] for r in eval_a}
    assert train_prompts.isdisjoint(eval_prompts)
    assert len(eval_prompts) == 1


def test_load_candidates_rejects_duplicate_prompts_and_empty_answers() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidates.jsonl"
        rec = _record()
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n" + json.dumps(rec, ensure_ascii=False) + "\n")
        try:
            load_candidates([path])
        except ValueError as exc:
            assert "duplicate prompt" in str(exc)
        else:
            raise AssertionError("duplicate prompts must be rejected")

        rec["prompt"] = "unique"
        rec["candidates"][2]["answer"] = "  "
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n")
        try:
            load_candidates([path])
        except ValueError as exc:
            assert "empty answer" in str(exc)
        else:
            raise AssertionError("empty candidate answers must be rejected")


def test_annotation_jobs_retry_then_checkpoint_and_resume() -> None:
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td) / "runs.jsonl"
        record = _record()
        record["prompt_fingerprint"] = "fp0"
        calls = []

        def flaky_call(model: str, system_prompt: str, user_prompt: str) -> str:
            calls.append((model, system_prompt, user_prompt))
            if len(calls) < 3:
                raise RuntimeError("temporary failure")
            return '{"ranking":["A","B","C","D","E","F"]}'

        first = run_annotation_jobs(
            [record], ["judge-a"], runs_per_model=1, checkpoint_path=checkpoint,
            call_fn=flaky_call, workers=1, retry_delay=0,
        )
        assert len(calls) == 3
        assert len(first) == 1
        assert first[0]["ranking"] == first[0]["presented"]

        def must_not_call(*args) -> str:
            raise AssertionError("completed checkpoint entry should be resumed")

        second = run_annotation_jobs(
            [record], ["judge-a"], runs_per_model=1, checkpoint_path=checkpoint,
            call_fn=must_not_call, workers=1, retry_delay=0,
        )
        assert second == first
        assert load_checkpoint(checkpoint) == first


def test_checkpoint_never_contains_api_credentials() -> None:
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td) / "runs.jsonl"
        record = _record()
        record["prompt_fingerprint"] = "fp0"
        secret = "SECRET-API-KEY"

        def call_without_exposing_secret(model: str, system_prompt: str, user_prompt: str) -> str:
            assert secret not in system_prompt
            assert secret not in user_prompt
            return '{"ranking":["F","E","D","C","B","A"]}'

        run_annotation_jobs(
            [record], ["judge-a"], runs_per_model=1, checkpoint_path=checkpoint,
            call_fn=call_without_exposing_secret, workers=1, retry_delay=0,
        )
        assert secret not in checkpoint.read_text()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

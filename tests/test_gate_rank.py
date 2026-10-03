#!/usr/bin/env python3
"""Unit tests for gate-and-rank selection logic. Run: python3 tests/test_gate_rank.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "serve"))

from gate_rank import Candidate, REFUSAL_FALLBACK, format_text, score_checkpoint_spec, select


def test_format_matches_training_exactly():
    # Must equal the training-time format in train_reward_model.py / train_cost_model_v2.py
    assert format_text("你好", "您好") == "User: 你好\nAssistant: 您好"


def test_picks_highest_reward_among_safe():
    cands = [
        Candidate("a", cost=-1.0, reward=0.5),
        Candidate("b", cost=-2.0, reward=2.0),
        Candidate("c", cost=3.0, reward=9.9),  # unsafe with top reward — must be gated
    ]
    s = select(cands)
    assert s.response == "b"
    assert s.chosen_index == 1
    assert not s.used_fallback


def test_all_gated_returns_fallback():
    s = select([Candidate("x", cost=0.5, reward=1.0), Candidate("y", cost=2.0, reward=2.0)])
    assert s.used_fallback
    assert s.response == REFUSAL_FALLBACK
    assert s.chosen_index is None


def test_boundary_zero_cost_is_unsafe():
    assert not Candidate("z", cost=0.0).safe


def test_empty_candidate_list_falls_back():
    s = select([])
    assert s.used_fallback


def test_score_checkpoint_spec_detects_lora_adapter():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        (path / "reward_model_config.json").write_text(json.dumps({
            "base_model_name_or_path": "mistralai/example",
            "pooling": "last-token",
            "lora": {"enabled": True},
        }))
        assert score_checkpoint_spec(path) == {
            "is_lora": True,
            "base_model_name_or_path": "mistralai/example",
            "pooling": "last-token",
        }


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"{len(fns)} tests passed")

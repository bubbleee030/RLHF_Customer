#!/usr/bin/env python3
"""Behavior tests for deterministic paired actor evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ppo_lag"))

from eval_controls import (  # noqa: E402
    derived_sample_seed,
    generate_paired_samples,
    paired_reward_summary,
)


def test_derived_seed_is_stable_and_sensitive_to_every_coordinate() -> None:
    first = derived_sample_seed(42, "benign", 0, 0)
    assert first == derived_sample_seed(42, "benign", 0, 0)
    assert first != derived_sample_seed(43, "benign", 0, 0)
    assert first != derived_sample_seed(42, "harmful", 0, 0)
    assert first != derived_sample_seed(42, "benign", 1, 0)
    assert first != derived_sample_seed(42, "benign", 0, 1)
    assert 0 <= first < 2**31


def test_paired_reward_summary_matches_sample_positions_and_prompt_means() -> None:
    records = [
        {
            "track": "benign",
            "base": {"rewards": [1.0, 2.0]},
            "ppo": {"rewards": [2.0, 1.0]},
        },
        {
            "track": "benign",
            "base": {"rewards": [0.0, 0.0]},
            "ppo": {"rewards": [1.0, 3.0]},
        },
        {
            "track": "harmful",
            "base": {"rewards": [100.0]},
            "ppo": {"rewards": [-100.0]},
        },
    ]
    assert paired_reward_summary(records, "benign") == {
        "mean_delta": 1.0,
        "sample_win_rate": 0.75,
        "prompt_win_rate": 0.5,
        "n_samples": 4,
        "n_prompts": 2,
    }


def test_paired_reward_summary_rejects_unmatched_sample_counts() -> None:
    records = [{
        "track": "benign",
        "base": {"rewards": [1.0]},
        "ppo": {"rewards": [1.0, 2.0]},
    }]
    try:
        paired_reward_summary(records, "benign")
    except ValueError as error:
        assert "sample counts" in str(error)
    else:
        raise AssertionError("unmatched samples must be rejected")


def test_generate_paired_samples_reuses_each_seed_for_both_variants() -> None:
    calls: list[tuple[str, int]] = []

    def generate_one(variant: str, seed: int) -> str:
        calls.append((variant, seed))
        return f"{variant}:{seed}"

    result = generate_paired_samples(
        global_seed=42,
        track="benign",
        prompt_index=3,
        n_samples=2,
        generate_one=generate_one,
    )
    seed0 = derived_sample_seed(42, "benign", 3, 0)
    seed1 = derived_sample_seed(42, "benign", 3, 1)
    assert calls == [
        ("base", seed0), ("ppo", seed0),
        ("base", seed1), ("ppo", seed1),
    ]
    assert result == {
        "sample_seeds": [seed0, seed1],
        "base": [f"base:{seed0}", f"base:{seed1}"],
        "ppo": [f"ppo:{seed0}", f"ppo:{seed1}"],
    }


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

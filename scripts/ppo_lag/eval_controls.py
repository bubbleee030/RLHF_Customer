#!/usr/bin/env python3
"""Pure controls for deterministic, paired policy evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def derived_sample_seed(global_seed: int, track: str, prompt_index: int,
                        sample_index: int) -> int:
    payload = f"{global_seed}\0{track}\0{prompt_index}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 2**31


def generate_paired_samples(global_seed: int, track: str, prompt_index: int,
                            n_samples: int,
                            generate_one: Callable[[str, int], T]) -> dict:
    result: dict[str, list] = {"sample_seeds": [], "base": [], "ppo": []}
    for sample_index in range(n_samples):
        seed = derived_sample_seed(
            global_seed, track, prompt_index, sample_index)
        result["sample_seeds"].append(seed)
        for variant in ("base", "ppo"):
            result[variant].append(generate_one(variant, seed))
    return result


def paired_reward_summary(records: list[dict], track: str) -> dict:
    rows = [row for row in records if row["track"] == track]
    if not rows:
        raise ValueError(f"no paired records for track {track!r}")

    deltas: list[float] = []
    prompt_wins = 0
    for row in rows:
        base = row["base"]["rewards"]
        ppo = row["ppo"]["rewards"]
        if len(base) != len(ppo):
            raise ValueError("base and PPO sample counts do not match")
        sample_deltas = [float(after) - float(before)
                         for before, after in zip(base, ppo)]
        deltas.extend(sample_deltas)
        prompt_wins += int(sum(sample_deltas) / len(sample_deltas) > 0)

    return {
        "mean_delta": sum(deltas) / len(deltas),
        "sample_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        "prompt_win_rate": prompt_wins / len(rows),
        "n_samples": len(deltas),
        "n_prompts": len(rows),
    }

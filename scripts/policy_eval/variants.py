"""Shared definitions for the final blinded policy/PPO evaluation."""

from __future__ import annotations


VARIANTS = (
    "base_raw",
    "base_policy_zh",
    "base_policy_bilingual",
    "ppo_raw",
    "ppo_policy",
)
PPO_VARIANTS = frozenset({"ppo_raw", "ppo_policy"})


def uses_policy_prompt(variant: str) -> bool:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    return variant in {
        "base_policy_zh",
        "base_policy_bilingual",
        "ppo_policy",
    }


def adapter_path_for_variant(
    variant: str, raw_adapter: str | None, policy_adapter: str | None
) -> str | None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant == "ppo_raw":
        return raw_adapter
    if variant == "ppo_policy":
        return policy_adapter
    return None


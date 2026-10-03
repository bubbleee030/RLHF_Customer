#!/usr/bin/env python3
"""CPU behavior tests for the customer-actor PPO mode."""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ppo_lag"))

from models_ppo import (  # noqa: E402
    _extract_language_model,
    _select_actor_peft_base,
    customer_prompt_text,
    strip_think_response,
)
from train_ppo_lag import (  # noqa: E402
    checkpoint_budget_mib,
    checkpoint_components,
    has_checkpoint_space,
    load_prompts,
    max_updates_reached,
    metrics_are_finite,
)


def test_customer_prompt_uses_checkpoint_training_format() -> None:
    assert customer_prompt_text("如何提高 API 配額？") == "[INST]如何提高 API 配額？[/INST]"


def test_visible_response_strips_private_think_block() -> None:
    raw = "[THINK]internal reasoning[/THINK]\n這是使用者可見答案。"
    assert strip_think_response(raw) == "這是使用者可見答案。"
    assert strip_think_response("沒有思考區塊") == "沒有思考區塊"
    assert strip_think_response("[THINK]尚未完成的內部推理") == ""


def test_max_updates_zero_means_unlimited() -> None:
    assert max_updates_reached(global_step=999, max_updates=0) is False


def test_max_updates_stops_at_exact_boundary() -> None:
    assert max_updates_reached(global_step=29, max_updates=30) is False
    assert max_updates_reached(global_step=30, max_updates=30) is True


def test_extracts_nested_mistral3_language_model() -> None:
    class Wrapper:
        pass

    raw, multimodal, language = Wrapper(), Wrapper(), object()
    multimodal.language_model = language
    raw.model = multimodal
    assert _extract_language_model(raw) is language


def test_actor_keeps_generation_capable_outer_wrapper() -> None:
    class Wrapper:
        pass

    raw, multimodal, language = Wrapper(), Wrapper(), object()
    multimodal.language_model = language
    raw.model = multimodal
    raw.prepare_inputs_for_generation = lambda: None
    assert _select_actor_peft_base(raw) is raw


def test_load_prompts_accepts_pair_dataset_and_deduplicates() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pairs.jsonl"
        path.write_text("\n".join(json.dumps({"input": value})
                                  for value in ["甲", "乙", "甲"]))
        assert load_prompts(str(path)) == ["甲", "乙"]


def test_non_finite_training_metrics_fail_the_guard() -> None:
    assert metrics_are_finite([0.0, -2.3, 8.0]) is True
    assert metrics_are_finite([0.0, float("nan")]) is False
    assert metrics_are_finite([float("inf")]) is False


def test_checkpoint_space_guard_preserves_reserve() -> None:
    gib = 2**30
    assert has_checkpoint_space(free_bytes=gib, estimated_write_bytes=400 * 2**20,
                                reserve_bytes=450 * 2**20) is True
    assert has_checkpoint_space(free_bytes=800 * 2**20,
                                estimated_write_bytes=400 * 2**20,
                                reserve_bytes=450 * 2**20) is False


def test_actor_only_checkpoint_budget_and_components() -> None:
    assert checkpoint_budget_mib(save_critics=False) == (230, 150)
    assert checkpoint_components(save_critics=False) == ("actor",)
    assert checkpoint_budget_mib(save_critics=True) == (450, 450)
    assert checkpoint_components(save_critics=True) == (
        "actor", "reward_critic", "cost_critic")


def test_actor_gradient_checkpointing_uses_non_reentrant_mode() -> None:
    class Config:
        use_cache = True

    class Model:
        config = Config()
        kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            self.kwargs = gradient_checkpointing_kwargs

    from models_ppo import LoRAActor

    actor = LoRAActor.__new__(LoRAActor)
    actor.model = Model()
    actor.enable_gradient_checkpointing()
    assert actor.model.kwargs == {"use_reentrant": False}
    assert actor.model.config.use_cache is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

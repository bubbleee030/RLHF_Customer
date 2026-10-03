#!/usr/bin/env python3
"""Behavior tests for RM model-selection and trainability controls."""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reward.train_reward_model import (  # noqa: E402
    EarlyStopper,
    build_checkpoint_metadata,
    configure_lora,
    select_best_epoch,
    trainable_parameter_names,
)
from scripts.reward.eval_reward import (  # noqa: E402
    _extract_backbone,
    checkpoint_load_spec,
    row_prompt_id,
)


def test_best_epoch_is_selected_by_eval_loss_not_accuracy() -> None:
    history = [
        {"epoch": 1, "loss": 0.60, "accuracy": 0.70},
        {"epoch": 2, "loss": 0.50, "accuracy": 0.65},
        {"epoch": 3, "loss": 0.70, "accuracy": 0.80},
    ]
    assert select_best_epoch(history)["epoch"] == 2


def test_early_stopper_counts_consecutive_non_improvements() -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.is_improvement(1.0) is True
    assert stopper.update(1.0) is False
    assert stopper.is_improvement(1.1) is False
    assert stopper.update(1.1) is False
    assert stopper.update(1.2) is True

    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(1.0) is False
    assert stopper.update(1.1) is False
    assert stopper.update(0.9) is False
    assert stopper.update(0.95) is False


def test_trainable_parameter_names_excludes_frozen_backbone_weights() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    names = trainable_parameter_names(model)
    assert names == ["1.weight", "1.bias"]


def test_checkpoint_metadata_records_reload_contract() -> None:
    metadata = build_checkpoint_metadata(
        base_model_name_or_path="mistralai/example",
        pooling="last-token",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    assert metadata == {
        "format_version": 1,
        "base_model_name_or_path": "mistralai/example",
        "pooling": "last-token",
        "lora": {"enabled": True, "r": 16, "alpha": 32, "dropout": 0.05},
    }


def test_configure_lora_uses_expected_rm_targets() -> None:
    backbone = object()
    captured = {}

    def config_factory(**kwargs):
        captured.update(kwargs)
        return "config"

    def peft_factory(got_backbone, got_config):
        assert got_backbone is backbone
        assert got_config == "config"
        return "wrapped"

    wrapped = configure_lora(
        backbone, r=16, alpha=32, dropout=0.05,
        config_factory=config_factory, peft_factory=peft_factory,
    )
    assert wrapped == "wrapped"
    assert captured == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "task_type": "FEATURE_EXTRACTION",
    }


def test_checkpoint_load_spec_detects_lora_adapter() -> None:
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        (checkpoint / "reward_model_config.json").write_text(json.dumps({
            "base_model_name_or_path": "mistralai/example",
            "pooling": "last-token",
            "lora": {"enabled": True, "r": 16, "alpha": 32, "dropout": 0.05},
        }))
        assert checkpoint_load_spec(checkpoint) == {
            "is_lora": True,
            "base_model_name_or_path": "mistralai/example",
            "pooling": "last-token",
        }


def test_eval_extracts_nested_mistral3_language_model() -> None:
    class Wrapper:
        pass

    raw, multimodal, language = Wrapper(), Wrapper(), object()
    multimodal.language_model = language
    raw.model = multimodal
    assert _extract_backbone(raw) is language


def test_eval_uses_customer_prompt_fingerprint_as_prompt_id() -> None:
    assert row_prompt_id({"prompt_fingerprint": "abc"}) == "abc"
    assert row_prompt_id({"prompt_id": "legacy", "prompt_fingerprint": "abc"}) == "legacy"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

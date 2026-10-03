#!/usr/bin/env python3
"""Behavior tests for LoRA support in the cost-model trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_cost_model_v2 import (  # noqa: E402
    configure_cost_lora,
    cost_trainable_parameter_names,
    save_checkpoint,
)


class _TinyBackbone(nn.Module):
    """Minimal stand-in with the projection names LoRA targets."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.q_proj(x))


class _SaveableBackbone(_TinyBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.saved_to: Path | None = None

    def save_pretrained(self, out_dir: Path, safe_serialization: bool = True) -> None:
        self.saved_to = Path(out_dir)


class _Tokenizer:
    def save_pretrained(self, out_dir: Path) -> None:
        Path(out_dir, "tokenizer.saved").touch()


def test_zero_rank_returns_backbone_unchanged() -> None:
    backbone = _TinyBackbone()
    result = configure_cost_lora(backbone, r=0, alpha=32, dropout=0.05)
    assert result is backbone


def test_negative_rank_returns_backbone_unchanged() -> None:
    backbone = _TinyBackbone()
    result = configure_cost_lora(backbone, r=-1, alpha=32, dropout=0.05)
    assert result is backbone


def test_positive_rank_wraps_and_freezes_base_weights() -> None:
    pytest = __import__("pytest")
    peft = pytest.importorskip("peft")
    backbone = _TinyBackbone()
    wrapped = configure_cost_lora(backbone, r=4, alpha=8, dropout=0.0)

    assert isinstance(wrapped, peft.PeftModel)
    names = cost_trainable_parameter_names(wrapped)
    assert names, "LoRA wrapping produced no trainable parameters"
    assert all("lora" in n.lower() for n in names), \
        f"non-LoRA parameters left trainable: {[n for n in names if 'lora' not in n.lower()]}"


def test_full_ft_leaves_all_parameters_trainable() -> None:
    backbone = _TinyBackbone()
    result = configure_cost_lora(backbone, r=0, alpha=32, dropout=0.05)
    names = cost_trainable_parameter_names(result)
    assert len(names) == len(list(result.named_parameters()))


def test_lora_checkpoint_saves_adapter_when_full_backbone_saving_is_disabled(
    tmp_path: Path,
) -> None:
    """LoRA's trained adapter must survive the disk-saving CLI setting."""
    backbone = _SaveableBackbone()
    model = SimpleNamespace(backbone=backbone, score_head=nn.Linear(8, 1))
    args = SimpleNamespace(lora_r=8)

    save_checkpoint(tmp_path, model, _Tokenizer(), args, save_backbone=False)

    assert backbone.saved_to == tmp_path

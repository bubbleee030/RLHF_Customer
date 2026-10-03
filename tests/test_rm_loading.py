#!/usr/bin/env python3
"""CPU tests for scripts/reward/test_rm_loading.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "reward"))

from test_rm_loading import (  # noqa: E402
    build_parser,
    checkpoint_load_spec,
    last_nonpadding_pool,
    load_reward_model,
    load_score_head,
    render_input,
    resolve_dtype,
)


def _write_head(path: Path, hidden_size: int = 3, bias: float = -0.25) -> None:
    torch.save(
        {
            "weight": torch.arange(hidden_size, dtype=torch.float32).reshape(1, -1),
            "bias": torch.tensor([bias], dtype=torch.float32),
        },
        path / "score_head.pt",
    )


def test_cli_keeps_loader_and_template_as_independent_axes():
    args = build_parser().parse_args([
        "--checkpoint", "/tmp/rm",
        "--load-mode", "lora",
        "--template-mode", "chat",
        "--prompt", "p",
        "--response", "r",
    ])
    assert args.load_mode == "lora"
    assert args.template_mode == "chat"
    assert args.dtype == "float16"
    assert args.max_length == 576


def test_render_training_format_matches_reward_training_exactly():
    assert render_input(None, "你好", "您好", "training") == (
        "User: 你好\nAssistant: 您好"
    )


def test_render_chat_format_delegates_to_tokenizer():
    class FakeTokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, tokenize=False):
            self.messages = messages
            assert tokenize is False
            return "<chat-rendered>"

    tokenizer = FakeTokenizer()
    assert render_input(tokenizer, "P", "R", "chat") == "<chat-rendered>"
    assert tokenizer.messages == [
        {"role": "user", "content": "P"},
        {"role": "assistant", "content": "R"},
    ]


def test_dtype_names_resolve_to_torch_dtypes():
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32


def test_lora_spec_reads_base_model_and_requires_adapter():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint)
        (checkpoint / "adapter_model.safetensors").touch()
        (checkpoint / "reward_model_config.json").write_text(json.dumps({
            "base_model_name_or_path": "mistralai/base",
            "lora": {"enabled": True},
        }))
        spec = checkpoint_load_spec(checkpoint, "lora")
        assert spec.load_mode == "lora"
        assert spec.base_model_name_or_path == "mistralai/base"


def test_lora_base_model_override_works_without_metadata():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint)
        (checkpoint / "adapter_model.bin").touch()
        spec = checkpoint_load_spec(
            checkpoint, "lora", base_model_override="/models/base",
        )
        assert spec.base_model_name_or_path == "/models/base"


def test_full_mode_rejects_lora_adapter_directory():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint)
        (checkpoint / "adapter_model.safetensors").touch()
        try:
            checkpoint_load_spec(checkpoint, "full")
        except ValueError as exc:
            assert "LoRA adapter" in str(exc)
        else:
            raise AssertionError("full mode must reject a LoRA adapter directory")


def test_external_score_head_loads_weight_and_bias_strictly():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint, hidden_size=3, bias=-0.125)
        head = load_score_head(checkpoint, hidden_size=3)
        assert head.bias is not None
        assert head.bias.item() == -0.125
        assert tuple(head.weight.shape) == (1, 3)


def test_last_nonpadding_pool_supports_left_and_right_padding():
    hidden = torch.tensor([
        [[10.0], [11.0], [12.0], [13.0]],
        [[20.0], [21.0], [22.0], [23.0]],
    ])
    mask = torch.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ])
    pooled, indexes = last_nonpadding_pool(hidden, mask)
    assert indexes.tolist() == [1, 3]
    assert pooled.squeeze(-1).tolist() == [11.0, 23.0]


class _FakeBackbone:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=3)
        self.device_seen = None
        self.eval_called = False

    def to(self, device):
        self.device_seen = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class _FakeAutoModel:
    calls = []
    next_model = None

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return cls.next_model


class _FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, source):
        cls.calls.append(source)
        return SimpleNamespace()


class _FakePeftModel:
    calls = []
    merged_backbone = None

    @classmethod
    def from_pretrained(cls, backbone, checkpoint):
        cls.calls.append((backbone, checkpoint))
        return SimpleNamespace(merge_and_unload=lambda: cls.merged_backbone)


def test_lora_loader_uses_base_then_adapter_merge():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint)
        (checkpoint / "adapter_model.safetensors").touch()
        (checkpoint / "reward_model_config.json").write_text(json.dumps({
            "base_model_name_or_path": "mistralai/base",
            "lora": {"enabled": True},
        }))
        base = _FakeBackbone()
        merged = _FakeBackbone()
        _FakeAutoModel.calls = []
        _FakeAutoModel.next_model = SimpleNamespace(language_model=base)
        _FakeAutoTokenizer.calls = []
        _FakePeftModel.calls = []
        _FakePeftModel.merged_backbone = merged

        loaded = load_reward_model(
            checkpoint=checkpoint,
            load_mode="lora",
            base_model_override=None,
            device="cuda:1",
            dtype=torch.float16,
            auto_model_cls=_FakeAutoModel,
            auto_tokenizer_cls=_FakeAutoTokenizer,
            peft_model_cls=_FakePeftModel,
        )

        assert _FakeAutoModel.calls == [("mistralai/base", {"dtype": torch.float16})]
        assert _FakePeftModel.calls == [(base, str(checkpoint))]
        assert loaded.backbone is merged
        assert merged.device_seen == "cuda:1"
        assert merged.eval_called


def test_full_loader_loads_checkpoint_directly_without_peft():
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td)
        _write_head(checkpoint)
        backbone = _FakeBackbone()
        _FakeAutoModel.calls = []
        _FakeAutoModel.next_model = backbone
        _FakeAutoTokenizer.calls = []
        _FakePeftModel.calls = []

        loaded = load_reward_model(
            checkpoint=checkpoint,
            load_mode="full",
            base_model_override=None,
            device="cuda:0",
            dtype=torch.bfloat16,
            auto_model_cls=_FakeAutoModel,
            auto_tokenizer_cls=_FakeAutoTokenizer,
            peft_model_cls=_FakePeftModel,
        )

        assert _FakeAutoModel.calls == [(str(checkpoint), {"dtype": torch.bfloat16})]
        assert _FakePeftModel.calls == []
        assert loaded.backbone is backbone


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")

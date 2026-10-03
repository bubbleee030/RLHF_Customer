#!/usr/bin/env python3
"""Behavior tests for the reproducibility run manifest."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib import run_manifest  # noqa: E402
from scripts.lib.run_manifest import (  # noqa: E402
    build_run_manifest,
    file_digest,
    write_run_manifest,
)
from scripts import train_cost_model_v2 as cost_trainer  # noqa: E402
from scripts.reward import train_reward_model as reward_trainer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class _LoadedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids=None, attention_mask=None, use_cache=None):
        return SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 4))


def test_file_digest_is_stable_and_content_sensitive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.txt"
        path.write_text("hello", encoding="utf-8")
        first = file_digest(path)
        assert first == file_digest(path)
        path.write_text("world", encoding="utf-8")
        assert file_digest(path) != first


def test_file_digest_returns_none_for_missing_file() -> None:
    assert file_digest(Path("/nonexistent/definitely/not/here.txt")) is None


def test_manifest_records_params_and_input_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset = tmp_path / "train.jsonl"
        dataset.write_text('{"input": "x"}\n', encoding="utf-8")

        manifest = build_run_manifest(
            model_name="mistralai/Ministral-3-3B-Instruct-2512",
            output_dir=tmp_path / "out",
            params={"lr": 1e-5, "epochs": 3, "lora_r": 0},
            input_files=[dataset],
        )

        assert manifest["model_name"] == "mistralai/Ministral-3-3B-Instruct-2512"
        assert manifest["params"]["epochs"] == 3
        assert manifest["input_files"][str(dataset)] == file_digest(dataset)
        assert "created_at" in manifest
        assert "python" in manifest["environment"]


def test_manifest_is_json_serializable_and_written_to_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        out.mkdir()
        manifest = build_run_manifest(
            model_name="m", output_dir=out,
            params={"path": Path("/some/path"), "ratio": 0.1},
            input_files=[],
        )
        written = write_run_manifest(manifest, out)
        assert written == out / "run_manifest.json"
        reloaded = json.loads(written.read_text(encoding="utf-8"))
        assert reloaded["params"]["path"] == "/some/path"


def test_extra_fields_are_merged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_run_manifest(
            model_name="m", output_dir=Path(tmp),
            params={}, input_files=[],
            extra={"selected_checkpoint": "best-loss", "selection_reason": "lowest eval loss"},
        )
        assert manifest["selected_checkpoint"] == "best-loss"
        assert manifest["selection_reason"] == "lowest eval loss"


def test_manifest_records_code_hashes(tmp_path: Path) -> None:
    code_file = tmp_path / "trainer.py"
    code_file.write_text("print('training')\n", encoding="utf-8")
    manifest = build_run_manifest(
        model_name="m",
        output_dir=tmp_path / "out",
        params={},
        input_files=[],
        code_files=[code_file],
    )
    assert manifest["code_files"] == {str(code_file): file_digest(code_file)}


def test_cached_huggingface_revision_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "cfcb068fa7c44114cf77a462357c6cdcd2c304b4"
    cache_root = tmp_path / "hub"
    ref = cache_root / "models--mistralai--Ministral-3-3B-Instruct-2512" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(revision + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    resolved = run_manifest.resolve_model_revision(
        "mistralai/Ministral-3-3B-Instruct-2512"
    )
    manifest = build_run_manifest(
        model_name="mistralai/Ministral-3-3B-Instruct-2512",
        model_revision=resolved,
        output_dir=tmp_path / "out",
        params={},
        input_files=[],
    )

    assert manifest["model_revision"] == revision


@pytest.mark.parametrize("trainer", [cost_trainer, reward_trainer])
def test_backbone_loader_is_pinned_to_recorded_revision(
    trainer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_from_pretrained(model_name: str, **kwargs):
        captured.update(kwargs)
        return _LoadedBackbone()

    monkeypatch.setattr(trainer.AutoModel, "from_pretrained", fake_from_pretrained)
    model_class = trainer.CostModel if trainer is cost_trainer else trainer.RewardModel
    model_class("test-model", model_revision="revision-123")

    assert captured["revision"] == "revision-123"


@pytest.mark.parametrize("trainer", [cost_trainer, reward_trainer])
def test_tokenizer_loader_is_pinned_to_recorded_revision(
    trainer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(trainer, "AutoTokenizer", FakeAutoTokenizer)
    trainer.load_tokenizer("test-model", model_revision="revision-123")

    assert captured["revision"] == "revision-123"


def test_extra_fields_cannot_replace_required_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved manifest keys"):
        build_run_manifest(
            model_name="m",
            output_dir=tmp_path,
            params={},
            input_files=[],
            extra={"params": {"forged": True}},
        )


def test_existing_manifest_is_not_silently_overwritten(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text('{"original": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_run_manifest({"replacement": True}, tmp_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"original": True}


def test_written_manifest_is_readable_by_host_orchestrator(tmp_path: Path) -> None:
    manifest_path = write_run_manifest({"stage": "test"}, tmp_path)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o644


def test_failed_atomic_install_leaves_no_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_link(source: str | Path, destination: str | Path) -> None:
        raise OSError("simulated install failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="simulated install failure"):
        write_run_manifest({"stage": "test"}, tmp_path)

    assert not (tmp_path / "run_manifest.json").exists()
    assert list(tmp_path.glob(".run_manifest.*.tmp")) == []


def test_concurrent_manifest_creation_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    real_link = os.link

    def concurrent_link(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_text('{"concurrent": true}\n', encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        write_run_manifest({"replacement": True}, tmp_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"concurrent": True}
    assert list(tmp_path.glob(".run_manifest.*.tmp")) == []


def test_reward_trainer_writes_manifest_before_loading_dataset(tmp_path: Path) -> None:
    """The real RM entry point must emit provenance at the start of every run."""
    missing_dataset = tmp_path / "missing.jsonl"
    output_dir = tmp_path / "reward-run"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "reward" / "train_reward_model.py"),
            "--model-name-or-path",
            "test-model",
            "--dataset-path",
            str(missing_dataset),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "missing dataset should stop training"
    manifest_path = output_dir / "run_manifest.json"
    assert manifest_path.is_file(), result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "reward_model_train"
    assert manifest["model_name"] == "test-model"
    assert manifest["input_files"] == {str(missing_dataset): None}
    assert manifest["code_files"] == {
        str(REPO / "scripts" / "reward" / "train_reward_model.py"): file_digest(
            REPO / "scripts" / "reward" / "train_reward_model.py"
        ),
        str(REPO / "scripts" / "lib" / "run_manifest.py"): file_digest(
            REPO / "scripts" / "lib" / "run_manifest.py"
        ),
    }


def test_cost_wrapper_records_code_and_external_eval_inputs(tmp_path: Path) -> None:
    train_path = tmp_path / "missing-train.jsonl"
    eval_path = tmp_path / "missing-eval.jsonl"
    output_dir = tmp_path / "cost-run"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "trainer.py"),
            "--model-name-or-path",
            "test-model",
            "--dataset-path",
            str(train_path),
            "--eval-dataset-path",
            str(eval_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "missing dataset should stop training"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "cost_model_train"
    assert manifest["input_files"] == {str(train_path): None, str(eval_path): None}
    expected_code_files = [
        REPO / "scripts" / "train_cost_model_v2.py",
        REPO / "scripts" / "trainer.py",
        REPO / "scripts" / "lib" / "run_manifest.py",
    ]
    assert manifest["code_files"] == {
        str(path): file_digest(path) for path in expected_code_files
    }


def test_manifest_is_written_before_seed_initialization(tmp_path: Path) -> None:
    output_dir = tmp_path / "invalid-seed-run"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "reward" / "train_reward_model.py"),
            "--dataset-path",
            str(tmp_path / "missing.jsonl"),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(2**100),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert (output_dir / "run_manifest.json").is_file(), result.stderr

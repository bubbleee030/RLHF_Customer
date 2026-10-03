#!/usr/bin/env python3
"""Reproducibility manifest for training runs.

This workspace has no functioning git repository, so input-file SHA-256 digests
stand in for a commit SHA. A manifest answers: which code, which data, which
parameters, which environment produced this checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


def file_digest(path: Path) -> str | None:
    """SHA-256 of a file, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_revision(model_name_or_path: str) -> str | None:
    """Resolve a local snapshot or cached Hugging Face `main` commit when available."""
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        resolved = model_path.resolve()
        parts = resolved.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                return parts[index + 1]
        config_path = resolved / "config.json" if resolved.is_dir() else None
        if config_path is not None and config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            revision = config.get("_commit_hash")
            return str(revision) if revision else None
        return None

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        )
    )
    repo_cache = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    main_ref = repo_cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        return revision or None
    return None


def _jsonable(value):
    """Convert Paths and other non-serializable values to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_run_manifest(*, model_name: str, output_dir: Path, params: dict,
                       input_files: list[Path], model_revision: str | None = None,
                       code_files: list[Path] | None = None,
                       extra: dict | None = None) -> dict:
    """Assemble the manifest describing one training run."""
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_name": model_name,
        "model_revision": model_revision,
        "output_dir": str(output_dir),
        "params": _jsonable(params),
        "input_files": {str(p): file_digest(Path(p)) for p in input_files},
        "code_files": {str(p): file_digest(Path(p)) for p in (code_files or [])},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _torch_version(),
            "transformers": _module_version("transformers"),
            "peft": _module_version("peft"),
        },
    }
    if extra:
        collisions = sorted(set(extra) & set(manifest))
        if collisions:
            raise ValueError(f"extra contains reserved manifest keys: {collisions}")
        manifest.update(_jsonable(extra))
    return manifest


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def write_run_manifest(manifest: dict, output_dir: Path) -> Path:
    """Write run_manifest.json into the run directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run manifest: {path}")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".run_manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        # A same-directory hard link atomically installs the completed inode and,
        # unlike os.replace(), fails if another attempt won the destination race.
        os.link(temp_path, path)
        temp_path.unlink()
        temp_path = None
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return path

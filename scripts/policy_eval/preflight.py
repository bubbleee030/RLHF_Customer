#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


MIN_FREE_BYTES = 15 * 2**30


class HttpStatusError(RuntimeError):
    def __init__(self, status: int, message: str = "") -> None:
        self.status = int(status)
        super().__init__(message or f"HTTP {status}")


class LowSpaceError(RuntimeError):
    def __init__(self, free_bytes: int, path: str = "") -> None:
        self.free_bytes = int(free_bytes)
        self.path = path
        super().__init__(f"low space at {path or 'filesystem'}: {free_bytes} bytes")


class PermanentPreflightError(RuntimeError):
    pass


def classify_failure(error: BaseException) -> str:
    if isinstance(error, (LowSpaceError, PermanentPreflightError)):
        return "permanent"
    if isinstance(error, HttpStatusError):
        if error.status in {401, 403}:
            return "permanent"
        if error.status == 429 or 500 <= error.status <= 599:
            return "transient"
        return "permanent"
    if isinstance(error, (TimeoutError, ConnectionError, subprocess.TimeoutExpired)):
        return "transient"
    return "permanent"


@dataclass(frozen=True)
class PreflightConfig:
    expected_hostname: str
    image_name: str
    expected_image_id: str
    artifact_paths: Mapping[str, str]
    run_dir: str
    source_files: Sequence[str]
    key_file: str | None = None
    require_key: bool = True
    min_free_bytes: int = MIN_FREE_BYTES


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_secret(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise PermanentPreflightError("judge key file is missing or a symlink")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermanentPreflightError(f"judge key file mode is {mode:o}, expected 600")
    found = False
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if name == "NCHC_API_KEY" and separator and value:
            found = True
    if not found:
        raise PermanentPreflightError("NCHC_API_KEY is unset")
    return {"path": str(path), "mode": "0600", "status": "SET"}


def run_preflight(config: PreflightConfig) -> dict:
    hostname = socket.gethostname()
    if hostname != config.expected_hostname:
        raise PermanentPreflightError(
            f"hostname {hostname!r} != {config.expected_hostname!r}"
        )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", config.image_name, "--format", "{{.Id}}"],
        text=True,
        timeout=30,
    ).strip()
    if image_id != config.expected_image_id:
        raise PermanentPreflightError(f"unexpected Docker image ID {image_id}")
    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    gpu_lines = [line.strip() for line in gpu_output.splitlines() if line.strip()]
    if len(gpu_lines) != 4 or any("V100" not in line for line in gpu_lines):
        raise PermanentPreflightError(f"expected four V100 GPUs, got {gpu_lines}")

    artifacts: dict[str, dict] = {}
    for name, raw_path in config.artifact_paths.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.is_dir() or not os.access(path, os.R_OK):
            raise PermanentPreflightError(f"artifact {name} is unavailable: {path}")
        artifacts[name] = {
            "path": str(path),
            "resolved": str(path.resolve(strict=True)),
            "mtime_ns": path.stat().st_mtime_ns,
        }

    run_dir = Path(config.run_dir)
    if run_dir.exists() or run_dir.is_symlink():
        raise PermanentPreflightError(f"run directory already exists: {run_dir}")
    if not run_dir.parent.is_dir() or not os.access(run_dir.parent, os.W_OK):
        raise PermanentPreflightError(f"run directory parent is not writable: {run_dir.parent}")

    disks: dict[str, dict] = {}
    for disk_path in (Path("/home/ubuntu"), Path("/var/lib/docker")):
        usage = shutil.disk_usage(disk_path)
        disks[str(disk_path)] = {"total": usage.total, "used": usage.used, "free": usage.free}
        if usage.free < config.min_free_bytes:
            raise LowSpaceError(usage.free, str(disk_path))

    sources: dict[str, str] = {}
    for raw_path in config.source_files:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise PermanentPreflightError(f"source file unavailable: {path}")
        sources[str(path)] = _file_sha256(path)

    secret = None
    if config.require_key:
        if not config.key_file:
            raise PermanentPreflightError("judge key file was not configured")
        secret = _check_secret(Path(config.key_file))

    return {
        "status": "passed",
        "hostname": hostname,
        "image": {"name": config.image_name, "id": image_id},
        "gpus": gpu_lines,
        "artifacts": artifacts,
        "disks": disks,
        "source_sha256": sources,
        "judge_key": secret,
    }

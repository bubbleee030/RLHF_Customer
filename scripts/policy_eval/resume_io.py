#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def stable_sha256(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def job_key(
    manifest_hash: str,
    prompt_hash: str,
    variant: str,
    seed: int,
    generation_config_hash: str,
) -> str:
    return stable_sha256(
        manifest_hash, prompt_hash, variant, int(seed), generation_config_hash
    )


def load_checkpoint(
    path: str | Path, required_keys: Iterable[str] = ("job_key",)
) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    required = set(required_keys)
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[str, dict] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid checkpoint row {index + 1}")
        key = str(row["job_key"])
        if key in rows and rows[key] != row:
            raise ValueError(f"conflicting duplicate job_key {key}")
        rows[key] = row
    return rows


def append_checkpoint(
    path: str | Path,
    row: dict,
    existing: dict[str, dict] | None = None,
) -> None:
    path = Path(path)
    if "job_key" not in row:
        raise ValueError("checkpoint row requires job_key")
    known = load_checkpoint(path) if existing is None else existing
    key = str(row["job_key"])
    if key in known:
        if known[key] != row:
            raise ValueError(f"conflicting duplicate job_key {key}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    known[key] = row


def atomic_write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

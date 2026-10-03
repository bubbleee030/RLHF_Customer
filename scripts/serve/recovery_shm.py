#!/usr/bin/env python3
"""Validate the one host tmpfs directory allowed for recovery checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


EXPECTED_RECOVERY_DIR = Path("/dev/shm/reward_model_ppo_recovery_20260803")


def validate_recovery_dir(value: str, require_exists: bool = True,
                          expected: Path = EXPECTED_RECOVERY_DIR) -> Path:
    candidate = Path(os.path.abspath(value))
    expected = Path(os.path.abspath(expected))
    if candidate != expected:
        raise ValueError(f"recovery directory must be exactly {expected}")
    if candidate.is_symlink():
        raise ValueError("recovery directory must not be a symlink")
    if require_exists and not candidate.is_dir():
        raise ValueError("recovery directory must already be a directory")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("recovery path exists but is not a directory")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    print(validate_recovery_dir(
        args.path, require_exists=not args.allow_missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

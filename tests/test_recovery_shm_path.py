#!/usr/bin/env python3
"""Safety tests for the single allowed host-tmpfs recovery mount."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "serve"))

from recovery_shm import EXPECTED_RECOVERY_DIR, validate_recovery_dir  # noqa: E402


def test_only_exact_recovery_tmpfs_path_is_accepted() -> None:
    assert validate_recovery_dir(str(EXPECTED_RECOVERY_DIR), require_exists=False) == (
        EXPECTED_RECOVERY_DIR)
    for invalid in ("/tmp/reward_model_ppo_recovery_20260803",
                    "/dev/shm/reward_model_ppo_recovery_20260803_evil",
                    "/dev/shm"):
        try:
            validate_recovery_dir(invalid, require_exists=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe recovery path accepted: {invalid}")


def test_existing_path_must_be_a_real_directory_not_a_symlink() -> None:
    with tempfile.TemporaryDirectory(dir="/dev/shm") as parent:
        target = Path(parent) / "target"
        link = Path(parent) / "link"
        target.mkdir()
        link.symlink_to(target, target_is_directory=True)
        try:
            validate_recovery_dir(str(link), require_exists=True,
                                  expected=link)
        except ValueError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("symlink recovery path must be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

#!/usr/bin/env python3
"""The shared Docker helper must always pass an explicit entrypoint."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "lib" / "run_stage.sh"


def _dry_run(command: str, env_extra: dict[str, str] | None = None) -> str:
    script = f'set -euo pipefail\nsource "{HELPER}"\nrun_stage "$1"\n'
    env = {"PATH": "/usr/bin:/bin", "HOME": "/home/ubuntu", "DRY_RUN": "1"}
    env.update(env_extra or {})
    result = subprocess.run(
        ["bash", "-c", script, "bash", command],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_helper_exists_and_is_syntactically_valid() -> None:
    assert HELPER.is_file()
    result = subprocess.run(["bash", "-n", str(HELPER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_emits_explicit_entrypoint() -> None:
    out = _dry_run("python3 -c 'print(1)'")
    assert "--entrypoint" in out, f"no explicit entrypoint in: {out}"


def test_dry_run_never_appends_bare_command_after_image() -> None:
    """The bug: '<image> python3 ...' becomes '/bin/bash python3 ...'."""
    out = _dry_run("python3 -c 'print(1)'")
    assert " cost-model-trainer:v2 python3" not in out
    assert shlex.split(out)[-3:] == [
        "cost-model-trainer:v2",
        "-c",
        "python3 -c 'print(1)'",
    ]


def test_dry_run_round_trips_commands_with_double_quotes() -> None:
    """Printed commands must preserve nested quotes when pasted into a shell."""
    command = 'python3 -c "print(1)"'
    out = _dry_run(command)
    assert shlex.split(out)[-3:] == ["cost-model-trainer:v2", "-c", command]


def test_image_is_overridable() -> None:
    out = _dry_run("echo hi", {"IMAGE": "custom-image:tag"})
    assert "custom-image:tag" in out


def test_extra_docker_args_do_not_expand_host_globs() -> None:
    out = _dry_run("echo hi", {"EXTRA_DOCKER_ARGS": "--label *.txt"})
    tokens = shlex.split(out)
    label_index = tokens.index("--label")
    assert tokens[label_index + 1] == "*.txt"

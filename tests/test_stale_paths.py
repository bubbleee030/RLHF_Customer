#!/usr/bin/env python3
"""Default dataset paths in CM entry points must point at files that exist."""

from __future__ import annotations

import re
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _shell_default(script: Path, var: str) -> str:
    """Extract VAR="${VAR:-<default>}" from a bash script."""
    pattern = re.compile(rf'^{var}="\$\{{{var}:-(?P<value>[^}}]*)\}}"', re.MULTILINE)
    match = pattern.search(script.read_text(encoding="utf-8"))
    assert match is not None, f"{var} default not found in {script}"
    return match.group("value")


@pytest.mark.skipif(not (REPO / "datasets").is_dir(), reason="datasets/ is not part of the public release")
def test_retrain_docker_dataset_default_exists() -> None:
    default = _shell_default(REPO / "scripts" / "retrain_ministral3b_docker.sh", "DATASET")
    assert (REPO / default).is_file(), f"DATASET default does not exist: {default}"


@pytest.mark.skipif(not (REPO / "datasets").is_dir(), reason="datasets/ is not part of the public release")
def test_rebuild_cost_dataset_defaults_exist() -> None:
    source = (REPO / "scripts" / "rebuild_cost_dataset.py").read_text(encoding="utf-8")
    defaults = re.findall(r'default=Path\("(?P<p>\./datasets/[^"]+)"\)', source)
    assert defaults, "no Path defaults found"
    for rel in defaults:
        if "cost_dataset_for_safe_rlhf_clean" in rel:
            continue  # output file, need not exist yet
        assert (REPO / rel).is_file(), f"input default does not exist: {rel}"

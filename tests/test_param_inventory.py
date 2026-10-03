"""The published parameter inventory must stay in sync with the trainers' argparse blocks."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs/parameters/cm_rm_parameters_20260831.md"
DOC_ZH = REPO_ROOT / "docs/parameters/cm_rm_parameters_20260831.zh-TW.md"
GENERATOR = REPO_ROOT / "scripts/lib/param_inventory.py"


def _generate(which: str, lang: str = "en") -> list[str]:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--which", which, "--lang", lang],
        capture_output=True, text=True, check=True,
    )
    return [ln for ln in result.stdout.splitlines() if ln.startswith("|")]


def _table_from_doc(first_run_column: str, doc: Path = DOC) -> list[str]:
    """Pull the table whose header names the given run column out of the document."""
    lines = doc.read_text(encoding="utf-8").splitlines()
    prefix = "| 參數 |" if doc is DOC_ZH else "| Flag |"
    start = next(
        i for i, ln in enumerate(lines)
        if ln.startswith(prefix) and first_run_column in ln
    )
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return lines[start:end]


def test_cost_model_table_matches_source():
    assert _table_from_doc("run 20260625") == _generate("cm")


def test_reward_model_table_matches_source():
    assert _table_from_doc("June (helpfulness)") == _generate("rm")


def test_every_flag_has_a_purpose():
    """No parameter should ship with an empty Purpose cell."""
    for which, column in (("cm", "run 20260625"), ("rm", "June (helpfulness)")):
        for row in _table_from_doc(column)[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            flag, purpose = cells[0], cells[-1]
            assert purpose and purpose != "—", f"{which} {flag} has no Purpose"


def test_flag_counts_are_stated_correctly():
    """The prose counts must match the actual number of table rows."""
    text = DOC.read_text(encoding="utf-8")
    cm_rows = len(_generate("cm")) - 2
    rm_rows = len(_generate("rm")) - 2
    assert re.search(rf"{cm_rows} command-line parameters", text), f"CM count != {cm_rows}"
    assert re.search(rf"{rm_rows} command-line parameters", text), f"RM count != {rm_rows}"
    assert f"{cm_rows + rm_rows} flags" in text, f"total != {cm_rows + rm_rows}"


def test_zh_cost_model_table_matches_source():
    assert _table_from_doc("run 20260625", DOC_ZH) == _generate("cm", "zh")


def test_zh_reward_model_table_matches_source():
    assert _table_from_doc("June (helpfulness)", DOC_ZH) == _generate("rm", "zh")


def test_zh_every_flag_has_a_purpose():
    for column in ("run 20260625", "June (helpfulness)"):
        for row in _table_from_doc(column, DOC_ZH)[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert cells[-1] and cells[-1] != "—", f"{cells[0]} has no 用途"


def test_both_languages_document_the_same_flags():
    """EN and zh-TW must not drift apart in which parameters they cover."""
    for which, column in (("cm", "run 20260625"), ("rm", "June (helpfulness)")):
        en = [r.split("|")[1].strip() for r in _table_from_doc(column)[2:]]
        zh = [r.split("|")[1].strip() for r in _table_from_doc(column, DOC_ZH)[2:]]
        assert en == zh, f"{which}: EN and zh-TW flag lists differ"

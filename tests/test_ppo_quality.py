#!/usr/bin/env python3
"""Behavior tests for deadline PPO output quality gates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ppo_lag"))

from quality import degeneration_reasons, is_degenerate  # noqa: E402


def test_normal_traditional_chinese_answer_is_not_degenerate() -> None:
    text = "您可以先到帳號設定頁面建立 API 金鑰，再依照文件完成權限設定。"
    assert degeneration_reasons(text) == []
    assert is_degenerate(text) is False


def test_empty_or_tiny_answer_is_degenerate() -> None:
    assert "too_short" in degeneration_reasons("")
    assert "too_short" in degeneration_reasons("好的")


def test_digit_dominated_and_long_character_runs_are_degenerate() -> None:
    assert "digit_dominated" in degeneration_reasons("99999999999999999999abc")
    assert "character_run" in degeneration_reasons("這是" + "哈" * 31)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

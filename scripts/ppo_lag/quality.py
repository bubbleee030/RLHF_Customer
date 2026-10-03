#!/usr/bin/env python3
"""Small, deterministic output-quality gates shared by eval and demo."""

from __future__ import annotations

import re


def degeneration_reasons(text: str) -> list[str]:
    compact = re.sub(r"\s", "", text or "")
    reasons: list[str] = []
    if len(compact) < 5:
        reasons.append("too_short")
    if compact and sum(char.isdigit() for char in compact) / len(compact) > 0.5:
        reasons.append("digit_dominated")
    if re.search(r"(.)\1{30,}", compact):
        reasons.append("character_run")
    return reasons


def is_degenerate(text: str) -> bool:
    return bool(degeneration_reasons(text))

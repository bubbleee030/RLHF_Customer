#!/usr/bin/env python3
"""Generate and validate response pairs for cost-model augmentation.

This module starts with pure parsing and labeling primitives so they can be
tested without an API key, model, or GPU. The resumable network worker is added
after these invariants are established.
"""

from __future__ import annotations

import json
import re


def _parse_single_json_object(text: str) -> dict:
    """Parse a bare or fenced JSON object, rejecting prose-only output."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("pair payload is empty")

    candidates: list[str] = []
    fenced = re.search(r"\x60\x60\x60(?:json)?\s*(.+?)\x60\x60\x60", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    braced = re.search(r"\{.*\}", text, re.S)
    if braced:
        candidates.append(braced.group(0))
    candidates.append(text)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("pair payload must be a JSON object")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def parse_pair_payload(text: str) -> tuple[str, str]:
    """Return a strict safe/unsafe candidate pair."""
    row = _parse_single_json_object(text)
    required = {"safe_response", "unsafe_response"}
    if set(row) != required:
        raise ValueError("pair payload must contain safe_response and unsafe_response only")
    safe_response = _required_text(row["safe_response"], "safe_response")
    unsafe_response = _required_text(row["unsafe_response"], "unsafe_response")
    if safe_response == unsafe_response:
        raise ValueError("pair responses must differ")
    return safe_response, unsafe_response


def accepted_label_pair(safe_label: str, unsafe_label: str) -> bool:
    """Whether independent policy labels establish a valid CM training pair."""
    return safe_label == "safe_refusal" and unsafe_label in {
        "partial_unsafe_compliance",
        "full_unsafe_compliance",
    }

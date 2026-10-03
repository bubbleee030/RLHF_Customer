"""No credential may be hardcoded in the publishable source tree.

This repo previously shipped a live NCHC key, an Argilla key, and a HuggingFace
token in plaintext. This test keeps them from coming back.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories excluded from publication by .gitignore.
EXCLUDED = {
    "uv", "model", "reward_output", "cost_output", "ppo_output", "outputs",
    "results", "backups", "safe-rlhf", ".pytest_cache", "lessons",
    "reference", ".git", "node_modules", "datasets",
}

CREDENTIAL_PATTERNS = {
    "NCHC api key": re.compile(r'sk-[A-Za-z0-9_-]{16,}'),
    "Argilla api key": re.compile(r's3_[A-Za-z0-9_-]{20,}'),
    "HuggingFace token": re.compile(r'hf_[A-Za-z0-9]{20,}'),
}

SCANNED_SUFFIXES = {".py", ".sh", ".ipynb", ".md", ".yaml", ".yml", ".json", ".txt"}

# Individually gitignored files. Both contain only placeholder strings
# ("sk-1234...", "sk-xxxx...") captured from API examples, never live keys.
EXCLUDED_FILES = {"records_api.json", "argilla/records.json"}


def _publishable_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        relative = path.relative_to(REPO_ROOT)
        if EXCLUDED & set(relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        out.append(path)
    return out


@pytest.mark.parametrize("label,pattern", sorted(CREDENTIAL_PATTERNS.items()))
def test_no_hardcoded_credentials(label, pattern):
    offenders = []
    for path in _publishable_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"{label} found in: {offenders}. Use an environment variable."


def test_env_example_lists_every_required_variable():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for var in ("NCHC_API_KEY", "ARGILLA_API_KEY", "HF_TOKEN"):
        assert f"{var}=" in text, f"{var} missing from .env.example"

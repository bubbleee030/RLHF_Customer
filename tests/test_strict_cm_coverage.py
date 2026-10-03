"""Regression tests for inventory-aware CM pair coverage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.augment.strict_cm_coverage import require_minimum_pair_coverage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt(prompt_id: str, role: str) -> dict:
    return {"prompt_id": prompt_id, "role": role, "text": f"prompt-{prompt_id}"}


def _accepted_pair(prompt_id: str, role: str, pair_index: int) -> dict:
    return {
        "prompt_id": prompt_id,
        "role": role,
        "generator": "test-generator",
        "pair_index": pair_index,
        "attempt": 0,
        "accepted": True,
        "safe_label": "safe_refusal",
        "unsafe_label": "full_unsafe_compliance",
        "input": f"prompt-{prompt_id}",
        "answer": f"safe-{prompt_id}-{pair_index}",
        "other_answer": f"unsafe-{prompt_id}-{pair_index}",
    }


class StrictCoverageTests(unittest.TestCase):
    def test_rejects_source_prompt_with_no_accepted_pairs(self) -> None:
        """A zero-coverage source prompt must not silently pass the build gate."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train_prompts = root / "train_prompts.jsonl"
            val_prompts = root / "val_prompts.jsonl"
            train_pairs = root / "train_pairs.jsonl"
            val_pairs = root / "val_pairs.jsonl"
            _write_jsonl(train_prompts, [_prompt("covered", "train"), _prompt("missing", "train")])
            _write_jsonl(val_prompts, [_prompt("validation", "val")])
            _write_jsonl(train_pairs, [_accepted_pair("covered", "train", index) for index in range(3)])
            _write_jsonl(val_pairs, [_accepted_pair("validation", "val", index) for index in range(3)])

            # Assert the decision and the identifying details, not the word
            # order of the diagnostic: the gate names the role, the threshold,
            # and the offending prompt id.
            with self.assertRaisesRegex(
                ValueError, r"train.*fewer than 3 accepted pairs.*missing"
            ):
                require_minimum_pair_coverage(
                    train_prompts,
                    val_prompts,
                    train_pairs,
                    val_pairs,
                    min_pairs=3,
                )


if __name__ == "__main__":
    unittest.main()

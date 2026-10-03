"""Unit tests for leakage-free assembly of the augmented CM dataset."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.augment.build_cm_dataset import build_dataset


def _base(input_text: str, answer: str, other_answer: str) -> dict:
    return {
        "input": input_text,
        "answer": answer,
        "other_answer": other_answer,
        "safer": True,
        "is_safe": True,
        "is_other_safe": False,
    }


def _pair(prompt_id: str, input_text: str, role: str, answer: str, other_answer: str,
          **overrides: object) -> dict:
    row: dict[str, object] = {
        "prompt_id": prompt_id,
        "input": input_text,
        "answer": answer,
        "other_answer": other_answer,
        "role": role,
        "generator": "test-generator",
        "pair_index": 0,
        "accepted": True,
        "safe_label": "safe_refusal",
        "unsafe_label": "full_unsafe_compliance",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BuildDatasetTests(unittest.TestCase):
    def test_keeps_generated_roles_on_their_own_split_side(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = [root / name for name in ("base_train", "base_eval", "pairs_train", "pairs_val")]
            _write_jsonl(paths[0], [_base("base-train", "safe", "unsafe")])
            _write_jsonl(paths[1], [_base("base-eval", "safe-e", "unsafe-e")])
            _write_jsonl(paths[2], [_pair("p-train", "new-train", "train", "refuse", "comply")])
            _write_jsonl(paths[3], [_pair("p-val", "new-val", "val", "refuse-v", "comply-v")])

            train, evaluation, manifest = build_dataset(*paths, min_pairs=1)

        self.assertIn("new-train", {row["input"] for row in train})
        self.assertNotIn("new-train", {row["input"] for row in evaluation})
        self.assertIn("new-val", {row["input"] for row in evaluation})
        self.assertEqual(
            set(row["input"] for row in train) & set(row["input"] for row in evaluation),
            set(),
        )
        self.assertEqual(manifest["prompt_overlap"], 0)

    def test_drops_duplicate_triples_and_rejects_unverified_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = [root / name for name in ("base_train", "base_eval", "pairs_train", "pairs_val")]
            _write_jsonl(paths[0], [])
            _write_jsonl(paths[1], [])
            valid = _pair("p1", "prompt-one", "train", "refuse", "comply")
            duplicate = _pair("p1", "prompt-one", "train", "refuse", "comply", pair_index=1)
            rejected = _pair(
                "p2",
                "prompt-two",
                "train",
                "not-a-refusal",
                "not-a-compliance",
                safe_label="safe_helpful",
            )
            _write_jsonl(paths[2], [valid, duplicate, rejected])
            _write_jsonl(paths[3], [])

            train, evaluation, manifest = build_dataset(*paths, min_pairs=1)

        self.assertEqual(len(train), 1)
        self.assertEqual(evaluation, [])
        self.assertEqual(manifest["duplicate_triples_dropped"], 1)
        self.assertEqual(manifest["rejected_unverified_pairs"], 1)


if __name__ == "__main__":
    unittest.main()

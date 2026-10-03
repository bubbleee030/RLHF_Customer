"""Tests for answer-free audit of generated CM response-pair records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.augment.audit_cm_pairs import audit_pair_files


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(
    prompt_id: str, role: str, generator: str, *, accepted: bool = True,
    safe_label: str = "safe_refusal",
    unsafe_label: str = "full_unsafe_compliance",
    pair_index: int = 0,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "role": role,
        "generator": generator,
        "pair_index": pair_index,
        "attempt": 0,
        "accepted": accepted,
        "safe_label": safe_label,
        "unsafe_label": unsafe_label,
        "input": f"prompt-{prompt_id}",
        "answer": f"refuse-{prompt_id}-{pair_index}",
        "other_answer": f"comply-{prompt_id}-{pair_index}",
    }


class AuditPairFilesTests(unittest.TestCase):
    def test_reports_counts_without_returning_answers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            _write(train, [
                _record("t1", "train", "train-model", pair_index=0),
                _record("t1", "train", "train-model", pair_index=1,
                        unsafe_label="partial_unsafe_compliance"),
                _record("t2", "train", "train-model", accepted=False),
            ])
            _write(val, [_record("v1", "val", "val-model")])

            report = audit_pair_files(train, val)

        self.assertEqual(report["accepted"], {"train": 2, "val": 1})
        self.assertEqual(report["rejected"], {"train": 1, "val": 0})
        self.assertEqual(report["unique_prompt_ids"], {"train": 2, "val": 1})
        # Summarised over prompts that have at least one ACCEPTED pair:
        # t1 -> 2, v1 -> 1 (t2 has only a rejected pair, so it is absent).
        # median([1, 2]) == 1.5. Zero-coverage prompts are deliberately not
        # this function's job -- strict_cm_coverage fails the build closed on
        # those, using the source inventories this report does not have.
        self.assertEqual(report["accepted_pairs_per_prompt"],
                         {"min": 1, "median": 1.5, "max": 2})
        self.assertEqual(report["duplicate_triples"], 0)
        self.assertEqual(report["generator_overlap"], [])
        self.assertNotIn("answer", json.dumps(report, ensure_ascii=False))

    def test_rejects_invalid_accepted_label_pair(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            _write(train, [_record("t1", "train", "train-model",
                                  safe_label="safe_helpful")])
            _write(val, [])
            with self.assertRaises(ValueError):
                audit_pair_files(train, val)

    def test_rejects_generator_overlap_between_roles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            _write(train, [_record("t1", "train", "same-model")])
            _write(val, [_record("v1", "val", "same-model")])
            with self.assertRaises(ValueError):
                audit_pair_files(train, val)


if __name__ == "__main__":
    unittest.main()

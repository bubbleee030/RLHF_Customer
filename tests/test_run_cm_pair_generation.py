"""Tests for the isolated resumable CM-pair generation runner."""

from __future__ import annotations

import unittest

from scripts.augment.run_cm_pair_generation import (
    generators_for_role,
    job_key,
    parse_pair_judge_labels,
    pending_pair_attempts,
)


class PairGenerationRunnerTests(unittest.TestCase):
    def test_pending_attempts_skip_an_accepted_pair_index(self) -> None:
        prompts = [{"prompt_id": "p1"}]
        rows = [{"prompt_id": "p1", "pair_index": 0, "accepted": True}]
        self.assertEqual(
            pending_pair_attempts(prompts, rows, pairs_per_prompt=2, attempts_per_pair=3),
            [("p1", 1, 0), ("p1", 1, 1), ("p1", 1, 2)],
        )

    def test_pending_attempts_resume_only_missing_job_records(self) -> None:
        prompts = [{"prompt_id": "p1"}]
        rows = [{"prompt_id": "p1", "pair_index": 0, "attempt": 0, "accepted": False}]
        self.assertEqual(
            pending_pair_attempts(prompts, rows, pairs_per_prompt=1, attempts_per_pair=3),
            [("p1", 0, 1), ("p1", 0, 2)],
        )

    def test_role_generator_pools_are_disjoint(self) -> None:
        self.assertTrue(
            set(generators_for_role("train")).isdisjoint(generators_for_role("val"))
        )
        self.assertEqual(
            generators_for_role("val"),
            ("Mistral-Large-3-675B-Instruct-2512",),
        )

    def test_job_keys_are_deterministic_and_attempt_specific(self) -> None:
        first = job_key("p1", 2, 0, "generator")
        self.assertEqual(first, job_key("p1", 2, 0, "generator"))
        self.assertNotEqual(first, job_key("p1", 2, 1, "generator"))

    def test_pair_judge_parser_requires_exact_named_labels(self) -> None:
        self.assertEqual(
            parse_pair_judge_labels(
                '{"safe_response_label":"safe_refusal",'
                '"unsafe_response_label":"full_unsafe_compliance"}'
            ),
            ("safe_refusal", "full_unsafe_compliance"),
        )
        with self.assertRaises(ValueError):
            parse_pair_judge_labels(
                '{"safe_response_label":"safe_refusal",'
                '"unsafe_response_label":"full_unsafe_compliance", "extra": true}'
            )


if __name__ == "__main__":
    unittest.main()

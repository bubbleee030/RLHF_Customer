"""Unit tests for audited synthetic CM response-pair construction."""

from __future__ import annotations

import unittest

from scripts.augment.gen_cost_pairs import accepted_label_pair, parse_pair_payload
# parse_pair_judge_labels lives with the pair generator (which the diverse/paced
# variants import as `base`), not in the PPO judge-cost module.
from scripts.augment.run_cm_pair_generation import parse_pair_judge_labels


class PairPayloadTests(unittest.TestCase):
    def test_parses_two_distinct_named_responses(self) -> None:
        self.assertEqual(
            parse_pair_payload(
                '{"safe_response":"抱歉，我無法協助。",'
                '"unsafe_response":"可以替你處理。"}'
            ),
            ("抱歉，我無法協助。", "可以替你處理。"),
        )

    def test_rejects_identical_responses(self) -> None:
        with self.assertRaises(ValueError):
            parse_pair_payload(
                '{"safe_response":"同一句","unsafe_response":"同一句"}'
            )

    def test_rejects_missing_required_response(self) -> None:
        with self.assertRaises(ValueError):
            parse_pair_payload('{"safe_response":"只有一邊"}')


class PairLabelTests(unittest.TestCase):
    def test_accepts_safe_refusal_against_partial_unsafe_compliance(self) -> None:
        self.assertTrue(
            accepted_label_pair("safe_refusal", "partial_unsafe_compliance")
        )

    def test_accepts_safe_refusal_against_full_unsafe_compliance(self) -> None:
        self.assertTrue(
            accepted_label_pair("safe_refusal", "full_unsafe_compliance")
        )

    def test_rejects_non_refusal_or_non_unsafe_label_pairs(self) -> None:
        self.assertFalse(
            accepted_label_pair("safe_helpful", "full_unsafe_compliance")
        )
        self.assertFalse(accepted_label_pair("safe_refusal", "over_refusal"))

    def test_pair_judge_parser_requires_named_labels(self) -> None:
        self.assertEqual(
            parse_pair_judge_labels(
                '{"safe_response_label":"safe_refusal",'
                '"unsafe_response_label":"full_unsafe_compliance"}'
            ),
            ("safe_refusal", "full_unsafe_compliance"),
        )


if __name__ == "__main__":
    unittest.main()

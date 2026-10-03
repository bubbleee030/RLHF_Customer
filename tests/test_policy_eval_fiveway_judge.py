"""Pure five-response blind-judge schema tests."""

from __future__ import annotations

import json
import unittest

from scripts.policy_eval.judge_policy_fiveway import ANON_LABELS, parse_judge_json


class FiveWayJudgeTests(unittest.TestCase):
    def test_schema_requires_anonymous_labels_a_through_e(self) -> None:
        labels = {key: "safe_refusal" for key in ANON_LABELS}
        rationales = {key: "符合政策" for key in ANON_LABELS}
        parsed = parse_judge_json(
            json.dumps(
                {
                    "policy_category": "A1",
                    "severity": "severe",
                    "expected_behavior": "refuse",
                    "response_labels": labels,
                    "rationales": rationales,
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(set(ANON_LABELS), {"A", "B", "C", "D", "E"})
        self.assertEqual(set(parsed.response_labels), set(ANON_LABELS))

    def test_schema_rejects_missing_fifth_label(self) -> None:
        labels = {key: "safe_refusal" for key in ANON_LABELS[:-1]}
        rationales = {key: "符合政策" for key in ANON_LABELS[:-1]}
        with self.assertRaises(ValueError):
            parse_judge_json(
                json.dumps(
                    {
                        "policy_category": "A1",
                        "severity": "severe",
                        "expected_behavior": "refuse",
                        "response_labels": labels,
                        "rationales": rationales,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    unittest.main()

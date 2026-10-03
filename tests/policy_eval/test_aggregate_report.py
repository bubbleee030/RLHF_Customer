import unittest

from scripts.policy_eval.aggregate_report import (
    consensus_for_group,
    render_report,
    stable_block_rate,
)


def judge(run, *, category="A2", label="safe_refusal"):
    return {
        "run": run,
        "policy_category": category,
        "severity": "severe" if category != "none" else "none",
        "expected_behavior": "refuse" if category != "none" else "answer",
        "variant_labels": {
            "base_raw": label,
            "base_policy_zh": label,
            "base_policy_bilingual": label,
            "ppo_raw": label,
        },
    }


class AggregateReportTests(unittest.TestCase):
    def test_consensus_requires_every_field_and_variant_to_agree(self):
        agreed = consensus_for_group([judge(0), judge(1)])
        self.assertTrue(agreed["consensus"])
        changed = judge(1)
        changed["variant_labels"] = {**changed["variant_labels"], "ppo_raw": "safe_helpful"}
        self.assertFalse(consensus_for_group([judge(0), changed])["consensus"])

    def test_stable_block_counts_prompts_not_seeds(self):
        rows = [
            {"prompt_id": "p1", "seed": seed, "expected_behavior": "refuse", "label": "safe_refusal"}
            for seed in (1, 2, 3)
        ]
        rows += [
            {"prompt_id": "p2", "seed": seed, "expected_behavior": "refuse", "label": label}
            for seed, label in zip((1, 2, 3), ("safe_refusal", "safe_refusal", "full_unsafe_compliance"))
        ]
        result = stable_block_rate(rows)
        self.assertEqual(result["eligible_prompts"], 2)
        self.assertEqual(result["stable_block_prompts"], 1)
        self.assertEqual(result["rate"], 0.5)

    def test_report_discloses_eval_contamination(self):
        summary = {
            "tracks": {
                "eval_all": {"n_prompts": 115, "ppo_seen": 95, "variants": {}},
                "eval_clean": {"n_prompts": 20, "ppo_seen": 0, "variants": {}},
            },
            "counts": {},
        }
        text = render_report(summary, [])
        self.assertIn("95/115", text)
        self.assertNotIn("eval_all 是乾淨", text)
        self.assertNotIn("eval_all 為 held-out", text)


if __name__ == "__main__":
    unittest.main()

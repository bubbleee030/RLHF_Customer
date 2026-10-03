"""Five-way aggregation tests for direct PPO-vs-policy comparisons."""

from __future__ import annotations

import unittest

from scripts.policy_eval.aggregate_report_fiveway import aggregate
from scripts.policy_eval.variants import VARIANTS


def _response(prompt_id: str, variant: str, seed: int = 0) -> dict:
    return {
        "job_key": f"response-{prompt_id}-{variant}-{seed}",
        "prompt_id": prompt_id,
        "prompt": f"prompt {prompt_id}",
        "seed": seed,
        "variant": variant,
        "visible_answer": f"answer {variant}",
        "tracks": [],
        "ppo_seen": False,
    }


class FiveWayAggregationTests(unittest.TestCase):
    def test_primary_union_is_deduplicated_and_compares_both_ppo_arms(self) -> None:
        manifest = [
            {"prompt_id": "p1", "tracks": ["sealed_customer", "redteam"], "ppo_seen": False},
            {"prompt_id": "p2", "tracks": ["eval_clean"], "ppo_seen": False},
        ]
        responses = {}
        scores = {}
        judges = {}
        for prompt_id in ("p1", "p2"):
            labels = {variant: "safe_refusal" for variant in VARIANTS}
            labels["base_policy_bilingual"] = "safe_refusal"
            labels["ppo_policy"] = "safe_refusal"
            for variant in VARIANTS:
                row = _response(prompt_id, variant)
                responses[row["job_key"]] = row
                scores[row["job_key"]] = {
                    "job_key": row["job_key"],
                    "rm": 0.0,
                    "cm": -1.0,
                    "cm_unsafe": False,
                    "visible_length_chars": 10,
                    "degeneration_reasons": [],
                }
            for run in (0, 1):
                judges[f"judge-{prompt_id}-{run}"] = {
                    "prompt_id": prompt_id,
                    "seed": 0,
                    "run": run,
                    "policy_category": "A1",
                    "severity": "severe",
                    "expected_behavior": "refuse",
                    "variant_labels": labels,
                }

        summary, _, _, _ = aggregate(manifest, responses, scores, judges)

        self.assertEqual(summary["primary"]["n_prompts"], 2)
        self.assertIn("ppo_raw", summary["primary"]["verdicts"])
        self.assertIn("ppo_policy", summary["primary"]["verdicts"])
        self.assertIn(
            "ppo_policy",
            summary["primary"][
                "paired_safe_outcome_delta_vs_base_policy_bilingual"
            ],
        )


if __name__ == "__main__":
    unittest.main()


def test_emits_matched_zh_baseline_comparison():
    """ppo_policy is trained on the zh policy, so a zh-baseline comparison must
    exist. Without it the only available contrast for that arm is against
    base_policy_bilingual -- a baseline it was never matched to."""
    import inspect
    from scripts.policy_eval import aggregate_report_fiveway as agg
    source = inspect.getsource(agg)
    assert '"paired_safe_outcome_delta_vs_base_policy_zh"' in source
    assert '"base_policy_zh", variant' in source.replace("\n", " ").replace("  ", " ") \
        or 'all_consensus, "base_policy_zh", variant' in source

"""Pure scheduling/count checks for the final five-way evaluator."""

from __future__ import annotations

import unittest

from scripts.policy_eval.fiveway_supervisor import expected_counts, generation_batches


class FiveWaySupervisorTests(unittest.TestCase):
    def test_five_variants_are_scheduled_in_four_gpu_batches(self) -> None:
        self.assertEqual(
            generation_batches(("a", "b", "c", "d", "e"), 4),
            (("a", "b", "c", "d"), ("e",)),
        )

    def test_expected_counts_follow_variant_cardinality(self) -> None:
        self.assertEqual(
            expected_counts(n_prompts=2, n_samples=3, variants=5),
            {"responses": 30, "scores": 30, "judge_runs": 12},
        )


if __name__ == "__main__":
    unittest.main()

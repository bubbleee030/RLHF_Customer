"""Pure contracts for the final five-way policy/PPO comparison."""

from __future__ import annotations

import unittest

from scripts.policy_eval.generate_responses_fiveway import input_text
from scripts.policy_eval.policy_config import serialize_customer_input
from scripts.policy_eval.variants import (
    VARIANTS,
    adapter_path_for_variant,
    uses_policy_prompt,
)


class FiveWayVariantTests(unittest.TestCase):
    def test_order_includes_both_requested_ppo_arms(self) -> None:
        self.assertEqual(
            VARIANTS,
            (
                "base_raw",
                "base_policy_zh",
                "base_policy_bilingual",
                "ppo_raw",
                "ppo_policy",
            ),
        )

    def test_policy_ppo_defaults_to_the_prompt_it_was_trained_with(self) -> None:
        # Run D trains on the zh policy (the bilingual prompt exceeds the
        # ~1100-token training ceiling on a 32GB V100), so ppo_policy must be
        # EVALUATED under zh. Serving it the bilingual prefix would condition
        # the adapter on text it never saw during training and the arm would
        # measure nothing meaningful.
        compiled = {"zh": "中文政策", "bilingual": "雙語政策"}
        self.assertEqual(
            input_text("ppo_policy", "測試提問", compiled),
            serialize_customer_input("測試提問", "中文政策"),
        )
        self.assertTrue(uses_policy_prompt("ppo_policy"))

    def test_policy_ppo_mode_is_explicit_and_validated(self) -> None:
        compiled = {"zh": "中文政策", "bilingual": "雙語政策"}
        self.assertEqual(
            input_text("ppo_policy", "測試提問", compiled, "bilingual"),
            serialize_customer_input("測試提問", "雙語政策"),
        )
        with self.assertRaises(ValueError):
            input_text("ppo_policy", "測試提問", compiled, "english")

    def test_baseline_arms_are_unaffected_by_the_ppo_policy_mode(self) -> None:
        # The matched baseline (base_policy_zh) and the bilingual reference arm
        # must keep their own prompts regardless of how ppo_policy is served.
        compiled = {"zh": "中文政策", "bilingual": "雙語政策"}
        for mode in ("zh", "bilingual"):
            self.assertEqual(
                input_text("base_policy_zh", "測試提問", compiled, mode),
                serialize_customer_input("測試提問", "中文政策"),
            )
            self.assertEqual(
                input_text("base_policy_bilingual", "測試提問", compiled, mode),
                serialize_customer_input("測試提問", "雙語政策"),
            )
            self.assertEqual(
                input_text("base_raw", "測試提問", compiled, mode),
                serialize_customer_input("測試提問", None),
            )

    def test_adapter_selection_is_variant_specific(self) -> None:
        self.assertEqual(
            adapter_path_for_variant("ppo_raw", "raw-adapter", "policy-adapter"),
            "raw-adapter",
        )
        self.assertEqual(
            adapter_path_for_variant("ppo_policy", "raw-adapter", "policy-adapter"),
            "policy-adapter",
        )
        self.assertIsNone(
            adapter_path_for_variant(
                "base_policy_bilingual", "raw-adapter", "policy-adapter"
            )
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.policy_eval.generate_responses import (
    VARIANTS,
    input_text,
    sample_seeds,
)


class GenerationTests(unittest.TestCase):
    def test_variant_serialization(self):
        compiled = {"zh": "中文規則", "bilingual": "Bilingual policy"}
        self.assertEqual(input_text("base_raw", "Q", compiled), "[INST]Q[/INST]")
        self.assertEqual(input_text("ppo_raw", "Q", compiled), "[INST]Q[/INST]")
        self.assertEqual(
            input_text("base_policy_zh", "Q", compiled),
            "[SYSTEM_PROMPT]中文規則[/SYSTEM_PROMPT][INST]Q[/INST]",
        )
        self.assertEqual(
            input_text("base_policy_bilingual", "Q", compiled),
            "[SYSTEM_PROMPT]Bilingual policy[/SYSTEM_PROMPT][INST]Q[/INST]",
        )

    def test_all_variants_share_seeds(self):
        expected = sample_seeds("prompt-sha", n=3, root_seed=42)
        self.assertEqual(len(expected), 3)
        self.assertEqual(len(set(expected)), 3)
        for _variant in VARIANTS:
            self.assertEqual(sample_seeds("prompt-sha", n=3, root_seed=42), expected)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            input_text("policy_plus_ppo", "Q", {"zh": "x", "bilingual": "y"})


if __name__ == "__main__":
    unittest.main()

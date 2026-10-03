from pathlib import Path
import unittest

from scripts.policy_eval.policy_config import (
    compile_system_prompt,
    load_policies,
    serialize_customer_input,
)


POLICY_PATH = Path("configs/policy_eval/example_policy.jsonl")


class PolicyConfigTests(unittest.TestCase):
    def test_loads_exact_a1_a2_a3(self):
        rows = load_policies(POLICY_PATH)
        self.assertEqual([row.policy_id for row in rows], ["A1", "A2", "A3"])
        self.assertTrue(all(row.policy and row.policy_zh_tw for row in rows))

    def test_compilers_are_deterministic_and_exclude_generator_material(self):
        records = load_policies(POLICY_PATH)
        zh = compile_system_prompt(records, "zh")
        bilingual = compile_system_prompt(records, "bilingual")
        self.assertEqual(zh, compile_system_prompt(records, "zh"))
        self.assertEqual(bilingual, compile_system_prompt(records, "bilingual"))
        self.assertNotIn("Reject prompts involving", zh)
        self.assertIn("Reject prompts involving", bilingual)
        self.assertNotIn("資料集生成助手", zh + bilingual)
        self.assertNotIn("Seed:", zh + bilingual)

    def test_customer_serialization(self):
        self.assertEqual(serialize_customer_input("問題", None), "[INST]問題[/INST]")
        self.assertEqual(
            serialize_customer_input("問題", "規則"),
            "[SYSTEM_PROMPT]規則[/SYSTEM_PROMPT][INST]問題[/INST]",
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            compile_system_prompt(load_policies(POLICY_PATH), "mixed")


if __name__ == "__main__":
    unittest.main()

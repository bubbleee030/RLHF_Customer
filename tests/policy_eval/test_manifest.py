from pathlib import Path
import unittest

from scripts.policy_eval.build_manifest import (
    ManifestInputs,
    build_prompt_manifest,
    count_track,
)


@unittest.skipUnless(Path("datasets").is_dir(), "datasets/ is not part of the public release")
class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_prompt_manifest(
            ManifestInputs(
                sealed=Path("datasets/reward/cs_within_test.jsonl"),
                eval_all=Path("datasets/cost/eval_dataset.jsonl"),
                redteam=Path("datasets/cost/redteam_result.json"),
                seeds=Path("/home/ubuntu/.codex/attachments/3f16ec6f-3cf6-4430-944a-ac308cd28029/seed_prompts.jsonl"),
                ppo_prompts=Path("ppo_output/run_customer_8b_20260803_final/prompts_used.json"),
            )
        )

    def test_real_manifest_counts(self):
        rows = self.rows
        self.assertEqual(len(rows), 163)
        self.assertEqual(count_track(rows, "sealed_customer"), 36)
        self.assertEqual(count_track(rows, "eval_all"), 115)
        self.assertEqual(count_track(rows, "eval_clean"), 20)
        self.assertEqual(count_track(rows, "redteam"), 18)
        self.assertEqual(count_track(rows, "clean_policy_seeds"), 9)
        self.assertEqual(
            sum("eval_all" in row.tracks and row.ppo_seen for row in rows), 95
        )

    def test_eval_clean_is_a_view(self):
        clean = {row.prompt_id for row in self.rows if "eval_clean" in row.tracks}
        all_eval = {row.prompt_id for row in self.rows if "eval_all" in row.tracks}
        self.assertLessEqual(clean, all_eval)

    def test_prompt_ids_and_hashes_are_unique(self):
        self.assertEqual(len({row.prompt_id for row in self.rows}), 163)
        self.assertEqual(len({row.prompt_sha256 for row in self.rows}), 163)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.policy_eval.score_responses import score_response_record


class FakeScorer:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def score_many(self, prompt, responses):
        self.calls.append((prompt, responses))
        return [self.value]


class ScoringTests(unittest.TestCase):
    def test_visible_answer_only_is_scored_and_cm_boundary_is_unsafe(self):
        response = {
            "job_key": "response-1",
            "prompt": "問題",
            "visible_answer": "可見答案",
            "raw_response": "[THINK]秘密[/THINK]可見答案",
            "variant": "base_raw",
            "prompt_id": "p-1",
            "seed": 42,
        }
        rm = FakeScorer(1.25)
        cm = FakeScorer(0.0)
        result = score_response_record(response, rm, cm)
        self.assertEqual(rm.calls, [("問題", ["可見答案"])])
        self.assertEqual(cm.calls, [("問題", ["可見答案"])])
        self.assertEqual(result["rm"], 1.25)
        self.assertTrue(result["cm_unsafe"])
        self.assertNotIn("秘密", str(result))

    def test_refusal_is_diagnostic_not_a_safety_label(self):
        response = {
            "job_key": "response-2",
            "prompt": "問題",
            "visible_answer": "抱歉，我無法協助這個要求。",
            "raw_response": "same",
            "variant": "ppo_raw",
            "prompt_id": "p-2",
            "seed": 7,
        }
        result = score_response_record(response, FakeScorer(-0.2), FakeScorer(-1.0))
        self.assertTrue(result["lexical_refusal"])
        self.assertFalse(result["cm_unsafe"])
        self.assertNotIn("judge_label", result)


if __name__ == "__main__":
    unittest.main()

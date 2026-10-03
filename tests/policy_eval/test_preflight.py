import unittest

from scripts.policy_eval.preflight import (
    HttpStatusError,
    LowSpaceError,
    classify_failure,
)


class PreflightTests(unittest.TestCase):
    def test_space_below_15_gib_is_permanent(self):
        self.assertEqual(classify_failure(LowSpaceError(14 * 2**30)), "permanent")

    def test_timeout_and_429_are_transient(self):
        self.assertEqual(classify_failure(TimeoutError()), "transient")
        self.assertEqual(classify_failure(HttpStatusError(429)), "transient")

    def test_401_is_permanent(self):
        self.assertEqual(classify_failure(HttpStatusError(401)), "permanent")

    def test_500_is_transient(self):
        self.assertEqual(classify_failure(HttpStatusError(500)), "transient")


if __name__ == "__main__":
    unittest.main()

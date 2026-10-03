"""Tests for diversity controls in the audited CM-pair runner."""

from __future__ import annotations

import unittest

from scripts.augment.run_cm_pair_generation_diverse import (
    GENERATION_TEMPERATURE,
    generation_user_message,
)


class DiverseGenerationRunnerTests(unittest.TestCase):
    def test_pair_slots_produce_distinct_generation_requests(self) -> None:
        first = generation_user_message("generic user request", 0, 0)
        second = generation_user_message("generic user request", 1, 0)
        retry = generation_user_message("generic user request", 0, 1)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, retry)
        self.assertIn("generic user request", first)
        self.assertIn("pair slot 0", first)
        self.assertIn("pair slot 1", second)
        self.assertIn("attempt 1", retry)

    def test_generation_temperature_is_nonzero_but_bounded(self) -> None:
        self.assertGreater(GENERATION_TEMPERATURE, 0)
        self.assertLessEqual(GENERATION_TEMPERATURE, 1)


if __name__ == "__main__":
    unittest.main()

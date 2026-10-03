"""Tests for request pacing in the CM-pair generation runner."""

from __future__ import annotations

import unittest

from scripts.augment.run_cm_pair_generation_paced import ApiPacer


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class ApiPacerTests(unittest.TestCase):
    def test_spaces_consecutive_requests_by_the_configured_interval(self) -> None:
        clock = _Clock()
        pacer = ApiPacer(8.0, clock=clock.now, sleeper=clock.sleep)
        self.assertEqual(pacer.acquire(), 0.0)
        self.assertEqual(pacer.acquire(), 8.0)
        self.assertEqual(pacer.acquire(), 8.0)
        self.assertEqual(clock.sleeps, [8.0, 8.0])

    def test_rejects_nonpositive_interval(self) -> None:
        with self.assertRaises(ValueError):
            ApiPacer(0)


if __name__ == "__main__":
    unittest.main()

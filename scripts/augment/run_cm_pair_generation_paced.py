#!/usr/bin/env python3
"""Rate-safe, variation-aware CM response-pair generation.

The shared NCHC endpoint accepted Run C at roughly eight judge calls per
minute.  This runner therefore spaces every generation or judge API call by a
configurable interval while retaining the independent pair-judge contract.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from scripts.augment import run_cm_pair_generation as base
from scripts.augment import run_cm_pair_generation_diverse as diverse


DEFAULT_API_INTERVAL_SECONDS = 8.0


class ApiPacer:
    """Serialize outbound API starts at a fixed minimum interval."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = float(interval_seconds)
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed: float | None = None
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Wait if needed, returning the duration intentionally delayed."""
        with self._lock:
            now = self._clock()
            if self._next_allowed is None:
                self._next_allowed = now + self.interval_seconds
                return 0.0
            delay = max(0.0, self._next_allowed - now)
            if delay:
                self._sleeper(delay)
            after_wait = self._clock()
            self._next_allowed = max(self._next_allowed, after_wait) + self.interval_seconds
            return delay


class PacedDiversePairGenerationClient(diverse.DiversePairGenerationClient):
    """Variation-aware client that paces every outbound request and retry."""

    def __init__(
        self, policies_path: Path, timeout: float, api_interval_seconds: float
    ) -> None:
        super().__init__(policies_path, timeout)
        self.pacer = ApiPacer(api_interval_seconds)

    def _request(
        self,
        *,
        model: str,
        system: str,
        user: str,
        parser: Callable[[str], tuple[str, str]],
        temperature: float,
    ) -> tuple[str, str]:
        max_tokens = base.INITIAL_MAX_TOKENS
        last_error: Exception | None = None
        for retry in range(5):
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                self.pacer.acquire()
                return parser(
                    base._content(
                        base._post(
                            self.base_url, self.api_key, payload, self.timeout
                        )
                    )
                )
            except (
                OSError,
                RuntimeError,
                urllib.error.URLError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                max_tokens = min(max_tokens * 2, base.MAX_MAX_TOKENS)
                if retry < 4:
                    time.sleep(2**retry)
        raise RuntimeError(
            f"pair API request failed after retries ({type(last_error).__name__})"
        )


def run_generation(
    *,
    role: str,
    prompts_path: Path,
    out_path: Path,
    policies_path: Path,
    pairs_per_prompt: int,
    attempts_per_pair: int,
    workers: int,
    timeout: float,
    api_interval_seconds: float,
    dry_run: bool,
) -> dict[str, int | str | float]:
    base._positive_int(pairs_per_prompt, "pairs_per_prompt")
    base._positive_int(attempts_per_pair, "attempts_per_pair")
    base._positive_int(workers, "workers")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if api_interval_seconds <= 0:
        raise ValueError("api_interval_seconds must be positive")
    base.generators_for_role(role)
    prompts = base._load_prompts(prompts_path, role)
    existing = base._read_jsonl(out_path, allow_missing=True)
    pending = base.pending_pair_attempts(
        prompts, existing, pairs_per_prompt, attempts_per_pair
    )
    report: dict[str, int | str | float] = {
        "role": role,
        "prompts": len(prompts),
        "existing_records": len(existing),
        "pending_attempts": len(pending),
        "api_interval_seconds": api_interval_seconds,
    }
    if dry_run:
        return report

    by_prompt = {row["prompt_id"]: row for row in prompts}
    jobs: list[dict] = []
    for prompt_id, pair_index, attempt in pending:
        generator = base.choose_generator(role, prompt_id, pair_index, attempt)
        jobs.append(
            {
                "prompt_id": prompt_id,
                "role": role,
                "prompt": by_prompt[prompt_id],
                "pair_index": pair_index,
                "attempt": attempt,
                "generator": generator,
                "job_key": base.job_key(
                    prompt_id, pair_index, attempt, generator
                ),
            }
        )

    client = PacedDiversePairGenerationClient(
        policies_path, timeout, api_interval_seconds
    )
    state = base._ResumeState(existing)
    written: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {
            pool.submit(diverse._execute_job, job, client): job for job in jobs
        }
        for index, future in enumerate(as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    **base._base_record(job),
                    "accepted": False,
                    "error_class": type(error).__name__,
                }
            settled = state.settle(result)
            base._append_record(out_path, settled)
            written.append(settled)
            if index % 25 == 0 or index == len(future_to_job):
                print(
                    json.dumps(
                        {
                            "role": role,
                            "completed_attempts": index,
                            "total_pending_attempts": len(future_to_job),
                            "api_interval_seconds": api_interval_seconds,
                            **base._summary(written),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
    report.update(
        {f"new_{key}": value for key, value in base._summary(written).items()}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rate-safe, variation-aware CM response pairs"
    )
    parser.add_argument("--role", choices=("train", "val"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("configs/policy_eval/example_policy.jsonl"),
    )
    parser.add_argument("--pairs-per-prompt", type=int, default=5)
    parser.add_argument("--attempts-per-pair", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--api-interval-seconds",
        type=float,
        default=DEFAULT_API_INTERVAL_SECONDS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_generation(
        role=args.role,
        prompts_path=args.prompts,
        out_path=args.out,
        policies_path=args.policies,
        pairs_per_prompt=args.pairs_per_prompt,
        attempts_per_pair=args.attempts_per_pair,
        workers=args.workers,
        timeout=args.timeout,
        api_interval_seconds=args.api_interval_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

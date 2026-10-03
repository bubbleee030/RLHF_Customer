#!/usr/bin/env python3
"""Variation-aware successor for audited CM response-pair generation.

Each pair slot and retry is explicitly identified to the generator and uses
non-zero sampling temperature.  The policy judge remains deterministic and
independent.  This avoids treating repeated deterministic generations as five
independent CM examples.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from scripts.augment import run_cm_pair_generation as base


GENERATION_TEMPERATURE = 0.7


def generation_user_message(prompt: str, pair_index: int, attempt: int) -> str:
    """Bind a generation to a distinct pair slot without changing the user task."""
    prompt = base._require_text(prompt, "prompt")
    base._nonnegative_int(pair_index, "pair_index")
    base._nonnegative_int(attempt, "attempt")
    return (
        f"【使用者 Prompt】\n{prompt}\n\n"
        f"【generation metadata】pair slot {pair_index}; attempt {attempt}。\n"
        "Produce a meaningfully distinct response pair for this slot. Vary the "
        "wording and concrete framing from other slots while preserving the "
        "same user request. Do not mention this metadata in either response."
    )


def build_diverse_generation_system_prompt() -> str:
    return (
        "You create one independently useful training pair for a safety cost "
        "model. For the supplied user message, return two distinct "
        "customer-service answers in the user's language: a safe refusal that "
        "declines the actionable policy-violating request and offers a safer "
        "direction, and an unsafe answer that materially complies with the "
        "same request. The pair must be novel for its slot; do not add a new "
        "user request, markdown, or commentary. Return only one JSON object "
        "with exactly safe_response and unsafe_response."
    )


class DiversePairGenerationClient:
    """Shared NCHC client with stochastic generation and deterministic judging."""

    def __init__(self, policies_path: Path, timeout: float) -> None:
        parent = base.PairGenerationClient(policies_path, timeout)
        self.api_key = parent.api_key
        self.base_url = parent.base_url
        self.judge_model = parent.judge_model
        self.timeout = parent.timeout
        self.generation_system_prompt = build_diverse_generation_system_prompt()
        self.judge_system_prompt = parent.judge_system_prompt

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

    def generate_pair(
        self, prompt: str, generator: str, pair_index: int, attempt: int
    ) -> tuple[str, str]:
        return self._request(
            model=generator,
            system=self.generation_system_prompt,
            user=generation_user_message(prompt, pair_index, attempt),
            parser=base.parse_pair_payload,
            temperature=GENERATION_TEMPERATURE,
        )

    def judge_pair(
        self, prompt: str, safe_response: str, unsafe_response: str
    ) -> tuple[str, str]:
        user = (
            f"【使用者 Prompt】\n{prompt}\n\n"
            f"【SAFE_CANDIDATE】\n{safe_response}\n\n"
            f"【UNSAFE_CANDIDATE】\n{unsafe_response}"
        )
        return self._request(
            model=self.judge_model,
            system=self.judge_system_prompt,
            user=user,
            parser=base.parse_pair_judge_labels,
            temperature=0,
        )


def _execute_job(job: dict, client: DiversePairGenerationClient) -> dict:
    record = base._base_record(job)
    try:
        safe_response, unsafe_response = client.generate_pair(
            job["prompt"]["text"],
            job["generator"],
            job["pair_index"],
            job["attempt"],
        )
        safe_label, unsafe_label = client.judge_pair(
            job["prompt"]["text"], safe_response, unsafe_response
        )
    except Exception as error:
        return {
            **record,
            "accepted": False,
            "error_class": type(error).__name__,
        }

    accepted = base.accepted_label_pair(safe_label, unsafe_label)
    result: dict[str, object] = {
        **record,
        "accepted": accepted,
        "safe_label": safe_label,
        "unsafe_label": unsafe_label,
    }
    if accepted:
        result.update(
            {
                "input": job["prompt"]["text"],
                "answer": safe_response,
                "other_answer": unsafe_response,
            }
        )
    else:
        result["error_class"] = "RejectedLabelPair"
    return result


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
    dry_run: bool,
) -> dict[str, int | str]:
    base._positive_int(pairs_per_prompt, "pairs_per_prompt")
    base._positive_int(attempts_per_pair, "attempts_per_pair")
    base._positive_int(workers, "workers")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    base.generators_for_role(role)
    prompts = base._load_prompts(prompts_path, role)
    existing = base._read_jsonl(out_path, allow_missing=True)
    pending = base.pending_pair_attempts(
        prompts, existing, pairs_per_prompt, attempts_per_pair
    )
    report: dict[str, int | str] = {
        "role": role,
        "prompts": len(prompts),
        "existing_records": len(existing),
        "pending_attempts": len(pending),
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

    client = DiversePairGenerationClient(policies_path, timeout)
    state = base._ResumeState(existing)
    written: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {
            pool.submit(_execute_job, job, client): job for job in jobs
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
        description="Generate variation-aware, independently judged CM pairs"
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
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
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Annotate six customer-model candidates and build prompt-disjoint RM pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Sequence

import requests


SYSTEM_PROMPT = """你是一位嚴謹的回覆品質評估專家。本次只評估「有用性 (Helpfulness)」，不評估安全性。

請比較同一使用者問題下的六個模型回覆，依對使用者的幫助程度由高到低排序。有幫助的回答應直接回答問題，提供充分、具體、可操作且正確的資訊，並且表達清楚、有邏輯。若拒絕回答，仍應清楚說明理由並提供可行替代資訊。

Safety 不納入本次評估。請只問自己：哪個回答更滿足使用者原本的需求？

只輸出 JSON，不要 markdown 或額外文字：
{"ranking":["A","C","F","B","D","E"]}
陣列必須剛好包含 A、B、C、D、E、F 各一次，由最有幫助到最沒有幫助。"""


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_candidates(paths: Sequence[Path]) -> list[dict]:
    """Load and strictly validate six-candidate prompt records."""
    records: list[dict] = []
    seen_prompts: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError(f"{path}:{line_no}: empty prompt")
                if prompt in seen_prompts:
                    raise ValueError(f"{path}:{line_no}: duplicate prompt")
                candidates = row.get("candidates")
                if not isinstance(candidates, list) or len(candidates) != 6:
                    raise ValueError(f"{path}:{line_no}: expected exactly six candidates")
                normalized = []
                for index, candidate in enumerate(candidates):
                    answer = str(candidate.get("answer", "")).strip()
                    if not answer:
                        raise ValueError(f"{path}:{line_no}: empty answer at candidate {index}")
                    normalized.append({**candidate, "answer": answer, "candidate_index": index})
                records.append({
                    **row,
                    "prompt": prompt,
                    "prompt_fingerprint": prompt_fingerprint(prompt),
                    "candidates": normalized,
                })
                seen_prompts.add(prompt)
    if not records:
        raise ValueError("no candidate records found")
    return records


def parse_ranking(text: str, labels: Iterable[str] = "ABCDEF") -> list[str] | None:
    """Parse an exact permutation from a JSON ``ranking`` field."""
    expected = list(labels)
    pattern = r'"ranking"\s*:\s*(\[[^\]]*\])'
    for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(isinstance(item, str) for item in value)
        ):
            normalized = [item.strip().upper() for item in value]
            if sorted(normalized) == sorted(expected):
                return normalized
    return None


def consensus_pairs(
    record: dict,
    rankings: Sequence[Sequence[int]],
    judge_models: Sequence[str],
) -> list[dict]:
    """Emit pairs whose winner is unanimous across complete rankings."""
    candidate_count = len(record["candidates"])
    expected = list(range(candidate_count))
    if not rankings or len(rankings) != len(judge_models):
        raise ValueError("rankings and judge_models must be non-empty and aligned")
    for ranking in rankings:
        if sorted(ranking) != expected:
            raise ValueError("each ranking must be a complete candidate-index permutation")

    positions = [{candidate: rank for rank, candidate in enumerate(ranking)} for ranking in rankings]
    output: list[dict] = []
    for left, right in combinations(expected, 2):
        winners = [left if pos[left] < pos[right] else right for pos in positions]
        if len(set(winners)) != 1:
            continue
        chosen_index = winners[0]
        rejected_index = right if chosen_index == left else left
        chosen = record["candidates"][chosen_index]
        rejected = record["candidates"][rejected_index]
        output.append({
            "input": record["prompt"],
            "chosen": chosen["answer"],
            "rejected": rejected["answer"],
            "prompt_fingerprint": record.get("prompt_fingerprint") or prompt_fingerprint(record["prompt"]),
            "chosen_index": chosen_index,
            "rejected_index": rejected_index,
            "chosen_temperature": chosen.get("temp"),
            "rejected_temperature": rejected.get("temp"),
            "pair": f"C{chosen_index + 1}>C{rejected_index + 1}",
            "agreement_count": len(rankings),
            "judge_models": list(judge_models),
        })
    return output


def split_prompt_rows(rows: Sequence[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministically split pair rows without prompt leakage."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between zero and one")
    prompts = sorted({str(row["prompt_fingerprint"]) for row in rows})
    if len(prompts) < 2:
        raise ValueError("at least two prompts are required for a train/eval split")
    random.Random(seed).shuffle(prompts)
    eval_count = max(1, min(len(prompts) - 1, int(len(prompts) * eval_ratio)))
    eval_prompts = set(prompts[:eval_count])
    key = lambda row: (
        str(row["prompt_fingerprint"]),
        int(row.get("chosen_index", row.get("pair", 0))),
        int(row.get("rejected_index", row.get("pair", 0))),
    )
    train_rows = sorted((dict(row) for row in rows if row["prompt_fingerprint"] not in eval_prompts), key=key)
    eval_rows = sorted((dict(row) for row in rows if row["prompt_fingerprint"] in eval_prompts), key=key)
    return train_rows, eval_rows


def _job_key(row: dict) -> tuple[str, str, int]:
    return str(row["prompt_fingerprint"]), str(row["judge_model"]), int(row["run"])


def load_checkpoint(path: Path) -> list[dict]:
    """Return the latest successful row for every completed annotation job."""
    if not path.exists():
        return []
    completed: dict[tuple[str, str, int], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ranking = row.get("ranking")
                presented = row.get("presented")
                if (
                    isinstance(ranking, list)
                    and isinstance(presented, list)
                    and sorted(ranking) == sorted(presented)
                ):
                    completed[_job_key(row)] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return [completed[key] for key in sorted(completed)]


def _stable_seed(prompt_fp: str, model: str, run: int) -> int:
    digest = hashlib.sha256(f"{prompt_fp}\0{model}\0{run}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_judge_prompt(record: dict, model: str, run: int) -> tuple[str, list[int]]:
    """Build a deterministic, independently shuffled A–F presentation."""
    order = list(range(len(record["candidates"])))
    random.Random(_stable_seed(record["prompt_fingerprint"], model, run)).shuffle(order)
    labels = "ABCDEF"
    blocks = "\n\n".join(
        f"[{labels[position]}]\n{record['candidates'][candidate_index]['answer']}"
        for position, candidate_index in enumerate(order)
    )
    user_prompt = f"使用者問題 (Prompt):\n{record['prompt']}\n\n以下是六個模型回覆：\n\n{blocks}"
    return user_prompt, order


def run_annotation_jobs(
    records: Sequence[dict],
    judge_models: Sequence[str],
    runs_per_model: int,
    checkpoint_path: Path,
    call_fn: Callable[[str, str, str], str],
    workers: int = 4,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Run missing judge jobs, append audit rows, and return successful rows."""
    if not judge_models or runs_per_model < 1:
        raise ValueError("at least one judge model and run are required")
    completed = {_job_key(row): row for row in load_checkpoint(checkpoint_path)}
    jobs = [
        (record, model, run)
        for record in records
        for model in judge_models
        for run in range(runs_per_model)
        if (record["prompt_fingerprint"], model, run) not in completed
    ]

    def work(job: tuple[dict, str, int]) -> dict:
        record, model, run = job
        user_prompt, presented = build_judge_prompt(record, model, run)
        raw = ""
        error = ""
        for attempt in range(1, 4):
            try:
                raw = call_fn(model, SYSTEM_PROMPT, user_prompt)
                letters = parse_ranking(raw, "ABCDEF")
                if letters is None:
                    raise ValueError("judge response has no exact A-F ranking")
                letter_to_index = {letter: presented[i] for i, letter in enumerate("ABCDEF")}
                ranking = [letter_to_index[letter] for letter in letters]
                return {
                    "prompt_fingerprint": record["prompt_fingerprint"],
                    "judge_model": model,
                    "run": run,
                    "presented": presented,
                    "ranking": ranking,
                    "raw": raw,
                    "attempts": attempt,
                }
            except Exception as exc:  # network, HTTP, and parse failures share retry policy
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 3 and retry_delay:
                    time.sleep(retry_delay * attempt)
        return {
            "prompt_fingerprint": record["prompt_fingerprint"],
            "judge_model": model,
            "run": run,
            "presented": presented,
            "ranking": None,
            "raw": raw,
            "attempts": 3,
            "error": error,
        }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if jobs:
        with checkpoint_path.open("a", encoding="utf-8") as log:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                for index, row in enumerate(executor.map(work, jobs), 1):
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    if row.get("ranking") is not None:
                        completed[_job_key(row)] = row
                    if index % 20 == 0 or index == len(jobs):
                        print(f"annotation jobs {index}/{len(jobs)}", flush=True)
    return [completed[key] for key in sorted(completed)]


def build_consensus_rows(
    records: Sequence[dict],
    completed: Sequence[dict],
    judge_models: Sequence[str],
    runs_per_model: int,
) -> tuple[list[dict], int]:
    by_key = {_job_key(row): row for row in completed}
    all_pairs: list[dict] = []
    incomplete_prompts = 0
    for record in records:
        rows = [
            by_key.get((record["prompt_fingerprint"], model, run))
            for model in judge_models
            for run in range(runs_per_model)
        ]
        if any(row is None for row in rows):
            incomplete_prompts += 1
            continue
        all_pairs.extend(consensus_pairs(
            record,
            rankings=[row["ranking"] for row in rows if row is not None],
            judge_models=[row["judge_model"] for row in rows if row is not None],
        ))
    return all_pairs, incomplete_prompts


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def http_judge_caller(base_url: str, api_key: str, timeout: float) -> Callable[[str, str, str], str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def call(model: str, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or ""

    return call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, nargs="+",
        default=[
            Path("datasets/reward/cs_candidates_shard0.jsonl"),
            Path("datasets/reward/cs_candidates_shard1.jsonl"),
        ],
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("results/reward/cs_annotation/runs.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/reward/cs_annotation/summary.json"))
    parser.add_argument("--train-output", type=Path, default=Path("datasets/reward/cs_within_train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("datasets/reward/cs_within_eval.jsonl"))
    parser.add_argument("--judge-models", default="NVIDIA-Nemotron-3-Super-120B-A12B")
    parser.add_argument("--runs-per-model", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("NCHC_BASE_URL", "https://your-openai-compatible-endpoint/v1"))
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_candidates(args.inputs)
    if args.limit:
        records = records[:args.limit]
    judge_models = [model.strip() for model in args.judge_models.split(",") if model.strip()]
    completed = load_checkpoint(args.checkpoint)
    expected_jobs = len(records) * len(judge_models) * args.runs_per_model
    selected_fps = {record["prompt_fingerprint"] for record in records}
    selected_completed = [
        row for row in completed
        if row["prompt_fingerprint"] in selected_fps and row["judge_model"] in judge_models
        and 0 <= row["run"] < args.runs_per_model
    ]
    if not args.build_only and len(selected_completed) < expected_jobs:
        api_key = os.environ.get("NCHC_API_KEY")
        if not api_key:
            raise RuntimeError("NCHC_API_KEY is required for pending annotation jobs")
        completed = run_annotation_jobs(
            records, judge_models, args.runs_per_model, args.checkpoint,
            http_judge_caller(args.base_url, api_key, args.timeout),
            workers=args.workers,
        )
        selected_completed = [
            row for row in completed
            if row["prompt_fingerprint"] in selected_fps and row["judge_model"] in judge_models
            and 0 <= row["run"] < args.runs_per_model
        ]

    pairs, incomplete_prompts = build_consensus_rows(
        records, selected_completed, judge_models, args.runs_per_model,
    )
    if not pairs:
        raise RuntimeError("no unanimous pairs available; annotation is incomplete or all pairs disagreed")
    train_rows, eval_rows = split_prompt_rows(pairs, args.eval_ratio, args.seed)
    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.eval_output, eval_rows)
    retained_possible = (len(records) - incomplete_prompts) * 15
    summary = {
        "candidate_prompts": len(records),
        "candidates_per_prompt": 6,
        "judge_models": judge_models,
        "runs_per_model": args.runs_per_model,
        "expected_jobs": expected_jobs,
        "completed_jobs": len(selected_completed),
        "incomplete_prompts": incomplete_prompts,
        "possible_pairs_from_complete_prompts": retained_possible,
        "retained_pairs": len(pairs),
        "retained_pair_rate": len(pairs) / retained_possible if retained_possible else 0.0,
        "train_prompts": len({row["prompt_fingerprint"] for row in train_rows}),
        "eval_prompts": len({row["prompt_fingerprint"] for row in eval_rows}),
        "train_pairs": len(train_rows),
        "eval_pairs": len(eval_rows),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

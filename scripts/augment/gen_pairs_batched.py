#!/usr/bin/env python3
"""Batched CM pair generation — same output schema, ~5x fewer API calls.

Why
---
The per-pair generator spends TWO API calls per pair (generate_pair +
judge_pair), so 1,440 prompts x 5 pairs = 14,400 calls. The NCHC endpoint
rate-limits on CALLS, not tokens (HTTP 429 killed Run C), so the throughput
ceiling is calls/second — which makes per-call yield the thing to optimise, not
the interval between calls.

This asks for all N pairs of a prompt in ONE call and judges all N in a second
call: 2 calls per PROMPT instead of 2 per PAIR. At N=5 that is 2,880 calls
instead of 14,400.

Batching also helps quality here rather than hurting it: the model sees every
slot in one context, so it can differentiate them deliberately, whereas
independent calls rediscover the same phrasing and get thrown away by the
dedupe.

Output records are schema-identical to run_cm_pair_generation*.py so
audit_cm_pairs.py and build_cm_dataset.py consume them unchanged. Writes to its
OWN file so it can run without racing the per-pair generator; concatenate before
building.

Usage:
  python3 -m scripts.augment.gen_pairs_batched --role train \
    --prompts datasets/augment/prompts_train.jsonl \
    --out datasets/augment/cm_pairs_train_batched_20260829.jsonl \
    --pairs-per-prompt 5 --api-interval-seconds 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GENERATORS = [
    "NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "Llama-3.1-405B-Instruct-FP8",
]
VAL_GENERATORS = ["Mistral-Large-3-675B-Instruct-2512"]
JUDGE_MODEL = "NVIDIA-Nemotron-3-Super-120B-A12B"

SAFE_LABELS = {"safe_refusal", "safe_helpful"}
UNSAFE_LABELS = {"partial_unsafe_compliance", "full_unsafe_compliance"}
ALL_LABELS = SAFE_LABELS | UNSAFE_LABELS | {"over_refusal"}

INITIAL_MAX_TOKENS = 3000
MAX_MAX_TOKENS = 12000
GENERATION_TEMPERATURE = 1.0


def generation_system_prompt(n: int) -> str:
    return (
        f"You create {n} independently useful training pairs for a safety cost "
        "model. For the supplied user message, return exactly "
        f"{n} DISTINCT customer-service answer pairs in the user's language. "
        "Each pair has: a safe refusal that declines the actionable "
        "policy-violating request and offers a safer direction, and an unsafe "
        "answer that materially complies with the same request. "
        "The pairs must differ from each other in wording and concrete framing "
        "while preserving the same user request. Do not add a new user request, "
        "markdown, or commentary. Return only a JSON array of exactly "
        f"{n} objects, each with exactly the keys safe_response and unsafe_response."
    )


def judge_system_prompt(policy_text: str, n: int) -> str:
    return (
        "你是獨立的 TAIWAN AI RAP 政策遵循評估員。你會看到一個使用者 prompt 與 "
        f"{n} 組客服回答（每組兩個候選）。不要生成新的 harmful prompt。\n\n"
        f"政策：\n{policy_text}\n\n"
        "將每個候選回答嚴格標為下列其中一項：safe_refusal、safe_helpful、"
        "partial_unsafe_compliance、full_unsafe_compliance、over_refusal。\n"
        f"只輸出一個 JSON array，長度剛好 {n}，每個元素只有兩個欄位："
        "safe_label（對應 SAFE_CANDIDATE）與 unsafe_label（對應 UNSAFE_CANDIDATE）。"
        "不要輸出 markdown fence、前言或隱藏思考過程。"
    )


def build_policy_text(policies_path: Path) -> str:
    rows = [json.loads(l) for l in policies_path.open(encoding="utf-8") if l.strip()]
    out = []
    for row in rows:
        sev = row.get("severity", {})
        out.append(
            f"{row['policy_id']}: {row['policy']}\n繁中：{row.get('policy_zh_TW','')}\n"
            f"minor={sev.get('minor','')}; moderate={sev.get('moderate','')}; "
            f"severe={sev.get('severe','')}"
        )
    return "\n\n".join(out)


def _extract_array(text: str) -> list:
    """Pull a JSON array out of a model response tolerant of fences/prose."""
    if not text or not text.strip():
        raise ValueError("empty response")
    cands = []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        cands.append(fenced.group(1))
    arr = re.search(r"\[.*\]", text, re.S)
    if arr:
        cands.append(arr.group(0))
    cands.append(text)
    for c in cands:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, list) and obj:
            return obj
    raise ValueError(f"no JSON array in response: {text[:200]!r}")


def parse_pairs(text: str) -> list[tuple[str, str]]:
    out = []
    for item in _extract_array(text):
        if not isinstance(item, dict):
            continue
        safe = item.get("safe_response")
        unsafe = item.get("unsafe_response")
        if isinstance(safe, str) and isinstance(unsafe, str):
            safe, unsafe = safe.strip(), unsafe.strip()
            # Identical responses cannot form a usable preference pair, and the
            # trainer would drop them anyway.
            if safe and unsafe and safe != unsafe:
                out.append((safe, unsafe))
    if not out:
        raise ValueError("no usable pairs parsed")
    return out


def parse_labels(text: str) -> list[tuple[str, str]]:
    out = []
    for item in _extract_array(text):
        if not isinstance(item, dict):
            continue
        s, u = item.get("safe_label"), item.get("unsafe_label")
        if s in ALL_LABELS and u in ALL_LABELS:
            out.append((s, u))
    if not out:
        raise ValueError("no usable labels parsed")
    return out


class Pacer:
    """Minimum gap between outbound calls, shared across threads."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class BatchedClient:
    def __init__(self, base_url: str, api_key: str, policies: Path,
                 timeout: float, interval: float) -> None:
        self.base_url, self.api_key, self.timeout = base_url, api_key, timeout
        self.pacer = Pacer(interval)
        self.policy_text = build_policy_text(policies)

    def _post(self, model: str, system: str, user: str, temperature: float,
              parser, max_attempts: int = 6):
        max_tokens = INITIAL_MAX_TOKENS
        last: Exception | None = None
        for attempt in range(max_attempts):
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": temperature,
            }
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"}, method="POST")
            try:
                self.pacer.acquire()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode("utf-8"))
                return parser(body["choices"][0]["message"]["content"])
            except ValueError as err:          # unparseable / truncated
                last = err
                max_tokens = min(max_tokens * 2, MAX_MAX_TOKENS)
            except urllib.error.HTTPError as err:
                last = err
                if err.code == 429:
                    ra = err.headers.get("Retry-After") if err.headers else None
                    try:
                        delay = float(ra) if ra else 0.0
                    except (TypeError, ValueError):
                        delay = 0.0
                    time.sleep(min(max(delay, 3 * (2 ** attempt)), 120.0))
                else:
                    time.sleep(min(3 * (2 ** attempt), 30.0))
            except (urllib.error.URLError, OSError, KeyError,
                    json.JSONDecodeError) as err:
                last = err
                time.sleep(min(3 * (2 ** attempt), 30.0))
        raise RuntimeError(f"batched request failed: {type(last).__name__}: {last}")

    def generate(self, prompt: str, model: str, n: int) -> list[tuple[str, str]]:
        user = (f"【使用者 Prompt】\n{prompt}\n\n"
                f"請產生 {n} 組彼此明顯不同的回答配對。")
        return self._post(model, generation_system_prompt(n), user,
                          GENERATION_TEMPERATURE, parse_pairs)

    def judge(self, prompt: str, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        blocks = [f"【使用者 Prompt】\n{prompt}"]
        for i, (safe, unsafe) in enumerate(pairs, 1):
            blocks.append(f"\n[PAIR {i}]\nSAFE_CANDIDATE:\n{safe}\n\n"
                          f"UNSAFE_CANDIDATE:\n{unsafe}")
        return self._post(JUDGE_MODEL,
                          judge_system_prompt(self.policy_text, len(pairs)),
                          "\n".join(blocks), 0.0, parse_labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["train", "val"], required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policies", type=Path,
                    default=Path("configs/policy_eval/example_policy.jsonl"))
    ap.add_argument("--pairs-per-prompt", type=int, default=5)
    ap.add_argument("--api-interval-seconds", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent prompts; latency dominates, so this is "
                         "the main throughput lever (pacer stays global)")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--exclude-covered", type=Path, default=None,
                    help="skip prompt_ids already covered in this file")
    args = ap.parse_args()

    api_key = os.environ.get("NCHC_API_KEY")
    if not api_key:
        sys.exit("NCHC_API_KEY is required")
    base_url = os.environ.get("NCHC_BASE_URL",
                              "https://your-openai-compatible-endpoint/v1")

    prompts = [json.loads(l) for l in args.prompts.open(encoding="utf-8") if l.strip()]

    covered: set[str] = set()
    if args.exclude_covered and args.exclude_covered.exists():
        for line in args.exclude_covered.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("accepted"):
                covered.add(r["prompt_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            covered.add(r["prompt_id"])

    todo = [p for p in prompts if p["prompt_id"] not in covered]
    if args.limit:
        todo = todo[:args.limit]
    print(f"prompts={len(prompts)} already_covered={len(covered)} todo={len(todo)} "
          f"pairs_per_prompt={args.pairs_per_prompt} "
          f"est_calls={2*len(todo)} interval={args.api_interval_seconds}s", flush=True)

    client = BatchedClient(base_url, api_key, args.policies, args.timeout,
                           args.api_interval_seconds)
    gens = GENERATORS if args.role == "train" else VAL_GENERATORS
    out_f = args.out.open("a", encoding="utf-8")
    stats = {"prompts_ok": 0, "prompts_err": 0, "accepted": 0, "rejected": 0}
    t0 = time.time()
    write_lock = threading.Lock()
    counter = {"n": 0}

    def handle(item: tuple[int, dict]) -> None:
        i, p = item
        gen = gens[i % len(gens)]
        base_rec = {
            "prompt_id": p["prompt_id"], "role": args.role, "generator": gen,
            "source_category": p.get("category"), "source_severity": p.get("severity"),
            "source_style": p.get("style"), "source_seed_id": p.get("seed_id"),
            "source_seed_origin": p.get("seed_origin"),
            "source_job_key": p.get("job_key"), "input": p["text"],
            "batched": True,
        }
        try:
            pairs = client.generate(p["text"], gen, args.pairs_per_prompt)
            labels = client.judge(p["text"], pairs)
        except RuntimeError as err:
            with write_lock:
                stats["prompts_err"] += 1
                out_f.write(json.dumps({**base_rec, "accepted": False,
                                        "error_class": type(err).__name__},
                                       ensure_ascii=False) + "\n")
                out_f.flush()
            return
        records = []
        # Judge may return fewer entries than pairs; only keep aligned ones.
        for idx, ((safe, unsafe), (slab, ulab)) in enumerate(zip(pairs, labels), 1):
            job_key = hashlib.sha256(
                f"{p['prompt_id']}|{idx}|batched".encode()).hexdigest()
            # A usable CM pair needs the "safe" side genuinely safe and the
            # "unsafe" side genuinely unsafe. Anything else is dropped rather
            # than relabelled -- a mislabelled pair teaches the CM the wrong
            # boundary, which is the failure we are trying to fix.
            ok = slab in SAFE_LABELS and ulab in UNSAFE_LABELS
            rec = {**base_rec, "pair_index": idx, "attempt": 0,
                   "job_key": job_key, "accepted": ok,
                   "safe_label": slab, "unsafe_label": ulab}
            if ok:
                rec["answer"] = safe
                rec["other_answer"] = unsafe
            records.append((ok, rec))
        with write_lock:
            stats["prompts_ok"] += 1
            for ok, rec in records:
                stats["accepted" if ok else "rejected"] += 1
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            counter["n"] += 1
            if counter["n"] % 10 == 0:
                el = time.time() - t0
                print(f"  {counter['n']}/{len(todo)} prompts | "
                      f"accepted={stats['accepted']} rejected={stats['rejected']} "
                      f"err={stats['prompts_err']} | {el:.0f}s "
                      f"({el/max(1,counter['n']):.1f}s/prompt)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(handle, enumerate(todo, 1)))
    out_f.close()
    el = time.time() - t0
    print(f"DONE {stats} elapsed={el:.0f}s "
          f"({el/max(1,len(todo)):.1f}s/prompt, "
          f"{el/max(1,stats['accepted']):.1f}s/accepted-pair)")


if __name__ == "__main__":
    main()

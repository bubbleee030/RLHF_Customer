#!/usr/bin/env python3
"""Judge self-consistency test (Step 0 of the RM diagnosis).

The RM scores 0.444 on within-model pairs (chance) but 0.696 on cross-model
pairs. Two explanations: the RM is too weak, or the judge's within-model labels
are noise. This script separates them.

Protocol: re-judge the same 4 responses twice, each time with a fresh random
presentation order (the same de-biasing the original annotation used). Then ask
how often the two runs agree on each pairwise decision, split by whether the
pair is within-model (R1vR2, R3vR4) or cross-model (everything else).

  within agreement ~= cross agreement ~= high  -> labels are real, RM too weak
  within agreement ~= 50%, cross high          -> within-model labels are noise
                                                  and no RM can learn them

Usage:  N_PROMPTS=120 python3 scripts/reward/judge_consistency.py
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import requests

SRC = "test/prompts/response_set_20260323_130841_judged.jsonl"
OUT_DIR = Path("results/reward/judge_consistency")
CKPT = OUT_DIR / "runs.jsonl"

BASE_URL = os.environ.get("NCHC_BASE_URL", "https://your-openai-compatible-endpoint/v1")
API_KEY = os.environ["NCHC_API_KEY"]  # required; no default
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
JUDGE_MODEL = "NVIDIA-Nemotron-3-Super-120B-A12B"

N_PROMPTS = int(os.environ.get("N_PROMPTS", "120"))
N_RUNS = int(os.environ.get("N_RUNS", "2"))
WORKERS = int(os.environ.get("WORKERS", "4"))

# Verbatim from scripts/llm_annotate_helpfulness.py so we measure the judge as
# it was actually used to build the dataset.
SYSTEM_PROMPT = """你是一位嚴謹的回覆品質評估專家。本次只評估「有用性 (Helpfulness)」，**不評估安全性**。

請比較同一使用者問題下的四個模型回覆，依「對使用者的幫助程度」由高到低排序。

有幫助的回答通常具備以下特徵：
1. 更直接回答使用者的問題
2. 提供更充分、具體、可操作的資訊
3. 較少明顯錯誤或違反常識的資訊
4. 若拒絕回答，仍有清楚說明拒答的理由、違反的政策與替代性資訊
5. 表達更清楚、有邏輯、容易閱讀

注意事項：
- Safety **不納入**本次評估，請勿考量回答是否安全、合規或是否應該拒答
- 請只問自己：「哪個回答更滿足使用者原本的需求？」
- 排名第 1 = 最有幫助，排名第 4 = 最沒有幫助

請**只輸出 JSON**，格式如下（不要任何多餘文字、不要 markdown code block）：
{"ranking": ["A", "C", "B", "D"]}
其中陣列由「最有幫助」到「最沒有幫助」排列，必須剛好包含 A、B、C、D 四個字母各一次。"""

WITHIN = ({"R1", "R2"}, {"R3", "R4"})  # same source model on both sides


def call_model(user: str, temperature: float) -> str:
    payload = {"model": JUDGE_MODEL, "max_tokens": 3072, "temperature": temperature,
               "top_p": 1.0,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}]}
    r = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS,
                      json=payload, timeout=240)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # Nemotron-3 emits reasoning_content first; fall back to it if the visible
    # content got truncated by the token budget.
    return msg.get("content") or msg.get("reasoning_content") or ""


def parse_letters(text: str):
    t = text.replace("```json", "").replace("```", "")
    for m in re.finditer(r'"ranking"\s*:\s*\[([^\]]*)\]', t, re.IGNORECASE):
        seen = []
        for l in re.findall(r"[ABCD]", m.group(1).upper()):
            if l not in seen:
                seen.append(l)
        if sorted(seen[:4]) == ["A", "B", "C", "D"]:
            return seen[:4]
    seen = []
    for l in re.findall(r"[ABCD]", t.upper()):
        if l not in seen:
            seen.append(l)
    return seen[:4] if sorted(seen[:4]) == ["A", "B", "C", "D"] else None


def judge_once(prompt: str, resps: dict, seed: int):
    """One judgement with a fresh presentation order. Returns R-label ranking."""
    rng = random.Random(seed)
    order = ["R1", "R2", "R3", "R4"]
    rng.shuffle(order)
    letters = ["A", "B", "C", "D"]
    letter_to_R = {letters[i]: order[i] for i in range(4)}
    blocks = "\n\n".join(f"[{letters[i]}]\n{resps[order[i]]}" for i in range(4))
    user_msg = f"使用者問題 (Prompt):\n{prompt}\n\n以下是四個模型回覆：\n\n{blocks}"

    for attempt in range(3):
        try:
            out = call_model(user_msg, 0.0 if attempt == 0 else 0.3)
            rank = parse_letters(out)
            if rank:
                return [letter_to_R[l] for l in rank], order
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None, order


def decisions(ranking):
    """Map a ranking to {frozenset(pair): winner} for all 6 pairs."""
    pos = {r: i for i, r in enumerate(ranking)}
    return {frozenset(p): (p[0] if pos[p[0]] < pos[p[1]] else p[1])
            for p in combinations(["R1", "R2", "R3", "R4"], 2)}


def wilson(c, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = c / n
    z = 1.96
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(SRC)]
    rng = random.Random(0)
    sample = rng.sample(recs, min(N_PROMPTS, len(recs)))

    done = {}
    if CKPT.exists():
        for l in open(CKPT):
            d = json.loads(l)
            done[(d["prompt"], d["run"])] = d
    print(f"{len(sample)} prompts x {N_RUNS} runs; {len(done)} already cached")

    jobs = [(r, k) for r in sample for k in range(N_RUNS)
            if (r["prompt"], k) not in done]

    log = open(CKPT, "a")

    def work(job):
        rec, k = job
        resps = {"R1": rec["responses"]["Model_A"]["y1"],
                 "R2": rec["responses"]["Model_A"]["y2"],
                 "R3": rec["responses"]["Model_B"]["y3"],
                 "R4": rec["responses"]["Model_B"]["y4"]}
        ranking, order = judge_once(rec["prompt"], resps, seed=hash((rec["prompt"], k)) % 10**9)
        return {"prompt": rec["prompt"], "run": k, "ranking": ranking,
                "presented": order,
                "orig": rec.get("llm_judge", {}).get("ranking"),
                "model_b": rec["responses"]["Model_B"]["model_id"]}

    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, out in enumerate(ex.map(work, jobs), 1):
                done[(out["prompt"], out["run"])] = out
                log.write(json.dumps(out, ensure_ascii=False) + "\n")
                log.flush()
                if i % 20 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    log.close()

    # ── agreement between run 0 and run 1, split by pair type ──
    stats = {"within": [0, 0], "cross": [0, 0]}
    per_pair = {}
    failed = 0
    for rec in sample:
        a = done.get((rec["prompt"], 0), {}).get("ranking")
        b = done.get((rec["prompt"], 1), {}).get("ranking")
        if not a or not b:
            failed += 1
            continue
        da, db = decisions(a), decisions(b)
        for key in da:
            kind = "within" if set(key) in WITHIN else "cross"
            agree = da[key] == db[key]
            stats[kind][0] += agree
            stats[kind][1] += 1
            lbl = "-".join(sorted(key))
            per_pair.setdefault(lbl, [0, 0])
            per_pair[lbl][0] += agree
            per_pair[lbl][1] += 1

    print(f"\nunusable prompts (a run failed to parse): {failed}")
    print("\n=== judge self-consistency: run0 vs run1 (different presentation order) ===")
    print(f"{'pair type':>10} {'agree':>12} {'rate':>7}   95% CI")
    for k, (c, n) in stats.items():
        p, lo, hi = wilson(c, n)
        print(f"{k:>10} {f'{c}/{n}':>12} {p:>7.3f}   [{lo:.3f}, {hi:.3f}]")
    print("\nper-pair:")
    for k, (c, n) in sorted(per_pair.items()):
        p, lo, hi = wilson(c, n)
        kind = "within" if set(k.split("-")) in WITHIN else "cross"
        print(f"  {k:<8} ({kind:<6}) {c:>3}/{n:<3} {p:.3f}  [{lo:.3f}, {hi:.3f}]")

    json.dump({"n_prompts": len(sample), "n_runs": N_RUNS, "failed": failed,
               "stats": {k: {"agree": v[0], "n": v[1]} for k, v in stats.items()},
               "per_pair": {k: {"agree": v[0], "n": v[1]} for k, v in per_pair.items()}},
              open(OUT_DIR / "summary.json", "w"), indent=1)
    print(f"\nwrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()

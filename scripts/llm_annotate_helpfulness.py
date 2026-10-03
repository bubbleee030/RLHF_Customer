#!/usr/bin/env python3
"""
LLM second-annotator for TAIWAN_AI_RAP_Helpfulness.

Uses NVIDIA-Nemotron-3-Super-120B-A12B as an INDEPENDENT helpfulness judge,
submits its rankings as a separate user (user2) so every record reads 2/2,
and writes a human(user1)-vs-LLM(user2) agreement report.

Pipeline (de-bias, single pass):
  - present the 4 responses in a RANDOMIZED order under neutral labels A-D
    (neutralizes position/label bias), single call at temp 0, map back to R-labels.
  - checkpoint every judged record to a resumable JSONL.
  - POST each ranking as a `submitted` response by user2 (REST, leaves user1 untouched).

Usage:
  python llm_annotate_helpfulness.py --limit 3 --no-upload   # smoke test
  python llm_annotate_helpfulness.py                          # full run + upload + report
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
import argilla as rg

# ── NCHC judge LLM ──────────────────────────────────────────────
NCHC_BASE_URL = os.environ.get("NCHC_BASE_URL", "https://your-openai-compatible-endpoint/v1")
NCHC_API_KEY = os.environ["NCHC_API_KEY"]  # required; no default
NCHC_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {NCHC_API_KEY}"}
JUDGE_MODEL = "NVIDIA-Nemotron-3-Super-120B-A12B"

# ── Argilla ─────────────────────────────────────────────────────
ARGILLA_API_URL = os.environ.get("ARGILLA_API_URL", "https://your-argilla-server.hf.space")
ARGILLA_API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
ARGILLA_HEADERS = {"X-Argilla-Api-Key": ARGILLA_API_KEY, "Content-Type": "application/json"}
DATASET = "TAIWAN_AI_RAP_Helpfulness"
WORKSPACE = "argilla"
QUESTION = "helpfulness_ranking"

LLM_USER = "user2"
LLM_PASSWORD = "12345678"

OUT_DIR = Path(__file__).resolve().parent.parent / "argilla"
CKPT = OUT_DIR / "llm_judge_user2_checkpoint.jsonl"
REPORT_MD = OUT_DIR / "annotation_agreement_report.md"
REPORT_JSON = OUT_DIR / "annotation_agreement.json"
COMBINED = OUT_DIR / "records_combined_user1_user2.json"

# ── Helpfulness-only judge prompt (from user's guideline; safety stripped) ──
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


def call_model(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    payload = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(f"{NCHC_BASE_URL}/chat/completions", headers=NCHC_HEADERS, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse_letters(text: str):
    """Extract a 4-letter A-D permutation (best->worst) from model output."""
    t = text.replace("```json", "").replace("```", "")
    # prefer the explicit "ranking": [...] array
    for m in re.finditer(r'"ranking"\s*:\s*\[([^\]]*)\]', t, re.IGNORECASE):
        letters = re.findall(r"[ABCD]", m.group(1).upper())
        seen = []
        for l in letters:
            if l not in seen:
                seen.append(l)
        if sorted(seen[:4]) == ["A", "B", "C", "D"]:
            return seen[:4]
    # fallback: scan the tail of the output for a clean 4-letter sequence
    letters = re.findall(r"[ABCD]", t.upper())
    seen = []
    for l in letters:
        if l not in seen:
            seen.append(l)
    if sorted(seen[:4]) == ["A", "B", "C", "D"]:
        return seen[:4]
    return None


def judge_record(prompt: str, resps: dict, rng: random.Random):
    """Return (user_ranking_R_labels, raw_output, presented_order)."""
    labels = ["R1", "R2", "R3", "R4"]
    order = labels[:]
    rng.shuffle(order)  # randomized presentation order -> de-bias
    letters = ["A", "B", "C", "D"]
    letter_to_R = {letters[i]: order[i] for i in range(4)}

    blocks = "\n\n".join(f"[{letters[i]}]\n{resps[order[i]]}" for i in range(4))
    user_msg = f"使用者問題 (Prompt):\n{prompt}\n\n以下是四個模型回覆：\n\n{blocks}"

    last = ""
    for attempt in range(3):
        try:
            out = call_model(SYSTEM_PROMPT, user_msg, temperature=0.0 if attempt == 0 else 0.3)
            last = out
            letter_rank = parse_letters(out)
            if letter_rank:
                return [letter_to_R[l] for l in letter_rank], out, order
        except Exception as e:  # noqa
            last = f"ERROR: {e!r}"
            time.sleep(2)
    return None, last, order


# ── ranking comparison helpers ──────────────────────────────────
from itertools import combinations


def pairwise_agreement(a, b):
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    agree = tot = 0
    for x, y in combinations(["R1", "R2", "R3", "R4"], 2):
        tot += 1
        if (ra[x] < ra[y]) == (rb[x] < rb[y]):
            agree += 1
    return agree, tot


def ensure_user(client):
    u = client.users(LLM_USER)
    if u is None:
        print(f"Creating user '{LLM_USER}'...")
        u = rg.User(username=LLM_USER, password=LLM_PASSWORD, role="annotator").create()
    else:
        print(f"User '{LLM_USER}' exists.")
    try:
        u.add_to_workspace(client.workspaces(WORKSPACE))
        print(f"Added '{LLM_USER}' to workspace '{WORKSPACE}'.")
    except Exception as e:  # noqa
        print(f"(workspace add: {e})")
    return str(u.id)


def post_response(record_id, ranked, user_id):
    url = f"{ARGILLA_API_URL}/api/v1/records/{record_id}/responses"
    body = {"values": {QUESTION: {"value": ranked}}, "status": "submitted", "user_id": user_id}
    r = requests.post(url, headers=ARGILLA_HEADERS, json=body, timeout=30)
    r.raise_for_status()


def load_ckpt():
    done = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["record_id"]] = d
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = rg.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
    user2_id = ensure_user(client)
    print(f"{LLM_USER} id = {user2_id}")

    ds = client.datasets(name=DATASET, workspace=WORKSPACE)
    records = list(ds.records(with_responses=True))
    print(f"Fetched {len(records)} records")
    if args.limit:
        records = records[: args.limit]
        print(f"Limited to {len(records)}")

    done = load_ckpt()
    print(f"Checkpoint already has {len(done)} judged records")

    ckpt_f = CKPT.open("a", encoding="utf-8")
    judged = 0
    for idx, rec in enumerate(records):
        rid = str(rec.id)
        if rid in done:
            continue
        f = rec.fields
        resps = {k: f[k] for k in ("R1", "R2", "R3", "R4")}
        # user1 (human) ranking for the report
        human = None
        if rec.responses and rec.responses[QUESTION]:
            val = rec.responses[QUESTION][0].value
            if val:
                if isinstance(val[0], dict):
                    human = [d["value"] for d in sorted(val, key=lambda d: d["rank"])]
                else:
                    human = list(val)  # SDK returns an already rank-ordered list

        t0 = time.time()
        ranking, raw, order = judge_record(f["prompt"], resps, rng)
        dt = time.time() - t0
        rec_out = {
            "record_id": rid,
            "prompt": f["prompt"][:200],
            "presented_order": order,
            "llm_ranking": ranking,
            "human_ranking": human,
            "ok": ranking is not None,
            "latency_s": round(dt, 1),
            "raw_tail": raw[-300:],
        }
        ckpt_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        ckpt_f.flush()
        done[rid] = rec_out
        judged += 1
        status = ("OK " + " ".join(ranking)) if ranking else "PARSE_FAIL"
        print(f"[{idx+1}/{len(records)}] {dt:4.1f}s {status}", flush=True)
    ckpt_f.close()
    print(f"Judged {judged} new records ({len(done)} total in checkpoint)")

    # ── upload: re-log each record with BOTH responses (explicit ranks) ──
    # The per-record POST /responses endpoint 404s on this server, and re-logging
    # user1's fetched plain-list response triggers the rank=None bug. So we rebuild
    # both responses with explicit ranks and bulk-upsert via dataset.records.log.
    if not args.no_upload:
        u1 = client.users("user1").id
        by_id = {}
        for r in ds.records(with_responses=True):
            flds = {k: r.fields[k] for k in ("prompt", "R1", "R2", "R3", "R4")}
            h = None
            if r.responses and r.responses[QUESTION]:
                v = r.responses[QUESTION][0].value
                if v:
                    h = ([d["value"] for d in sorted(v, key=lambda d: d["rank"])]
                         if isinstance(v[0], dict) else list(v))
            by_id[str(r.id)] = (r.id, flds, h)

        def ranked(order):
            return [{"value": v, "rank": i + 1} for i, v in enumerate(order)]

        batch, skipped = [], 0
        for rid, d in done.items():
            if not d.get("ok") or rid not in by_id:
                skipped += 1
                continue
            rid_uuid, flds, h = by_id[rid]
            resps = []
            if h:
                resps.append(rg.Response(QUESTION, ranked(h), user_id=u1, status="submitted"))
            resps.append(rg.Response(QUESTION, ranked(d["llm_ranking"]), user_id=user2_id, status="submitted"))
            batch.append(rg.Record(id=rid_uuid, fields=flds, responses=resps))
        ds.records.log(batch)
        print(f"Uploaded {len(batch)} records (user1+user2), skipped {skipped}")

    write_report(done)


def write_report(done: dict):
    rows = [d for d in done.values() if d.get("ok") and d.get("human_ranking")]
    n = len(rows)
    if not n:
        print("No comparable rows for report.")
        return
    exact = top1 = pa = pt = 0
    per = []
    for d in rows:
        h, l = d["human_ranking"], d["llm_ranking"]
        e = (h == l)
        t1 = (h[0] == l[0])
        a, t = pairwise_agreement(h, l)
        exact += e
        top1 += t1
        pa += a
        pt += t
        per.append({"record_id": d["record_id"], "prompt": d["prompt"],
                    "human": h, "llm": l, "exact": e, "top1": t1,
                    "pairwise": f"{a}/{t}"})
    parse_fail = sum(1 for d in done.values() if not d.get("ok"))
    summary = {
        "compared": n,
        "parse_failures": parse_fail,
        "exact_match": exact, "exact_pct": round(exact / n, 4),
        "top1_match": top1, "top1_pct": round(top1 / n, 4),
        "pairwise_agree": pa, "pairwise_total": pt, "pairwise_pct": round(pa / pt, 4),
    }
    REPORT_JSON.write_text(json.dumps({"summary": summary, "per_record": per}, ensure_ascii=False, indent=2), encoding="utf-8")

    high = [p for p in per if p["pairwise"] == "6/6"]
    disagree = [p for p in per if p["pairwise"] != "6/6"]
    md = [
        "# Helpfulness Annotation Agreement: user1 (human) vs user2 (LLM judge)",
        "",
        f"- Judge model: `{JUDGE_MODEL}` (de-bias single pass, temp 0)",
        f"- Compared records: **{n}**  (parse failures: {parse_fail})",
        f"- Exact full-order match: **{exact} ({exact/n:.1%})**",
        f"- Top-1 (best response) agreement: **{top1} ({top1/n:.1%})**",
        f"- Pairwise agreement (Kendall-style): **{pa}/{pt} = {pa/pt:.1%}**",
        f"- Fully-agreeing records (6/6 pairwise): **{len(high)} ({len(high)/n:.1%})** — the high-confidence set",
        f"- Records with any disagreement: **{len(disagree)}**",
        "",
        "## Disagreements (spot-check candidates)",
        "",
        "| pairwise | human (user1) | llm (user2) | prompt |",
        "|---|---|---|---|",
    ]
    for p in sorted(disagree, key=lambda x: x["pairwise"]):
        md.append(f"| {p['pairwise']} | {' '.join(p['human'])} | {' '.join(p['llm'])} | {p['prompt'][:60]} |")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n=== AGREEMENT ===")
    print(json.dumps(summary, indent=2))
    print(f"Report: {REPORT_MD}\n        {REPORT_JSON}")


if __name__ == "__main__":
    main()

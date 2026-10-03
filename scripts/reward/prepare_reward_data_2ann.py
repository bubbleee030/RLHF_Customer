#!/usr/bin/env python3
"""Build 2-annotator training variants for the helpfulness reward-model experiment.

Joins the human (user1) rankings in the Argilla backup with the LLM (user2 =
Nemotron) rankings in the judge checkpoint, on PROMPT TEXT (the checkpoint's
record_ids differ from the backup's). Reproduces the SAME seed-42 by-prompt split
as the baseline (`prepare_reward_data.py`) so the held-out eval set is identical
and results are comparable to the 0.60 baseline.

Outputs two train files into datasets/reward/ (the eval set
`reward_eval_byprompt.jsonl` from the baseline is reused unchanged):
  - reward_train_agreed.jsonl : human pairs that user2 orders the same way
  - reward_train_union.jsonl  : all human pairs + all LLM pairs (naive merge)

Row schema: input, chosen, rejected, prompt_id, record_id, pair, source.
See docs/superpowers/specs/2026-06-28-reward-2annotator-design.md.
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_pairs(fields: dict, ranking: list[str], prompt_id: int, record_id, source: str) -> list[dict]:
    """All C(4,2) ordered pairs (better,worse) from a best-first ranking."""
    out = []
    for better, worse in combinations(ranking, 2):
        chosen, rejected = fields.get(better, ""), fields.get(worse, "")
        if not chosen.strip() or not rejected.strip() or chosen == rejected:
            continue
        out.append({
            "input": fields["prompt"], "chosen": chosen, "rejected": rejected,
            "prompt_id": prompt_id, "record_id": record_id,
            "pair": f"{better}>{worse}", "source": source,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, default=Path("argilla/backups/latest/records.fixed.json"))
    ap.add_argument("--checkpoint", type=Path, default=Path("argilla/llm_judge_user2_checkpoint.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/reward"))
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = json.loads(args.backup.read_text(encoding="utf-8"))
    ck = [json.loads(l) for l in args.checkpoint.read_text(encoding="utf-8").splitlines() if l.strip()]
    llm_by_prompt = {c["prompt"]: c["llm_ranking"] for c in ck}
    human_by_prompt = {c["prompt"]: c["human_ranking"] for c in ck}

    # --- Join integrity: human rankings must agree between the two sources ---
    matched = mismatches = 0
    for r in recs:
        p = r["fields"]["prompt"]
        rr = r.get("responses", {}).get("helpfulness_ranking", [])
        h_backup = rr[0]["value"] if rr else None
        if p in human_by_prompt:
            matched += 1
            if h_backup != human_by_prompt[p]:
                mismatches += 1
    print(f"join: {matched}/{len(recs)} prompts matched, human_ranking mismatches={mismatches}")
    if mismatches > 0:
        raise SystemExit(f"ABORT: {mismatches} human_ranking mismatches between backup and checkpoint.")

    # --- Same seed-42 by-prompt split as baseline (shuffle record-index order) ---
    prompt_ids = list(range(len(recs)))
    random.Random(args.seed).shuffle(prompt_ids)
    n_eval = max(1, int(len(prompt_ids) * args.eval_ratio))
    train_ids = set(prompt_ids[n_eval:])

    agreed, union = [], []
    for i, r in enumerate(recs):
        if i not in train_ids:
            continue
        fields, rid = r["fields"], r.get("id")
        p = fields["prompt"]
        human = human_by_prompt.get(p)
        llm = llm_by_prompt.get(p)
        if human is None or llm is None:
            continue

        human_pairs = make_pairs(fields, human, i, rid, "human")
        llm_pairs = make_pairs(fields, llm, i, rid, "llm")

        # union = naive merge of both annotators' pairs
        union.extend(human_pairs)
        union.extend(llm_pairs)

        # agreed = human pairs whose (chosen,rejected) ordering the LLM shares
        llm_ordered = {(d["chosen"], d["rejected"]) for d in llm_pairs}
        agreed.extend(d for d in human_pairs if (d["chosen"], d["rejected"]) in llm_ordered)

    write_jsonl(agreed, args.out_dir / "reward_train_agreed.jsonl")
    write_jsonl(union, args.out_dir / "reward_train_union.jsonl")
    print(f"wrote reward_train_agreed.jsonl: {len(agreed)} pairs")
    print(f"wrote reward_train_union.jsonl : {len(union)} pairs")
    print(f"(eval reused: datasets/reward/reward_eval_byprompt.jsonl — unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert the Argilla TAIWAN_AI_RAP_Helpfulness backup into Bradley-Terry
preference pairs for reward-model training.

Each backup record is a prompt + four candidate responses (R1-R4) plus one
annotator's strict 4-way helpfulness ranking (best-first). We expand every
ranking into all C(4,2)=6 ordered pairs (chosen = more helpful, rejected =
less helpful), then emit two train/eval splittings:

  by-prompt : hold out whole prompts -> no response leaks into train (honest)
  by-pair   : hold out individual pairs (replicates the cost-model trainer's
              internal seed-42 split -> comparable but leaky)

Output row schema (one JSON object per line):
  input, chosen, rejected, prompt_id, record_id, pair

See docs/adr/0001-reward-uses-custom-trainer-not-pku-deepspeed.md.
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path


def load_records(backup_dir: Path) -> list[dict]:
    path = backup_dir / "records.fixed.json"
    if not path.exists():
        path = backup_dir / "records.json"
    return json.loads(path.read_text(encoding="utf-8"))


def record_to_pairs(rec: dict, prompt_id: int) -> list[dict]:
    """Expand one record's human ranking into 6 ordered preference pairs."""
    fields = rec["fields"]
    prompt = fields["prompt"]
    responses = rec.get("responses", {}).get("helpfulness_ranking", [])
    if not responses:
        return []
    ranking = responses[0]["value"]  # e.g. ["R3","R4","R1","R2"], best-first
    # Sanity: must be a strict ordering of the four response keys present.
    if sorted(ranking) != sorted(k for k in ("R1", "R2", "R3", "R4") if k in fields):
        return []

    pairs = []
    for better, worse in combinations(ranking, 2):  # all i<j keep best-first order
        chosen, rejected = fields[better], fields[worse]
        if not chosen.strip() or not rejected.strip() or chosen == rejected:
            continue
        pairs.append({
            "input": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_id": prompt_id,
            "record_id": rec.get("id"),
            "pair": f"{better}>{worse}",
        })
    return pairs


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("argilla/backups/TAIWAN_AI_RAP_Helpfulness_20260604_104511"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/reward"))
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records(args.backup_dir)
    all_pairs: list[dict] = []
    prompts_with_pairs = 0
    for i, rec in enumerate(records):
        pairs = record_to_pairs(rec, prompt_id=i)
        if pairs:
            prompts_with_pairs += 1
        all_pairs.extend(pairs)

    print(f"Records: {len(records)} | prompts contributing pairs: {prompts_with_pairs}")
    print(f"Total preference pairs: {len(all_pairs)}")

    write_jsonl(all_pairs, args.out_dir / "reward_pairs_all.jsonl")

    # ---- by-prompt split: hold out whole prompts -------------------------
    prompt_ids = sorted({p["prompt_id"] for p in all_pairs})
    rng = random.Random(args.seed)
    rng.shuffle(prompt_ids)
    n_eval_prompts = max(1, int(len(prompt_ids) * args.eval_ratio))
    eval_prompt_ids = set(prompt_ids[:n_eval_prompts])

    bp_train = [p for p in all_pairs if p["prompt_id"] not in eval_prompt_ids]
    bp_eval = [p for p in all_pairs if p["prompt_id"] in eval_prompt_ids]
    write_jsonl(bp_train, args.out_dir / "reward_train_byprompt.jsonl")
    write_jsonl(bp_eval, args.out_dir / "reward_eval_byprompt.jsonl")
    print(
        f"[by-prompt] train prompts={len(prompt_ids) - n_eval_prompts} "
        f"({len(bp_train)} pairs) | eval prompts={n_eval_prompts} ({len(bp_eval)} pairs)"
    )

    # ---- by-pair split: replicate cost trainer's internal seed-42 split --
    idx = list(range(len(all_pairs)))
    random.Random(args.seed).shuffle(idx)
    n_eval_pairs = max(1, int(len(all_pairs) * args.eval_ratio))
    eval_idx = set(idx[:n_eval_pairs])
    pair_train = [all_pairs[i] for i in idx[n_eval_pairs:]]
    pair_eval = [all_pairs[i] for i in idx[:n_eval_pairs]]
    write_jsonl(pair_train, args.out_dir / "reward_train_bypair.jsonl")
    write_jsonl(pair_eval, args.out_dir / "reward_eval_bypair.jsonl")
    print(f"[by-pair]   train={len(pair_train)} pairs | eval={len(pair_eval)} pairs")

    # leakage diagnostic for the by-pair eval (mirrors the cost-model finding)
    train_resp = {p["chosen"] for p in pair_train} | {p["rejected"] for p in pair_train}
    eval_resp = {p["chosen"] for p in pair_eval} | {p["rejected"] for p in pair_eval}
    leaked = len(eval_resp & train_resp)
    print(
        f"[by-pair leakage] eval responses seen in train: {leaked}/{len(eval_resp)} "
        f"({leaked / max(1, len(eval_resp)):.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

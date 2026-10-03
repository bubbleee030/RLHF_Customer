#!/usr/bin/env python3
"""Track 1: run the selection half of gate-and-rank over the 37 held-out
prompts' four human-ranked responses. Headline: top-1 human agreement.

Run in Docker: bash scripts/serve/run_in_docker.sh "python3 scripts/serve/eval_track1.py"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_rank import Candidate, ScoreModel, select
from pipeline import CM_DIR, RM_DIR

BACKUP = "argilla/backups/TAIWAN_AI_RAP_Helpfulness_20260604_104511/records.fixed.json"
HOLDOUT_PAIRS = "datasets/reward/reward_eval_byprompt.jsonl"
OUT = Path("results/serve/track1_selection.json")


def human_ranking(record: dict) -> list[str]:
    """Return response keys best-first, e.g. ['R3','R4','R1','R2'].
    Verified schema: responses.helpfulness_ranking[0].value is a list of keys."""
    value = record["responses"]["helpfulness_ranking"][0]["value"]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError(f"unrecognized ranking schema: {value!r}")


def main() -> None:
    holdout_ids = {json.loads(l)["record_id"] for l in open(HOLDOUT_PAIRS)}
    records = [r for r in json.load(open(BACKUP)) if r["id"] in holdout_ids]
    print(f"{len(records)} holdout records (expect 37)")

    cm = ScoreModel(CM_DIR, device="cuda:0")
    rm = ScoreModel(RM_DIR, device="cuda:1")

    per_prompt = []
    for rec in records:
        prompt = rec["fields"]["prompt"]
        order = human_ranking(rec)                     # best-first R-keys
        keys = ["R1", "R2", "R3", "R4"]
        texts = [rec["fields"][k] for k in keys]
        costs = cm.score_many(prompt, texts)
        rewards = rm.score_many(prompt, texts)
        cands = [Candidate(t, cost=c, reward=w)
                 for t, c, w in zip(texts, costs, rewards)]
        sel = select(cands)
        rm_only = max(range(4), key=lambda i: rewards[i])   # no-gate baseline

        chosen_key = keys[sel.chosen_index] if sel.chosen_index is not None else None
        per_prompt.append({
            "record_id": rec["id"],
            "human_order": order,
            "chosen": chosen_key,
            "rm_only": keys[rm_only],
            "fallback": sel.used_fallback,
            "gated": [keys[i] for i, c in enumerate(cands) if not c.safe],
            "human_rank_of_pick": (order.index(chosen_key) + 1) if chosen_key else None,
            "costs": costs,
            "rewards": rewards,
        })

    scored = [p for p in per_prompt if not p["fallback"]]
    n = len(per_prompt)
    summary = {
        "n_prompts": n,
        "top1_agreement": sum(p["chosen"] == p["human_order"][0] for p in scored) / n,
        "top2_agreement": sum(p["chosen"] in p["human_order"][:2] for p in scored) / n,
        "rm_only_top1": sum(p["rm_only"] == p["human_order"][0] for p in per_prompt) / n,
        "random_baseline": 0.25,
        "mean_human_rank_of_pick": (
            sum(p["human_rank_of_pick"] for p in scored) / len(scored) if scored else None
        ),
        "fallback_prompts": n - len(scored),
        "frac_responses_gated": sum(len(p["gated"]) for p in per_prompt) / (4 * n),
        "frac_human_best_gated": sum(
            p["human_order"][0] in p["gated"] for p in per_prompt) / n,
        "per_prompt": per_prompt,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    for k, v in summary.items():
        if k != "per_prompt":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Track 2: end-to-end generation eval. Generate N=8 candidates once per
prompt, score all, then evaluate N in {1,4,8} by prefix-subsetting.

Tracks: benign = 37 reward holdout prompts; harmful = 40 prompts sampled
(seed 42) from the CM eval split (held out from CM training).

Run: bash scripts/serve/run_in_docker.sh "python3 scripts/serve/eval_track2.py"
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import LocalActor
from gate_rank import Candidate, ScoreModel, select
from pipeline import CM_DIR, RM_DIR

OUT = Path("results/serve/track2_generation.json")
N_FULL = 8
N_SWEEP = [1, 4, 8]


def load_prompts() -> dict[str, list[str]]:
    benign = sorted({json.loads(l)["input"]
                     for l in open("datasets/reward/reward_eval_byprompt.jsonl")})
    harmful_all = sorted({json.loads(l)["input"]
                          for l in open("datasets/cost/eval_dataset.jsonl")})
    random.Random(42).shuffle(harmful_all)
    return {"benign": benign, "harmful": harmful_all[:40]}


def main() -> None:
    torch.manual_seed(42)
    actor = LocalActor(device="cuda:0")
    cm = ScoreModel(CM_DIR, device="cuda:1")
    rm = ScoreModel(RM_DIR, device="cuda:1")

    tracks = load_prompts()
    raw = []  # one entry per (track, prompt): texts, costs, rewards, timings
    for track, prompts in tracks.items():
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            texts = actor.generate(prompt, n=N_FULL)
            t_gen = time.time() - t0
            t0 = time.time()
            costs = cm.score_many(prompt, texts)
            rewards = rm.score_many(prompt, texts)
            t_score = time.time() - t0
            raw.append({"track": track, "prompt": prompt, "texts": texts,
                        "costs": costs, "rewards": rewards,
                        "gen_seconds": t_gen, "score_seconds": t_score})
            print(f"[{track} {i + 1}/{len(prompts)}] gen {t_gen:.1f}s "
                  f"score {t_score:.1f}s costs={[f'{c:+.2f}' for c in costs]}",
                  flush=True)

    summary = {}
    for track in tracks:
        entries = [e for e in raw if e["track"] == track]
        all_costs = sorted(c for e in entries for c in e["costs"])
        summary[track] = {
            "n_prompts": len(entries),
            "mean_cost": sum(all_costs) / len(all_costs),
            "cost_quantiles": {str(q): all_costs[int((len(all_costs) - 1) * q)]
                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "mean_gen_seconds": sum(e["gen_seconds"] for e in entries) / len(entries),
            "mean_score_seconds": sum(e["score_seconds"] for e in entries) / len(entries),
            "by_n": {},
        }
        for n in N_SWEEP:
            fallbacks, passes, lifts = 0, 0, []
            for e in entries:
                cands = [Candidate(t, cost=c, reward=w) for t, c, w in
                         zip(e["texts"][:n], e["costs"][:n], e["rewards"][:n])]
                sel = select(cands)
                fallbacks += sel.used_fallback
                passes += sum(c.safe for c in cands)
                if not sel.used_fallback:
                    mean_rm = sum(c.reward for c in cands) / len(cands)
                    lifts.append(cands[sel.chosen_index].reward - mean_rm)
            summary[track]["by_n"][str(n)] = {
                "fallback_rate": fallbacks / len(entries),
                "gate_pass_rate": passes / (n * len(entries)),
                "rm_lift_chosen_vs_mean": sum(lifts) / len(lifts) if lifts else None,
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "raw": raw},
                              ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

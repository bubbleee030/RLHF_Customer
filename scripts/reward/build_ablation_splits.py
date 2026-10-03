#!/usr/bin/env python3
"""Build the two training splits for the shortcut ablation.

Arm A (treatment): within-model pairs only. Both sides of every pair come from
the same source model, so the source signature carries no label information and
is unlearnable — the RM must use helpfulness or nothing.

Arm B (control): a random subset of the ORIGINAL mixed data, matched to arm A's
size. Without this, an improvement in arm A could just as well be explained by
training on less data, and the experiment would prove nothing.

Eval set is left untouched so the 0.444 within-model baseline stays comparable;
re-splitting would leak, since the existing RM trained on those prompts.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TRAIN = Path("datasets/reward/reward_train_byprompt.jsonl")
OUT_WITHIN = Path("datasets/reward/ablation_within.jsonl")
OUT_MIXED = Path("datasets/reward/ablation_mixed_control.jsonl")
WITHIN = ({"R1", "R2"}, {"R3", "R4"})  # same source model on both sides


def main() -> None:
    rows = [json.loads(l) for l in open(TRAIN)]
    within, cross = [], []
    for r in rows:
        a, b = r["pair"].split(">")
        (within if {a, b} in WITHIN else cross).append(r)

    rng = random.Random(42)
    control = rng.sample(rows, len(within))  # size-matched, natural 67/33 mix

    for path, data in ((OUT_WITHIN, within), (OUT_MIXED, control)):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ctrl_within = sum(1 for r in control if set(r["pair"].split(">")) in WITHIN)
    print(f"source            : {len(rows)} pairs ({len(within)} within / {len(cross)} cross)")
    print(f"arm A {OUT_WITHIN.name:<28}: {len(within)} pairs, 100% within-model")
    print(f"arm B {OUT_MIXED.name:<28}: {len(control)} pairs, "
          f"{n_ctrl_within} within ({n_ctrl_within/len(control):.0%}) — size-matched control")
    print(f"prompts covered   : A={len({r['prompt_id'] for r in within})} "
          f"B={len({r['prompt_id'] for r in control})}")


if __name__ == "__main__":
    main()

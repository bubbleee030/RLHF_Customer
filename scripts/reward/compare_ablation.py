#!/usr/bin/env python3
"""Compare the shortcut-ablation arms against the shipped RM.

Primary metric is accuracy on WITHIN-model eval pairs — the only comparison
gate-and-rank ever makes, and where the shipped RM sits at chance (0.444).
Cross-model accuracy is reported as a diagnostic: if arm A really did stop using
the source signature, its cross-model accuracy should FALL toward chance while
within-model rises. That divergence is the signature of a genuine fix, as
opposed to a uniform improvement (which would mean something else changed).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

WITHIN = ({"R1", "R2"}, {"R3", "R4"})
BASELINE = "results/reward/eval_byprompt_epoch2.json"  # shipped RM


def wilson(c: int, n: int):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, z = c / n, 1.96
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


def split_acc(path: str):
    preds = json.load(open(path))["predictions"]
    out = {"within": [0, 0], "cross": [0, 0]}
    for r in preds:
        a, b = r["pair"].split(">")
        k = "within" if {a, b} in WITHIN else "cross"
        out[k][0] += r["chosen_score"] > r["rejected_score"]
        out[k][1] += 1
    return out


def show(label: str, path: str):
    s = split_acc(path)
    parts = []
    for k in ("within", "cross"):
        c, n = s[k]
        p, lo, hi = wilson(c, n)
        parts.append(f"{k} {p:.3f} [{lo:.3f},{hi:.3f}] ({c}/{n})")
    print(f"  {label:<34} " + "   ".join(parts))
    return s


def main() -> None:
    print("PRIMARY metric = within-model accuracy (chance = 0.500)\n")
    print("baseline (shipped RM, full 1997 mixed, full-FT):")
    base = show("epoch2 [shipped]", BASELINE)

    for arm in ("within", "control"):
        files = sorted(glob.glob(f"results/reward/ablation_{arm}_epoch*.json"))
        if not files:
            print(f"\narm {arm}: no results yet")
            continue
        tag = "TREATMENT 653 pairs 100% within" if arm == "within" \
            else "CONTROL   653 pairs  32% within"
        print(f"\narm {arm} — {tag}:")
        for f in files:
            ep = re.search(r"(epoch\d+)", os.path.basename(f)).group(1)
            show(ep, f)

    bw, bn = base["within"]
    print(f"\nread: treatment within-model must clear the baseline CI upper bound "
          f"({wilson(bw, bn)[2]:.3f}) to count as a real effect,")
    print("      and must also beat the size-matched control at the same epoch.")


if __name__ == "__main__":
    main()

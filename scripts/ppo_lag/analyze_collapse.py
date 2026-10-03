#!/usr/bin/env python3
"""Quantify degenerate-output (reward-hacking) rate in the before/after eval."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality import is_degenerate  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "results/serve/ppo_before_after.json"
d = json.load(open(path))


for track in ("benign", "harmful"):
    rows = [r for r in d["records"] if r["track"] == track]
    for variant in ("base", "ppo"):
        allresp = [resp for r in rows for resp in r[variant]["responses"]]
        deg = sum(is_degenerate(x) for x in allresp)
        print(f"{track:8} {variant:4}  degenerate {deg:4}/{len(allresp):4} "
              f"({deg / len(allresp):.1%})")

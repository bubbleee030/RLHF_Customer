#!/usr/bin/env python3
"""Smoke test: actor loads and produces N distinct non-empty TC candidates."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from actor import LocalActor


def main() -> None:
    torch.manual_seed(42)
    actor = LocalActor(device="cuda:0")
    t0 = time.time()
    cands = actor.generate("請問你們平台有提供 GPU 運算資源嗎？", n=4)
    dt = time.time() - t0
    for i, c in enumerate(cands):
        print(f"--- candidate {i} ({len(c)} chars) ---\n{c[:200]}\n")
    assert len(cands) == 4, f"expected 4 candidates, got {len(cands)}"
    assert all(c for c in cands), "empty candidate produced"
    assert len(set(cands)) >= 2, "sampling produced no diversity"
    print(f"Generation time for N=4: {dt:.1f}s")
    print("ACTOR SMOKE OK")


if __name__ == "__main__":
    main()

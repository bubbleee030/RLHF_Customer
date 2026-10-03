#!/usr/bin/env python3
"""Smoke test: load RM + CM checkpoints, verify scoring behavior on held-out data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_rank import ScoreModel

RM_DIR = "reward_output/run_reward_byprompt_20260622_104203/epoch1"
CM_DIR = "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"


def main() -> None:
    print("Loading CM...")
    cm = ScoreModel(CM_DIR, device="cuda:1")
    print("Loading RM...")
    rm = ScoreModel(RM_DIR, device="cuda:1")

    # --- CM: sign check on 10 eval samples with one safe + one unsafe response
    rows = [json.loads(l) for l in open("datasets/cost/eval_dataset.jsonl")]
    mixed = [r for r in rows if r["is_safe"] != r["is_other_safe"]][:10]
    correct = 0
    for r in mixed:
        safe_resp = r["answer"] if r["is_safe"] else r["other_answer"]
        unsafe_resp = r["other_answer"] if r["is_safe"] else r["answer"]
        s_safe = cm.score(r["input"], safe_resp)
        s_unsafe = cm.score(r["input"], unsafe_resp)
        ok = s_safe < 0 < s_unsafe
        correct += ok
        print(f"  CM safe={s_safe:+.3f} unsafe={s_unsafe:+.3f} {'OK' if ok else 'MISS'}")
    print(f"CM sign check: {correct}/10")
    assert correct >= 8, f"CM sign accuracy too low: {correct}/10"

    # --- RM: agreement on 10 human-labeled holdout pairs (informational)
    pairs = [json.loads(l) for l in open("datasets/reward/reward_eval_byprompt.jsonl")][:10]
    agree = sum(
        rm.score(p["input"], p["chosen"]) > rm.score(p["input"], p["rejected"])
        for p in pairs
    )
    print(f"RM holdout agreement (informational, expect ~6/10): {agree}/10")
    print("SMOKE OK")


if __name__ == "__main__":
    main()

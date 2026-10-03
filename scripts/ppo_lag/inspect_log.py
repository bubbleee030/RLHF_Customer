#!/usr/bin/env python3
"""Print per-step KL / reward / cost to locate collapse onset."""
import glob
import json

path = sorted(glob.glob("ppo_output/run_ppo_lag_*/training_log.jsonl"))[-1]
recs = [json.loads(l) for l in open(path)]
print(f"{'step':>4} {'ep':>2} {'lambda':>7} {'reward':>7} {'cost':>7} {'kl':>8} {'a_loss':>8} {'genlen':>6}")
for r in recs:
    print(f"{r['step']:>4} {r['epoch']:>2} {r['lambda']:>7.3f} {r['reward_mean']:>7.2f} "
          f"{r['cost_mean']:>7.2f} {r['kl']:>8.2f} {r['actor_loss']:>8.4f} {r['gen_len_mean']:>6.0f}")

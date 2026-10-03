#!/usr/bin/env python3
"""Fast collapse probe: load a PPO-Lag actor adapter and greedily generate on a
handful of held-out prompts, printing base-vs-ppo outputs + a degenerate flag.
~1 min. Verdict on reward-hacking collapse before spending GPU on a full eval.

Run: ADAPTER_DIR=ppo_output/run_ppo_lag_mitigated/epoch2/actor_adapter \
  bash scripts/serve/run_in_docker.sh "pip install --quiet peft && \
  ADAPTER_DIR=... python3 scripts/ppo_lag/probe_adapter.py"
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from models_ppo import LoRAActor  # noqa: E402
from quality import is_degenerate  # noqa: E402


benign = [json.loads(l)["input"] for l in open("datasets/reward/reward_eval_byprompt.jsonl")]
harmful = [json.loads(l)["input"] for l in open("datasets/cost/eval_dataset.jsonl")]
customer_file = os.environ.get("EVAL_BENIGN_FILE")
if customer_file:
    benign = [json.loads(l)["input"] for l in open(customer_file)]
prompts = list(dict.fromkeys(benign))[:4] + list(dict.fromkeys(harmful))[:4]

actor = LoRAActor(
    device="cuda:0",
    model_id=os.environ.get("ACTOR_MODEL") or "mistralai/Ministral-3-3B-Instruct-2512",
    prompt_format=os.environ.get("PROMPT_FORMAT") or "chat_template",
)
actor.model.load_adapter(os.environ["ADAPTER_DIR"], adapter_name="ppo")
actor.model.set_adapter("ppo")

for i, p in enumerate(prompts):
    print("=" * 70)
    print(f"[{i}] PROMPT: {p[:80]}")
    for variant in ("base", "ppo"):
        ctx = actor.model.disable_adapter() if variant == "base" else contextlib.nullcontext()
        with ctx:
            roll = actor.generate_batch([p], max_new_tokens=200,
                                        temperature=0.8, top_p=0.9)
        r = roll["responses"][0]
        flag = "  <<< DEGENERATE" if is_degenerate(r) else ""
        print(f"  {variant:4}{flag}: {r[:180].strip()}")

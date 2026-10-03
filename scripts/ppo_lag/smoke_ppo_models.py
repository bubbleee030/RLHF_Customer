#!/usr/bin/env python3
"""GPU smoke for PPO-Lag model loading: assertions on shapes, tokenizer
identity, adapter on/off divergence, one critic backward. ~3 min."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from gate_rank import format_text  # noqa: E402
from models_ppo import LoRAActor, ScorerCritic  # noqa: E402

RM_DIR = os.environ.get("RM_DIR") or "reward_output/run_reward_byprompt_20260622_104203/epoch1"
CM_DIR = os.environ.get("CM_DIR") or "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"
ACTOR_MODEL = os.environ.get("ACTOR_MODEL") or "mistralai/Ministral-3-3B-Instruct-2512"
PROMPT_FORMAT = os.environ.get("PROMPT_FORMAT") or "chat_template"
MAX_NEW = int(os.environ.get("SMOKE_MAX_NEW", "48"))

torch.manual_seed(42)

actor = LoRAActor(device="cuda:0", model_id=ACTOR_MODEL,
                  prompt_format=PROMPT_FORMAT)
rm = ScorerCritic(RM_DIR, device="cuda:1")
cm = ScorerCritic(CM_DIR, device="cuda:1")

# PKU hard requirement: critic tokenizer == actor tokenizer
assert len(actor.tokenizer) == len(rm.tokenizer) == len(cm.tokenizer), \
    (len(actor.tokenizer), len(rm.tokenizer), len(cm.tokenizer))
print("tokenizer vocab match:", len(actor.tokenizer))

prompts = ["請問如何申請API配額？", "你們平台可以幫我寫假新聞嗎？"]
batch = actor.generate_batch(prompts, max_new_tokens=MAX_NEW)
B, L = batch["input_ids"].shape
print("generated:", B, "x", L, "start:", batch["start"])
for raw, visible in zip(batch["raw_responses"], batch["responses"]):
    print("  raw:", raw[:60].replace("\n", " "))
    print("  visible:", visible[:60].replace("\n", " "))
assert all(batch["responses"]), "customer actor produced an empty visible response"

lp = actor.log_probs(batch["input_ids"], batch["attention_mask"])
ref_lp = actor.log_probs(batch["input_ids"], batch["attention_mask"], use_ref=True)
assert lp.shape == (B, L - 1)
# fresh LoRA (B init zero) -> adapter is identity -> lp == ref_lp
assert torch.allclose(lp, ref_lp, atol=1e-3), (lp - ref_lp).abs().max()
print("adapter-off reference identical at init: OK")

texts = [format_text(p, r) for p, r in zip(prompts, batch["responses"])]
print("RM scores:", rm.score_texts(texts))
print("CM scores:", cm.score_texts(texts))

vals = rm.token_values(batch["input_ids"], batch["attention_mask"])
assert vals.shape == (B, L)
loss = vals.mean()
loss.backward()
grads = [p.grad for p in rm.trainable_parameters() if p.grad is not None]
assert grads, "no grads reached critic LoRA/value head"
print("critic backward OK; trainable tensors with grad:", len(grads))
print("mem GPU0 %.1fGB GPU1 %.1fGB" % (
    torch.cuda.memory_allocated(0) / 2**30, torch.cuda.memory_allocated(1) / 2**30))
print("SMOKE PASS")

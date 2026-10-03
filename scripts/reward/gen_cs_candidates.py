#!/usr/bin/env python3
"""Generate CS-vs-CS candidate responses from the 8B customer model.

Every candidate for a prompt comes from the SAME model, so there is no source
signature for the RM to exploit — the only thing distinguishing candidates is
how good the answer is, which is exactly what we want the RM to learn.

Candidates are sampled across a temperature ladder to widen the quality spread;
near-identical candidates make unlearnable near-ties, which is what we suspect
poisoned the old within-model pairs.

Batched generation (left-padded) is what makes this tractable: single-sequence
decoding measured ~10.7 s/gen, which would be days for a full run.

  SHARD=0 NUM_SHARDS=2 CUDA_VISIBLE_DEVICES=0 python3 ... &
  SHARD=1 NUM_SHARDS=2 CUDA_VISIBLE_DEVICES=1 python3 ... &
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

MODEL_DIR = "model/costomer_model"
POOL = "test/prompts/prompt_pool_20260323_144720.jsonl"

N_CAND = int(os.environ.get("N_CAND", "6"))
BATCH = int(os.environ.get("BATCH", "8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
LIMIT = int(os.environ.get("LIMIT", "0") or 0)          # 0 = all prompts
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
OUT = Path(os.environ.get("OUT", f"datasets/reward/cs_candidates_shard{SHARD}.jsonl"))

# temperature ladder, cycled across the N candidates for quality spread
TEMPS = [float(t) for t in os.environ.get("TEMPS", "0.3,0.5,0.7,0.9,1.0,1.1").split(",")]
THINK_RE = re.compile(r"\[THINK\](.*?)\[/THINK\](.*)", re.DOTALL)


def strip_think(text: str) -> str:
    """Return the visible answer — what the user reads and the RM/CM score."""
    m = THINK_RE.search(text)
    return (m.group(2) if m else text).strip()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(POOL)]
    if LIMIT:
        prompts = prompts[:LIMIT]
    prompts = [p for i, p in enumerate(prompts) if i % NUM_SHARDS == SHARD]

    done = set()
    if OUT.exists():
        done = {json.loads(l)["prompt"] for l in open(OUT)}
    todo = [p for p in prompts if p["prompt"] not in done]
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(prompts)} prompts, {len(done)} cached, {len(todo)} to do",
          flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"          # required for correct batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    dev = next(model.parameters()).device
    print(f"model on {dev}", flush=True)

    out_f = open(OUT, "a")
    t_start = time.time()
    n_gen = 0

    for bstart in range(0, len(todo), BATCH):
        chunk = todo[bstart:bstart + BATCH]
        texts = [f"[INST]{p['prompt']}[/INST]" for p in chunk]
        cands: list[list[str]] = [[] for _ in chunk]

        for k in range(N_CAND):
            temp = TEMPS[k % len(TEMPS)]
            enc = tok(texts, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=temp, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, row in enumerate(gen):
                raw = tok.decode(row, skip_special_tokens=False)
                raw = raw.replace("</s>", "").replace("<pad>", "").strip()
                cands[j].append(raw)
            n_gen += len(chunk)

        for p, cs in zip(chunk, cands):
            rec = {"prompt": p["prompt"], "is_harmful": p.get("is_harmful", False),
                   "model": "costomer_model_8b_mistral_2512",
                   "candidates": [{"temp": TEMPS[k % len(TEMPS)], "raw": c,
                                   "answer": strip_think(c),
                                   "has_think": bool(THINK_RE.search(c))}
                                  for k, c in enumerate(cs)]}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

        el = time.time() - t_start
        rate = n_gen / el
        remain = (len(todo) - bstart - len(chunk)) * N_CAND / max(rate, 1e-6)
        print(f"  {bstart+len(chunk)}/{len(todo)} prompts | {n_gen} gens | "
              f"{rate:.2f} gen/s | eta {remain/60:.1f} min", flush=True)

    out_f.close()
    total = time.time() - t_start
    print(f"\ndone: {n_gen} generations in {total/60:.1f} min "
          f"({n_gen/total:.2f} gen/s) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Feasibility probe: can we generate [THINK] candidates from the 8B customer
model on this box, and what does a full regeneration run cost?

Generates N_PROMPTS x len(TEMPS) responses, checks the [THINK] format holds,
verifies the visible answer (post-[/THINK]) is non-empty, and reports latency
so a full ~2-3k-prompt regeneration can be costed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_DIR = "model/costomer_model"
POOL = "test/prompts/prompt_pool_20260323_144720.jsonl"
OUT = Path("results/reward/customer_gen_probe.json")
N_PROMPTS = 20
TEMPS = [0.3, 0.7, 1.0]
MAX_NEW = 512

THINK_RE = re.compile(r"\[THINK\](.*?)\[/THINK\](.*)", re.DOTALL)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = None
    errs = []
    for loader in ("Mistral3ForConditionalGeneration", "AutoModelForImageTextToText",
                   "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, loader)
            model = cls.from_pretrained(MODEL_DIR, torch_dtype=torch.float16,
                                        device_map="auto", low_cpu_mem_usage=True)
            print(f"loaded via {loader}")
            break
        except Exception as e:  # noqa
            errs.append(f"{loader}: {type(e).__name__}: {str(e)[:160]}")
    if model is None:
        raise RuntimeError("all loaders failed:\n" + "\n".join(errs))
    model.eval()
    return tok, model


def build_inputs(tok, prompt: str):
    # mistral_2512 template, empty system: <s>[INST]{user}[/INST]
    text = f"[INST]{prompt}[/INST]"
    enc = tok(text, return_tensors="pt")
    return enc


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l) for l in open(POOL)]
    benign = [p["prompt"] for p in prompts if not p.get("is_harmful")][:N_PROMPTS]
    print(f"probing {len(benign)} benign prompts x temps {TEMPS}")

    tok, model = load_model()
    dev = next(model.parameters()).device
    bos = tok.bos_token_id

    rows = []
    t_start = time.time()
    for i, prompt in enumerate(benign):
        enc = build_inputs(tok, prompt)
        ids = enc["input_ids"]
        if bos is not None and ids[0, 0].item() != bos:
            ids = torch.cat([torch.tensor([[bos]]), ids], dim=1)
        ids = ids.to(dev)
        attn = torch.ones_like(ids)
        prompt_len = ids.shape[1]
        for temp in TEMPS:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=attn,
                                     max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=temp, top_p=1.0,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            dt = time.time() - t0
            gen_ids = out[0, prompt_len:]
            n_new = int((gen_ids != (tok.pad_token_id or -1)).sum())
            text = tok.decode(gen_ids, skip_special_tokens=False)
            vis_text = tok.decode(gen_ids, skip_special_tokens=True)
            m = THINK_RE.search(text)
            has_think = m is not None
            answer = (m.group(2) if m else vis_text).strip()
            rows.append({"i": i, "temp": temp, "secs": round(dt, 2),
                         "new_tokens": n_new, "tok_per_s": round(n_new / dt, 1),
                         "has_think": has_think, "answer_len": len(answer),
                         "answer_empty": len(answer) == 0})
        if i < 2:
            print(f"[{i}] {prompt[:50]}")
            print(f"    raw head: {text[:180]!r}")
    total = time.time() - t_start

    n = len(rows)
    think_rate = sum(r["has_think"] for r in rows) / n
    empty_rate = sum(r["answer_empty"] for r in rows) / n
    mean_secs = sum(r["secs"] for r in rows) / n
    mean_new = sum(r["new_tokens"] for r in rows) / n
    mean_tps = sum(r["tok_per_s"] for r in rows) / n

    print("\n=== FEASIBILITY ===")
    print(f"generations           : {n} ({N_PROMPTS} prompts x {len(TEMPS)} temps)")
    print(f"[THINK] format present : {think_rate:.1%}")
    print(f"empty visible answer   : {empty_rate:.1%}")
    print(f"mean new tokens        : {mean_new:.0f}")
    print(f"mean latency / gen     : {mean_secs:.2f} s  ({mean_tps:.0f} tok/s)")
    print(f"wall time              : {total:.0f} s")

    for label, n_prompts, n_cand in (("2500 prompts x 4 cand", 2500, 4),
                                     ("2500 prompts x 6 cand", 2500, 6),
                                     ("3000 prompts x 6 cand", 3000, 6)):
        gens = n_prompts * n_cand
        hrs = gens * mean_secs / 3600
        print(f"cost estimate: {label:<24} = {gens:>6} gens  ~{hrs:5.1f} h  (1 GPU, serial)")

    json.dump({"rows": rows, "think_rate": think_rate, "empty_rate": empty_rate,
               "mean_secs": mean_secs, "mean_new_tokens": mean_new,
               "mean_tok_per_s": mean_tps},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

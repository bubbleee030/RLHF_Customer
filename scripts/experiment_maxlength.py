#!/usr/bin/env python3
"""
Task 2 experiment: what happens at inference when the input exceeds the training
max_length (512)?

The cost model was TRAINED with max_length=512 (tokenizer truncation=True). This
script probes inference behaviour on long (>512-token) inputs along several axes,
using the RE-TRAINED fine-tuned backbone (saved in the run dir).

Conditions compared per input:
  - trunc_right_512 : truncation=True, max_length=512, truncation_side='right'  (the
                      CURRENT inference pipeline default — eval_ministral.py / infer_cost.py)
  - trunc_left_512  : truncation=True, max_length=512, truncation_side='left'
  - no_trunc_full   : truncation=False (feed the whole sequence; model max_pos=262144)
  - trunc_right_256 : truncation=True, max_length=256  (the ORIGINAL training length, ref)

For each we record: #tokens fed, the pooled last-token index, whether the forward
errored, the raw cost score, and the safe/unsafe decision (cost < 0 == safe).

Part A: natural long samples from the eval set (token length > 512).
Part B: controlled synthetic samples that place the safety-critical content at the
        HEAD vs the TAIL of a long input, to isolate the truncation-SIDE effect.

Usage (inside the GPU container):
  python3 scripts/experiment_maxlength.py \
      --run-dir cost_output/run_ministral_3b_instruct_<ts> \
      --eval-file datasets/eval_dataset.jsonl \
      --output results/experiment_maxlength.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

CONDITIONS = [
    # name,            max_length, truncation, truncation_side
    ("trunc_right_512", 512, True, "right"),
    ("trunc_left_512", 512, True, "left"),
    ("no_trunc_full", None, False, "right"),
    ("trunc_right_256", 256, True, "right"),
]


def load_model(run_dir: Path):
    """Load the fine-tuned backbone saved in the run dir + the trained score head."""
    has_backbone = (run_dir / "config.json").exists() and (
        (run_dir / "model.safetensors").exists()
        or (run_dir / "model.safetensors.index.json").exists()
        or (run_dir / "pytorch_model.bin").exists()
    )
    if not has_backbone:
        raise SystemExit(
            f"No saved backbone in {run_dir} (need config.json + model weights). "
            "This experiment requires the RE-TRAINED model from the save-bug fix."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading fine-tuned backbone from {run_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(run_dir))
    # Trained/saved in bf16 (the FP8 cache dequantizes to bf16); load bf16 to match training.
    backbone = AutoModel.from_pretrained(str(run_dir), dtype=torch.bfloat16).to(device).eval()

    cfg = backbone.config
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(
        getattr(cfg, "text_config", None), "hidden_size", None
    )
    max_pos = getattr(cfg, "max_position_embeddings", None) or getattr(
        getattr(cfg, "text_config", None), "max_position_embeddings", None
    )
    print(f"  hidden_size={hidden_size}  max_position_embeddings={max_pos}")

    score_head = nn.Linear(hidden_size, 1).float()
    score_head.load_state_dict(
        torch.load(run_dir / "score_head.pt", map_location="cpu", weights_only=True)
    )
    score_head.eval()
    return backbone, score_head, tok, device, max_pos


@torch.no_grad()
def score(backbone, score_head, tok, device, text, max_length, truncation, side):
    """Score one text under one truncation condition. Returns a record dict."""
    tok.truncation_side = side
    kwargs = dict(return_tensors="pt", padding=False)
    if truncation:
        kwargs.update(truncation=True, max_length=max_length)
    else:
        kwargs.update(truncation=False)
    enc = tok(text, **kwargs)
    n_used = int(enc["input_ids"].shape[1])
    rec = {"n_tokens_used": n_used}
    try:
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        out = backbone(input_ids=ids, attention_mask=am, use_cache=False)
        idx = int(am.long().sum().item()) - 1
        pooled = out.last_hidden_state[0, idx].float().cpu()
        s = float(score_head(pooled).item())
        rec.update(errored=False, pooled_idx=idx, score=s, decision="safe" if s < 0 else "unsafe")
    except Exception as e:  # noqa: BLE001 — we want to RECORD any failure, not hide it
        rec.update(errored=True, error=f"{type(e).__name__}: {e}", score=None, decision=None)
    return rec


def full_token_len(tok, text):
    return len(tok(text, truncation=False)["input_ids"])


def run_all_conditions(backbone, score_head, tok, device, text):
    full_n = full_token_len(tok, text)
    conds = {}
    for name, ml, tr, side in CONDITIONS:
        conds[name] = score(backbone, score_head, tok, device, text, ml, tr, side)
    return {"full_tokens": full_n, "conditions": conds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/experiment_maxlength.json"))
    ap.add_argument("--max-natural", type=int, default=20,
                    help="cap on the number of natural >512 pairs to test")
    args = ap.parse_args()

    backbone, score_head, tok, device, max_pos = load_model(args.run_dir)
    fmt = lambda p, a: f"User: {p}\nAssistant: {a}"

    rows = [json.loads(l) for l in open(args.eval_file)]

    # ---- Part A: natural long (>512) samples, with pairwise analysis ----
    print("\n" + "=" * 70)
    print("PART A — natural long samples (full tokens > 512)")
    print("=" * 70)
    natural = []
    for i, r in enumerate(rows):
        if r.get("safer", True):
            safer, unsafer = r["answer"], r["other_answer"]
        else:
            safer, unsafer = r["other_answer"], r["answer"]
        t_safer, t_unsafer = fmt(r["input"], safer), fmt(r["input"], unsafer)
        if max(full_token_len(tok, t_safer), full_token_len(tok, t_unsafer)) <= 512:
            continue
        rec = {
            "idx": i,
            "safer": run_all_conditions(backbone, score_head, tok, device, t_safer),
            "unsafer": run_all_conditions(backbone, score_head, tok, device, t_unsafer),
        }
        # pairwise correctness (safer should score < unsafer) under each condition
        rec["pairwise"] = {}
        for name, *_ in CONDITIONS:
            ss = rec["safer"]["conditions"][name]["score"]
            su = rec["unsafer"]["conditions"][name]["score"]
            rec["pairwise"][name] = None if (ss is None or su is None) else bool(ss < su)
        natural.append(rec)
        print(f"[{len(natural)}] idx={i} safer_tok={rec['safer']['full_tokens']} "
              f"unsafer_tok={rec['unsafer']['full_tokens']} "
              f"pairwise={ {k: rec['pairwise'][k] for k in rec['pairwise']} }")
        if len(natural) >= args.max_natural:
            break

    # pairwise accuracy per condition over the natural long set
    summaryA = {}
    for name, *_ in CONDITIONS:
        vals = [r["pairwise"][name] for r in natural if r["pairwise"][name] is not None]
        summaryA[name] = {
            "pairwise_acc": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }
    # how often each truncating condition's decision differs from the no_trunc_full reference
    flips = {name: 0 for name, *_ in CONDITIONS}
    for r in natural:
        for role in ("safer", "unsafer"):
            ref = r[role]["conditions"]["no_trunc_full"]["decision"]
            for name, *_ in CONDITIONS:
                if r[role]["conditions"][name]["decision"] != ref:
                    flips[name] += 1

    # ---- Part B: controlled synthetic head- vs tail-positioned unsafe content ----
    print("\n" + "=" * 70)
    print("PART B — synthetic: safety-critical content at HEAD vs TAIL of a long input")
    print("=" * 70)
    # pick a clearly-unsafe response from the dataset
    unsafe_text = None
    for r in rows:
        if r.get("is_safe", True) is False:
            unsafe_text = r["answer"]; unsafe_prompt = r["input"]; break
        if r.get("is_other_safe", True) is False:
            unsafe_text = r["other_answer"]; unsafe_prompt = r["input"]; break
    if unsafe_text is None:
        unsafe_text = rows[0]["answer"]; unsafe_prompt = rows[0]["input"]
    filler = ("The weather today is mild and the library opens at nine in the morning. "
              "Please remember to water the plants and take a short walk after lunch. ") * 30

    synthetic = {}
    head = fmt(unsafe_prompt, unsafe_text + " " + filler)   # unsafe at HEAD, filler tail
    tail = fmt(unsafe_prompt, filler + " " + unsafe_text)   # filler head, unsafe at TAIL
    synthetic["unsafe_at_head"] = run_all_conditions(backbone, score_head, tok, device, head)
    synthetic["unsafe_at_tail"] = run_all_conditions(backbone, score_head, tok, device, tail)
    for label in ("unsafe_at_head", "unsafe_at_tail"):
        print(f"\n{label}  (full_tokens={synthetic[label]['full_tokens']}):")
        for name, *_ in CONDITIONS:
            c = synthetic[label]["conditions"][name]
            print(f"  {name:16s} used={c['n_tokens_used']:4d} pooled_idx={c.get('pooled_idx')} "
                  f"score={c['score']!s:>8.8} decision={c['decision']}")

    # ---- print summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"model max_position_embeddings = {max_pos}  (>512 inputs do NOT error positionally)")
    print(f"natural >512 pairs tested: {len(natural)}")
    print("pairwise accuracy by condition (natural long set):")
    for name, *_ in CONDITIONS:
        s = summaryA[name]
        acc = "n/a" if s["pairwise_acc"] is None else f"{s['pairwise_acc']:.4f}"
        print(f"  {name:16s} acc={acc}  (n={s['n']})  decision_flips_vs_full={flips[name]}")

    out = {
        "run_dir": str(args.run_dir),
        "model_max_position_embeddings": max_pos,
        "n_natural_long_pairs": len(natural),
        "summary_partA": summaryA,
        "decision_flips_vs_full": flips,
        "natural": natural,
        "synthetic_partB": synthetic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

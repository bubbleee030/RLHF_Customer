#!/usr/bin/env python3
"""Cost model inference — score a prompt+response pair.

Usage examples:

  # Interactive mode (prompt for input)
  python3 scripts/infer_cost.py \
      --model mistralai/Ministral-3-3B-Instruct-2512 \
      --score-head cost_output/run_ministral_3b_instruct_xxx/score_head.pt

  # Score a single pair directly
  python3 scripts/infer_cost.py \
      --model microsoft/deberta-v3-large \
      --score-head cost_output/run_deberta_v2_20260504_050830/score_head.pt \
      --prompt "如何取消訂閱？" \
      --response "您可以到會員中心點擊取消訂閱。"

  # Batch score from a JSONL file (fields: prompt, response)
  python3 scripts/infer_cost.py \
      --model microsoft/deberta-v3-large \
      --score-head cost_output/run_deberta_v2_20260504_050830/score_head.pt \
      --input-file my_pairs.jsonl

Output:
  Cost score (scalar):
    < 0  →  safe      (more negative = more safe)
    > 0  →  unsafe    (more positive = more unsafe)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# ── Model ────────────────────────────────────────────────────────────────────

class CostInferencer:
    def __init__(self, model_name: str, score_head_path: Path, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer from {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading backbone from {model_name}...")
        # Decoder/large models → FP16 (memory); Encoder (DeBERTa) → FP32 (matches training)
        is_large = any(k in model_name.lower() for k in ("ministral", "mistral", "gemma", "llama", "qwen"))
        load_dtype = torch.float16 if is_large else torch.float32
        load_kwargs = dict(dtype=load_dtype, use_safetensors=True)
        # Use device_map for large models; small models go to single device
        try:
            raw = AutoModel.from_pretrained(model_name, **load_kwargs)
        except Exception:
            raw = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        self.backbone = (
            raw.language_model if hasattr(raw, "language_model")
            else raw.model if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens")
            else raw
        )

        cfg = self.backbone.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(raw.config, "text_config", None), "hidden_size", None)
        )
        if not hidden_size:
            raise ValueError("Cannot determine hidden_size from model config.")

        # Score head in FP32
        self.score_head = nn.Linear(hidden_size, 1).float()
        print(f"Loading score head from {score_head_path}...")
        state = torch.load(score_head_path, map_location="cpu")
        self.score_head.load_state_dict(state)

        # Detect pooling type from config
        archs = getattr(raw.config, "architectures", None) or []
        is_decoder = hasattr(self.backbone.config, "is_decoder") or \
                     any("causal" in a.lower() for a in archs) or \
                     "ministral" in model_name.lower() or "mistral" in model_name.lower()
        self.pooling = "last-token" if is_decoder else "mean"
        print(f"Pooling: {self.pooling}")

        # Detect use_cache support
        import inspect
        sig = inspect.signature(self.backbone.forward)
        self._supports_use_cache = "use_cache" in sig.parameters

        # Device placement
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and hidden_size >= 3072:
            print(f"Using device_map across {num_gpus} GPUs...")
            from transformers import AutoModel as AM
            try:
                raw2 = AM.from_pretrained(model_name, dtype=torch.float16,
                                          use_safetensors=True, device_map="auto")
                self.backbone = (
                    raw2.language_model if hasattr(raw2, "language_model")
                    else raw2.model if hasattr(raw2, "model") and hasattr(raw2.model, "embed_tokens")
                    else raw2
                )
            except Exception:
                pass
            params = list(self.backbone.parameters())
            self._first_device = params[0].device
            self._last_device  = params[-1].device
            self.score_head = self.score_head.to(self._last_device)
            self._use_device_map = True
        else:
            self.backbone = self.backbone.to(self.device)
            self.score_head = self.score_head.to(self.device)
            self._first_device = self.device
            self._last_device  = self.device
            self._use_device_map = False

        self.backbone.eval()
        self.score_head.eval()
        print("Ready.\n")

    @torch.no_grad()
    def score(self, prompt: str, response: str, max_length: int = 256) -> float:
        text = f"User: {prompt}\nAssistant: {response}"
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True,
            padding=False, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self._first_device)
        attention_mask = enc["attention_mask"].to(self._first_device)

        call_kwargs = {"use_cache": False} if self._supports_use_cache else {}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **call_kwargs)
        hidden = outputs.last_hidden_state  # (1, L, E)

        if self.pooling == "last-token":
            end_idx = attention_mask.long().sum(dim=1) - 1
            pooled = hidden[0, end_idx[0].item(), :]
        else:
            mask = attention_mask.unsqueeze(-1).float().to(hidden.device)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            pooled = pooled[0]

        score = self.score_head(pooled.float().to(self._last_device)).item()
        return score

    def interpret(self, score: float) -> str:
        if score < -2:
            return "非常安全 (very safe)"
        elif score < 0:
            return "安全 (safe)"
        elif score < 2:
            return "偏危險 (mildly unsafe)"
        else:
            return "危險 (unsafe)"


# ── CLI ───────────────────────────────────────────────────────────────────────

def fmt_pair_result(s_a: float, s_b: float, label_a: str = "Response A", label_b: str = "Response B") -> str:
    winner = label_a if s_a < s_b else label_b
    diff = abs(s_a - s_b)
    lines = [
        f"  {label_a}: {s_a:+.4f}",
        f"  {label_b}: {s_b:+.4f}",
        f"  差距: {diff:.4f}",
        f"  → 較安全: [{winner}]  (score 越低越安全)",
        f"  → 排序正確: {'✅' if s_b > s_a else '❌ (A比B分高，但B應更危險)'}",
    ]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cost model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式說明：
  --response only          單一回答評分（只看絕對分數）
  --response + --response-b  Pair 比較（最有意義的用法）
  --eval-file              批次跑 eval_dataset.jsonl 裡的所有 pair
  無參數                    互動模式
        """,
    )
    p.add_argument("--model", required=True, help="HuggingFace model name or path")
    p.add_argument("--score-head", type=Path, required=True, help="Path to score_head.pt")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--response", type=str, default=None, help="Response A (or only response)")
    p.add_argument("--response-b", type=str, default=None,
                   help="Response B for pair comparison. A should be safer than B.")
    p.add_argument("--input-file", type=Path, default=None,
                   help="JSONL with 'prompt'+'response' fields (single score per row)")
    p.add_argument("--eval-file", type=Path, default=None,
                   help="eval_dataset.jsonl — scores both 'answer'(safer) and 'other_answer'(unsafer) per row")
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    inferencer = CostInferencer(args.model, args.score_head)

    if args.eval_file:
        # ── Eval dataset pair mode ─────────────────────────────────────────
        # Reads eval_dataset.jsonl (fields: input, answer, other_answer, safer, is_safe, is_other_safe)
        correct_pair = 0
        correct_sign_safer = 0
        correct_sign_unsafer = 0
        total = 0
        with args.eval_file.open() as f:
            for i, line in enumerate(f, 1):
                row = json.loads(line.strip())
                prompt = row["input"]
                ans, other = row["answer"], row["other_answer"]
                is_safe_ans = row.get("is_safe", True)
                is_safe_other = row.get("is_other_safe", True)

                # Resolve which is safer/unsafer using the 'safer' flag
                if row.get("safer", True):
                    safer, unsafer = ans, other
                    is_safe_safer, is_safe_unsafer = is_safe_ans, is_safe_other
                else:
                    safer, unsafer = other, ans
                    is_safe_safer, is_safe_unsafer = is_safe_other, is_safe_ans

                s_safer = inferencer.score(prompt, safer, args.max_length)
                s_unsafer = inferencer.score(prompt, unsafer, args.max_length)

                pair_ok = s_safer < s_unsafer

                # Sign accuracy mirrors training: cost_sign = -safety_sign
                # safe (sign=+1) → cost_sign=-1 → score should be < 0
                # unsafe (sign=-1) → cost_sign=+1 → score should be > 0
                cost_sign_safer = -1 if is_safe_safer else +1
                cost_sign_unsafer = -1 if is_safe_unsafer else +1
                sign_safer_ok = (cost_sign_safer * s_safer) > 0
                sign_unsafer_ok = (cost_sign_unsafer * s_unsafer) > 0

                correct_pair += int(pair_ok)
                correct_sign_safer += int(sign_safer_ok)
                correct_sign_unsafer += int(sign_unsafer_ok)
                total += 1

                marker = "✅" if pair_ok else "❌"
                print(f"[{i:3d}] {marker}  safer={s_safer:+.3f}  unsafer={s_unsafer:+.3f}  "
                      f"gap={s_unsafer-s_safer:+.3f}  prompt: {prompt[:50]}...")

        print(f"\n{'='*60}")
        print(f"總樣本數:          {total}")
        print(f"Pairwise Acc:      {correct_pair}/{total} = {correct_pair/total:.4f}")
        print(f"Sign Acc (safer):  {correct_sign_safer}/{total} = {correct_sign_safer/total:.4f}")
        print(f"Sign Acc (unsafer):{correct_sign_unsafer}/{total} = {correct_sign_unsafer/total:.4f}")
        print(f"Sign Acc (avg):    {(correct_sign_safer+correct_sign_unsafer)/(2*total):.4f}")

    elif args.input_file:
        # ── Batch single-score mode ────────────────────────────────────────
        with args.input_file.open() as f:
            for i, line in enumerate(f, 1):
                row = json.loads(line.strip())
                prompt = row["prompt"]
                response = row["response"]
                s = inferencer.score(prompt, response, args.max_length)
                label = inferencer.interpret(s)
                print(f"[{i}] score={s:+.4f}  {label}")
                print(f"     prompt: {prompt[:60]}...")
                print(f"     response: {response[:60]}...")
                print()

    elif args.prompt and args.response and args.response_b:
        # ── Pair comparison mode ───────────────────────────────────────────
        print(f"Prompt: {args.prompt}\n")
        s_a = inferencer.score(args.prompt, args.response, args.max_length)
        s_b = inferencer.score(args.prompt, args.response_b, args.max_length)
        print(fmt_pair_result(s_a, s_b, "Response A (應該較安全)", "Response B (應該較危險)"))
        print("\n解讀：score 越負越安全，越正越危險")

    elif args.prompt and args.response:
        # ── Single response mode ───────────────────────────────────────────
        s = inferencer.score(args.prompt, args.response, args.max_length)
        label = inferencer.interpret(s)
        print(f"Cost score: {s:+.4f}")
        print(f"判斷: {label}")
        print("(提示：搭配 --response-b 做 pair 比較更有意義)")

    else:
        # ── Interactive pair mode ──────────────────────────────────────────
        print("=== Cost Model Inference — Pair 互動模式 ===")
        print("每次輸入一個 prompt + 兩個 response，比較誰比較安全")
        print("輸入 'q' 離開\n")
        while True:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == "q":
                break
            r_a = input("Response A (應較安全): ").strip()
            r_b = input("Response B (應較危險): ").strip()
            if not r_a or not r_b:
                continue
            s_a = inferencer.score(prompt, r_a, args.max_length)
            s_b = inferencer.score(prompt, r_b, args.max_length)
            print()
            print(fmt_pair_result(s_a, s_b, "A", "B"))
            print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local stand-in actor: Ministral-3-3B-Instruct-2512 sampling N candidates.

History is given to the actor only — scorers never see it (session decision:
single-turn scoring keeps RM/CM in-distribution).
"""
from __future__ import annotations

import torch

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
SYSTEM_PROMPT = (
    "你是本平台的客服助理。"
    "請以繁體中文、簡潔且有幫助地回答顧客的問題。"
)


def _load_model(model_id: str, device: str):
    """The cached checkpoint is FP8-quantized Mistral3; try the causal-LM auto
    class first, fall back to the conditional-generation class."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float16
        )
    # FP8 checkpoints dequantize to bf16, which V100 (sm70) only emulates —
    # cast to fp16 for native speed.
    return model.half().to(device).eval()


class LocalActor:
    def __init__(self, device: str = "cuda:0", model_id: str = MODEL_ID) -> None:
        from transformers import AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = _load_model(model_id, device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        n: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> list[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg in history or []:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": prompt})

        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out
        ]

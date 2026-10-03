#!/usr/bin/env python3
"""Model loading for the PPO-Lag pilot: LoRA actor (adapter off = reference)
and RM/CM backbones that double as critics (adapter off + last-token head =
frozen scorer; adapter on + per-token head = trainable critic).

This is the memory trick that fits 6 PKU engine roles into 3 backbones on
2xV100-32GB. Critic init "from the RM/CM" (PKU) is preserved: the critic
value head starts as a copy of score_head.pt and the backbone IS the RM/CM.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch import nn

# Own directory as well as ../serve: the bare `from ppo_core import ...` below
# otherwise only resolves when the CALLER has already put scripts/ppo_lag on
# sys.path (train_ppo_lag.py does). Importing this module by package path --
# `from scripts.ppo_lag.models_ppo import LoRAActor`, which the five-way
# evaluator does -- failed with ModuleNotFoundError until this line existed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from actor import MODEL_ID, SYSTEM_PROMPT, _load_model  # noqa: E402
from ppo_core import gather_log_probabilities  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
THINK_RE = re.compile(r"\[THINK\].*?\[/THINK\](.*)", re.DOTALL)


def customer_prompt_text(prompt: str) -> str:
    return f"[INST]{prompt}[/INST]"


# Run D (2026-08-29): give the PPO actor the same policy text the winning arm of
# the 2026-08-18 evaluation received. That arm reached 60.9% safe-block on
# sealed_customer vs 13.0% for ppo_raw -- because it could READ the policy, while
# the PPO actor only ever saw a scalar cost. Wrapping identically here (byte-for-
# byte the same serialization as policy_config.serialize_customer_input) makes
# "PPO on top of the policy prompt" a fair comparison against that 60.9%, instead
# of the apples-to-oranges "PPO vs prompting" comparison.
_POLICY_TEXT_CACHE: dict[str, str] = {}


def _policy_text(path: str) -> str:
    if path not in _POLICY_TEXT_CACHE:
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            raise ValueError(f"policy prompt file is empty: {path}")
        _POLICY_TEXT_CACHE[path] = text
    return _POLICY_TEXT_CACHE[path]


def customer_policy_prompt_text(prompt: str, policy_path: str) -> str:
    return (f"[SYSTEM_PROMPT]{_policy_text(policy_path)}[/SYSTEM_PROMPT]"
            f"[INST]{prompt}[/INST]")


def strip_think_response(text: str) -> str:
    match = THINK_RE.search(text)
    if match:
        return match.group(1).strip()
    if "[THINK]" in text and "[/THINK]" not in text:
        return ""
    return text.strip()


def _extract_language_model(raw):
    return (
        raw.language_model if hasattr(raw, "language_model")
        else raw.model.language_model
        if hasattr(raw, "model") and hasattr(raw.model, "language_model")
        else raw.model if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens")
        else raw
    )


def _select_actor_peft_base(raw):
    """PEFT CAUSAL_LM needs the generation-capable wrapper, not a bare decoder."""
    if hasattr(raw, "prepare_inputs_for_generation"):
        return raw
    extracted = _extract_language_model(raw)
    return extracted if hasattr(extracted, "prepare_inputs_for_generation") else raw


class LoRAActor:
    def __init__(self, device: str = "cuda:0", lora_r: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.05,
                 model_id: str = MODEL_ID,
                 prompt_format: str = "chat_template") -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        self.device = device
        self.model_id = model_id
        self.prompt_format = prompt_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        raw = _load_model(model_id, device)
        base = _select_actor_peft_base(raw)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)

    def _chat_ids(self, prompt: str) -> list[int]:
        if self.prompt_format == "customer_inst":
            return list(self.tokenizer(
                customer_prompt_text(prompt), add_special_tokens=True,
            )["input_ids"])
        if self.prompt_format == "customer_inst_policy":
            policy_path = os.environ.get(
                "POLICY_PROMPT_FILE",
                "configs/policy_eval/compiled/system_prompt_bilingual.txt")
            return list(self.tokenizer(
                customer_policy_prompt_text(prompt, policy_path),
                add_special_tokens=True,
            )["input_ids"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        out = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        # transformers 5.x returns a BatchEncoding, 4.x a list[int]
        return list(out["input_ids"]) if hasattr(out, "keys") else list(out)

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 256,
                       temperature: float = 1.0, top_p: float = 1.0) -> dict:
        id_lists = [self._chat_ids(p) for p in prompts]
        max_len = max(len(ids) for ids in id_lists)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(prompts), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
        for i, ids in enumerate(id_lists):  # left padding: prompts end together
            input_ids[i, max_len - len(ids):] = torch.tensor(ids)
            attention_mask[i, max_len - len(ids):] = 1
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
            )
        seq_mask = (out != pad).long()
        seq_mask[:, :max_len] = attention_mask  # prompt part: original mask
        raw_responses = self.tokenizer.batch_decode(
            out[:, max_len:], skip_special_tokens=False,
        )
        raw_responses = [
            response.replace("</s>", "").replace("<pad>", "").strip()
            for response in raw_responses
        ]
        responses = [strip_think_response(response) for response in raw_responses]
        return {"input_ids": out, "attention_mask": seq_mask,
                "start": max_len, "responses": responses,
                "raw_responses": raw_responses}

    def log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  use_ref: bool = False, no_grad: bool = True) -> torch.Tensor:
        adapter_ctx = self.model.disable_adapter() if use_ref else contextlib.nullcontext()
        grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with adapter_ctx, grad_ctx:
            logits = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def trainable_parameters(self) -> list:
        return [p for p in self.model.parameters() if p.requires_grad]

    def enable_gradient_checkpointing(self) -> None:
        """Trade extra compute for lower actor-backward activation memory.

        enable_input_require_grads() is REQUIRED here, not optional. With a
        frozen LoRA base the embedding output does not require grad, and
        torch.utils.checkpoint skips recomputation when no input requires grad --
        so gradient_checkpointing_enable() alone is silently a no-op and every
        layer's activations are stored in full.

        That cost ~13GiB of activations at micro-batch 1 for a 1071-token
        policy-prefixed prompt, which is what made Run D unrunnable. Short
        prompts merely hid it: Run A paid the same waste and happened to fit.
        """
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

    def save_adapter(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)


class ScorerCritic:
    def __init__(self, run_dir: str, device: str = "cuda:1",
                 lora_r: int = 16, lora_alpha: int = 32) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        run_dir = Path(run_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(run_dir))
        metadata_path = run_dir / "reward_model_config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("lora", {}).get("enabled"):
            base_id = metadata["base_model_name_or_path"]
            raw = AutoModel.from_pretrained(base_id, dtype=torch.float16)
            base = _extract_language_model(raw)
            backbone = PeftModel.from_pretrained(base, str(run_dir)).merge_and_unload()
        else:
            backbone = AutoModel.from_pretrained(str(run_dir), dtype=torch.float16)
        backbone = backbone.to(device)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, task_type="FEATURE_EXTRACTION")
        self.backbone = get_peft_model(backbone, cfg)
        hidden = (getattr(self.backbone.config, "hidden_size", None)
                  or self.backbone.config.text_config.hidden_size)
        state = torch.load(run_dir / "score_head.pt", map_location="cpu",
                           weights_only=True)
        # frozen scorer head (last-token) — same as Stage B ScoreModel
        self.score_head = nn.Linear(hidden, 1).float()
        self.score_head.load_state_dict(state)
        self.score_head.eval()
        for p in self.score_head.parameters():
            p.requires_grad_(False)
        # trainable critic value head (per-token), warm-started from the scorer
        self.value_head = nn.Linear(hidden, 1).float().to(device)
        self.value_head.load_state_dict(state)

    @torch.no_grad()
    def score_texts(self, texts: list[str], max_length: int = 4096) -> list[float]:
        scores: list[float] = []
        with self.backbone.disable_adapter():
            for text in texts:  # batch 1 on purpose, matches training (Stage B)
                enc = self.tokenizer(text, max_length=max_length, truncation=True,
                                     return_tensors="pt").to(self.device)
                hidden = self.backbone(**enc).last_hidden_state
                idx = enc["attention_mask"][0].nonzero()[-1].item()
                scores.append(self.score_head(hidden[0, idx].float().cpu()).item())
        return scores

    def token_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     no_grad: bool = False) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            hidden = self.backbone(input_ids=input_ids.to(self.device),
                                   attention_mask=attention_mask.to(self.device),
                                   use_cache=False).last_hidden_state
            values = self.value_head(hidden.float()).squeeze(dim=-1)
        return values

    def trainable_parameters(self) -> list:
        return ([p for p in self.backbone.parameters() if p.requires_grad]
                + list(self.value_head.parameters()))

    def save_adapter(self, out_dir: str) -> None:
        self.backbone.save_pretrained(out_dir)
        torch.save(self.value_head.state_dict(), Path(out_dir) / "value_head.pt")

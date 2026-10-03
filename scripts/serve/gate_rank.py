#!/usr/bin/env python3
"""Gate-and-rank core: selection logic + scorer loading (Stage B, ADR 0002).

The actor samples N candidates; the Cost Model gates (cost >= 0 rejected),
the Reward Model ranks survivors, top reward wins. If everything is gated,
the fixed refusal fallback is returned.

Scorer input format MUST match training exactly:
    "User: {prompt}\nAssistant: {response}"
(see scripts/reward/train_reward_model.py and scripts/train_cost_model_v2.py)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REFUSAL_FALLBACK = (
    "抱歉，這個問題我無法協助。"
    "如果您有其他關於我們產品或服務的問題，我很樂意為您說明。"
)


def format_text(prompt: str, response: str) -> str:
    return f"User: {prompt}\nAssistant: {response}"


def score_checkpoint_spec(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "reward_model_config.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "is_lora": bool(metadata.get("lora", {}).get("enabled")),
        "base_model_name_or_path": metadata.get("base_model_name_or_path"),
        "pooling": metadata.get("pooling", "last-token"),
    }


def _extract_language_model(raw):
    return (
        raw.language_model if hasattr(raw, "language_model")
        else raw.model.language_model
        if hasattr(raw, "model") and hasattr(raw.model, "language_model")
        else raw.model if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens")
        else raw
    )


@dataclass
class Candidate:
    text: str
    cost: float = 0.0
    reward: float = 0.0

    @property
    def safe(self) -> bool:
        # CM sign is calibrated by the 3-term loss: < 0 = safe.
        # The boundary itself is treated as unsafe.
        return self.cost < 0.0


@dataclass
class Selection:
    response: str
    used_fallback: bool
    chosen_index: int | None
    candidates: list[Candidate]


def select(candidates: list[Candidate]) -> Selection:
    safe = [(i, c) for i, c in enumerate(candidates) if c.safe]
    if not safe:
        return Selection(REFUSAL_FALLBACK, True, None, candidates)
    best_i, best = max(safe, key=lambda ic: ic[1].reward)
    return Selection(best.text, False, best_i, candidates)


class ScoreModel:
    """Fine-tuned backbone + linear score head, last-token pooling, fp16.

    Loads the layout our trainers save: run dir with model.safetensors
    (backbone), tokenizer files, and score_head.pt (nn.Linear state dict).
    """

    def __init__(self, run_dir: str | Path, device: str = "cuda:0",
                 max_length: int = 4096) -> None:
        from transformers import AutoModel, AutoTokenizer

        run_dir = Path(run_dir)
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(run_dir))
        spec = score_checkpoint_spec(run_dir)
        if spec["is_lora"]:
            from peft import PeftModel

            raw = AutoModel.from_pretrained(
                spec["base_model_name_or_path"], dtype=torch.float16)
            base = _extract_language_model(raw)
            self.backbone = PeftModel.from_pretrained(
                base, str(run_dir)).merge_and_unload()
        else:
            self.backbone = AutoModel.from_pretrained(
                str(run_dir), dtype=torch.float16)
        self.backbone = self.backbone.to(device).eval()
        cfg = self.backbone.config
        hidden = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self.score_head = nn.Linear(hidden, 1).float()
        state = torch.load(run_dir / "score_head.pt", map_location="cpu",
                           weights_only=True)
        self.score_head.load_state_dict(state)
        self.score_head.eval()

    def score(self, prompt: str, response: str) -> float:
        return self.score_many(prompt, [response])[0]

    def score_many(self, prompt: str, responses: list[str]) -> list[float]:
        # Batch size 1 per forward on purpose — matches training
        # (batch 1 x accum 32) and avoids padding/pooling interactions.
        scores: list[float] = []
        with torch.no_grad():
            for resp in responses:
                enc = self.tokenizer(
                    format_text(prompt, resp),
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.backbone(**enc).last_hidden_state
                idx = enc["attention_mask"][0].nonzero()[-1].item()
                pooled = hidden[0, idx].float().cpu()
                scores.append(self.score_head(pooled).item())
        return scores

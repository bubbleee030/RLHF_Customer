#!/usr/bin/env python3
"""Glue: actor + RM + CM -> gate-and-rank response. GPU/Docker only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import LocalActor
from gate_rank import Candidate, ScoreModel, Selection, select

RM_DIR = "reward_output/run_reward_byprompt_20260622_104203/epoch1"
CM_DIR = "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"


class GateRankPipeline:
    def __init__(self, actor_device: str = "cuda:0", scorer_device: str = "cuda:1",
                 n: int = 4, adapter_dir: str | None = None) -> None:
        self.actor = LocalActor(device=actor_device)
        if adapter_dir:  # PPO-Lag LoRA adapter (Stage A pilot)
            from peft import PeftModel
            self.actor.model = PeftModel.from_pretrained(
                self.actor.model, adapter_dir).eval()
        self.has_adapter = adapter_dir is not None
        self.cm = ScoreModel(CM_DIR, device=scorer_device)
        self.rm = ScoreModel(RM_DIR, device=scorer_device)
        self.n = n

    def actor_variant(self, use_ppo: bool):
        """Context manager: disable the LoRA adapter for the base variant."""
        import contextlib
        if self.has_adapter and not use_ppo:
            return self.actor.model.disable_adapter()
        return contextlib.nullcontext()

    def score_candidates(self, prompt: str, texts: list[str]) -> list[Candidate]:
        costs = self.cm.score_many(prompt, texts)
        rewards = self.rm.score_many(prompt, texts)
        return [Candidate(t, cost=c, reward=r)
                for t, c, r in zip(texts, costs, rewards)]

    def respond(self, prompt: str,
                history: list[tuple[str, str]] | None = None) -> Selection:
        texts = self.actor.generate(prompt, history=history, n=self.n)
        return select(self.score_candidates(prompt, texts))

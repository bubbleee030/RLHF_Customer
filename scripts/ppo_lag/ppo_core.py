#!/usr/bin/env python3
"""PPO-Lagrange algorithm core, ported from PKU-Alignment/safe-rlhf.

Sources (in the cloned repo at safe-rlhf/):
- gather_log_probabilities / masked_mean: safe_rlhf/utils.py
- kl_shaped_scores: PPOLagTrainer.add_kl_divergence_regularization
  (safe_rlhf/algorithms/ppo_lag/trainer.py:263-300)
- get_advantages_and_returns: safe_rlhf/trainers/rl_trainer.py:630-651
- actor_loss_fn: safe_rlhf/algorithms/ppo_lag/trainer.py:302-325
- critic_loss_fn: safe_rlhf/trainers/rl_trainer.py:653-669
- LagrangeMultiplier: safe_rlhf/algorithms/ppo_lag/trainer.py:57-66 + rl_step:317-326
"""
from __future__ import annotations

import math

import torch


def gather_log_probabilities(logits: torch.Tensor, labels: torch.Tensor,
                             chunk_size: int = 512) -> torch.Tensor:
    """Log-probability assigned to each label token.

    Mathematically identical to gathering from ``log_softmax`` -- for a target
    token t, ``log_softmax(x)[t] == -cross_entropy(x, t)`` -- but the fused
    cross-entropy never materialises the (batch, seq, vocab) log-probability
    tensor and does not retain it for backward.

    That matters at this model's vocab of 131072: one such tensor is ~1GB in
    fp32 for a 1900-token sequence, and the policy-prefixed variant (Run D)
    exhausted a 32GB V100 here even at batch size 1, where activation memory was
    not the constraint. Under no_grad the sequence axis is additionally chunked,
    which is exact because log_softmax is independent per position.
    """
    # NB: deliberately no logits.float() here. Upcasting the full
    # (batch, seq, vocab) tensor costs ~2GB by itself at vocab=131072 with a
    # policy-prefixed sequence -- that upcast, not log_softmax, was the residual
    # OOM. cross_entropy's fused kernel accumulates in fp32 internally, so half
    # precision inputs keep their accuracy without the materialised copy.
    # The .float() is applied to the RESULT, which is (batch, seq) -- a few
    # thousand values -- not to the logits. That preserves the original fp32
    # output dtype (downstream KL and ratio maths depend on it) while avoiding
    # the (batch, seq, vocab) fp32 copy.
    if not torch.is_grad_enabled() and logits.size(1) > chunk_size:
        parts = [
            -torch.nn.functional.cross_entropy(
                logits[:, index:index + chunk_size].transpose(1, 2),
                labels[:, index:index + chunk_size],
                reduction="none",
            ).float()
            for index in range(0, logits.size(1), chunk_size)
        ]
        return torch.cat(parts, dim=1)
    return -torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), labels, reduction="none").float()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum()


def kl_shaped_scores(
    reward: torch.Tensor,          # (B,)
    cost: torch.Tensor,            # (B,)
    log_probs: torch.Tensor,       # (B, L)
    ref_log_probs: torch.Tensor,   # (B, L)
    sequence_mask: torch.Tensor,   # (B, L) bool
    kl_coeff: float,
    clip_range_score: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token shaped rewards/costs: -kl_coeff*KL everywhere (+kl for cost),
    with the scalar RM/CM score added at each sequence's last valid token."""
    end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # (B,)
    kl_divergence_estimate = log_probs - ref_log_probs
    kl_penalty_rewards = -kl_coeff * kl_divergence_estimate
    rewards = torch.scatter_add(
        kl_penalty_rewards, dim=-1,
        index=end_index.unsqueeze(dim=-1),
        src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
    )
    costs = torch.scatter_add(
        -kl_penalty_rewards, dim=-1,
        index=end_index.unsqueeze(dim=-1),
        src=cost.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
    )
    return (
        torch.clamp(rewards, min=-clip_range_score, max=clip_range_score),
        torch.clamp(costs, min=-clip_range_score, max=clip_range_score),
    )


def get_advantages_and_returns(
    values: torch.Tensor,         # (B, L)
    rewards: torch.Tensor,        # (B, L)
    sequence_mask: torch.Tensor,  # (B, L)
    start: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over tokens [start, L)."""
    last_gae_lambda = 0.0
    advantages_reversed = []
    values = values * sequence_mask
    rewards = rewards * sequence_mask
    length = rewards.size(-1)
    for t in reversed(range(start, length)):
        next_values = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last_gae_lambda = delta + gamma * gae_lambda * last_gae_lambda
        advantages_reversed.append(last_gae_lambda)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, start:]
    return advantages.detach(), returns


def actor_loss_fn(
    log_probs: torch.Tensor,          # (B, T)
    old_log_probs: torch.Tensor,      # (B, T)
    reward_advantages: torch.Tensor,  # (B, T)
    cost_advantages: torch.Tensor,    # (B, T)
    mask: torch.Tensor,               # (B, T) bool
    multiplier: float,
    clip_range_ratio: float,
) -> torch.Tensor:
    advantages = (reward_advantages - multiplier * cost_advantages) / (1.0 + multiplier)
    ratios = torch.exp(log_probs - old_log_probs)
    surrogate1 = advantages * ratios
    surrogate2 = advantages * torch.clamp(
        ratios, 1.0 - clip_range_ratio, 1.0 + clip_range_ratio,
    )
    surrogate = torch.minimum(surrogate1, surrogate2)
    return -masked_mean(surrogate, mask)


def critic_loss_fn(
    values: torch.Tensor,      # (B, T)
    old_values: torch.Tensor,  # (B, T)
    returns: torch.Tensor,     # (B, T)
    mask: torch.Tensor,        # (B, T) bool
    clip_range_value: float,
) -> torch.Tensor:
    values_clipped = torch.clamp(
        values, old_values - clip_range_value, old_values + clip_range_value,
    )
    vf_loss1 = torch.square(values - returns)
    vf_loss2 = torch.square(values_clipped - returns)
    return 0.5 * masked_mean(torch.maximum(vf_loss1, vf_loss2), mask)


class LagrangeMultiplier:
    """log-space lambda with SGD, updated on the windowed mean episode cost:
    loss = -(E[cost] - threshold) * exp(log_lambda)."""

    def __init__(self, lambda_init: float = 1.0, lambda_lr: float = 0.1,
                 lambda_max: float | None = 5.0, threshold: float = 0.0) -> None:
        self.log_lambda = torch.nn.Parameter(torch.tensor(math.log(lambda_init)))
        self.log_lambda_max = math.log(lambda_max) if lambda_max else None
        self.optimizer = torch.optim.SGD([self.log_lambda], lr=lambda_lr)
        self.threshold = threshold

    @property
    def value(self) -> float:
        return self.log_lambda.exp().item()

    def update(self, episode_cost: float) -> float:
        lambda_loss = -(episode_cost - self.threshold) * self.log_lambda.exp()
        self.optimizer.zero_grad()
        lambda_loss.backward()
        self.optimizer.step()
        if self.log_lambda_max is not None:
            with torch.no_grad():
                self.log_lambda.clamp_(max=self.log_lambda_max)
        return self.value

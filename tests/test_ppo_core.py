#!/usr/bin/env python3
"""Unit tests for the PPO-Lagrange algorithm core (ported from PKU safe-rlhf).

Run: python3 tests/test_ppo_core.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "ppo_lag"))

from ppo_core import (
    LagrangeMultiplier,
    actor_loss_fn,
    critic_loss_fn,
    gather_log_probabilities,
    get_advantages_and_returns,
    kl_shaped_scores,
    masked_mean,
)


def test_gather_log_probabilities():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])  # B=1, L=2, V=2
    labels = torch.tensor([[0, 0]])
    out = gather_log_probabilities(logits, labels)
    expect0 = math.log(math.exp(2) / (math.exp(2) + 1))
    expect1 = math.log(1 / (math.exp(2) + 1))
    assert abs(out[0, 0].item() - expect0) < 1e-5
    assert abs(out[0, 1].item() - expect1) < 1e-5


def test_masked_mean():
    x = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert abs(masked_mean(x, mask).item() - 1.5) < 1e-6


def test_kl_shaped_scores_zero_kl():
    # log_probs == ref_log_probs -> KL penalty 0 everywhere; the scalar
    # reward lands on the LAST valid token, cost likewise.
    B, L = 1, 4
    lp = torch.zeros(B, L)
    mask = torch.tensor([[True, True, True, False]])
    rewards, costs = kl_shaped_scores(
        torch.tensor([3.0]), torch.tensor([-2.0]), lp, lp, mask,
        kl_coeff=0.01, clip_range_score=50.0,
    )
    assert rewards[0, 2].item() == 3.0 and rewards[0, 0].item() == 0.0
    assert costs[0, 2].item() == -2.0 and costs[0, 1].item() == 0.0


def test_kl_shaped_scores_kl_sign():
    # Actor more confident than ref (log_probs > ref) -> positive KL ->
    # NEGATIVE reward penalty and POSITIVE cost penalty on non-end tokens.
    lp = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[-1.0, -1.0]])
    mask = torch.tensor([[True, True]])
    rewards, costs = kl_shaped_scores(
        torch.tensor([0.0]), torch.tensor([0.0]), lp, ref, mask,
        kl_coeff=0.5, clip_range_score=50.0,
    )
    assert rewards[0, 0].item() == -0.5
    assert costs[0, 0].item() == 0.5


def test_gae_hand_computed():
    # Single token of interest: start=1, L=2, values known.
    # delta_1 = r_1 + gamma*0 (no next) - v_1 ; advantage_1 = delta_1
    values = torch.tensor([[0.5, 1.0]])
    rewards = torch.tensor([[0.0, 2.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    adv, ret = get_advantages_and_returns(values, rewards, mask, start=1,
                                          gamma=1.0, gae_lambda=0.95)
    assert adv.shape == (1, 1)
    assert abs(adv[0, 0].item() - (2.0 - 1.0)) < 1e-6
    assert abs(ret[0, 0].item() - (1.0 + 1.0)) < 1e-6  # adv + values[start:]


def test_actor_loss_direction():
    # multiplier=0 -> pure reward advantages; positive advantage with
    # ratio 1 gives loss = -advantage.
    lp = torch.tensor([[0.0]])
    mask = torch.tensor([[True]])
    adv_r = torch.tensor([[2.0]])
    adv_c = torch.tensor([[100.0]])  # must be ignored at multiplier=0
    loss = actor_loss_fn(lp, lp, adv_r, adv_c, mask,
                         multiplier=0.0, clip_range_ratio=0.2)
    assert abs(loss.item() + 2.0) < 1e-6
    # multiplier=1 -> (A_r - A_c)/2
    loss2 = actor_loss_fn(lp, lp, adv_r, adv_c, mask,
                          multiplier=1.0, clip_range_ratio=0.2)
    assert abs(loss2.item() + (2.0 - 100.0) / 2.0) < 1e-5


def test_actor_loss_ratio_clip():
    # Huge ratio with positive advantage must be clipped at 1+0.2.
    old = torch.tensor([[0.0]])
    new = torch.tensor([[5.0]])  # ratio e^5 >> 1.2
    mask = torch.tensor([[True]])
    adv = torch.tensor([[1.0]])
    zero = torch.zeros_like(adv)
    loss = actor_loss_fn(new, old, adv, zero, mask,
                         multiplier=0.0, clip_range_ratio=0.2)
    assert abs(loss.item() + 1.2) < 1e-5


def test_critic_loss_clip():
    # values far above old_values get clamped to old+clip before the MSE;
    # loss takes the max of clipped/unclipped errors.
    values = torch.tensor([[10.0]])
    old_values = torch.tensor([[0.0]])
    returns = torch.tensor([[0.0]])
    mask = torch.tensor([[True]])
    loss = critic_loss_fn(values, old_values, returns, mask, clip_range_value=5.0)
    assert abs(loss.item() - 0.5 * 100.0) < 1e-5  # unclipped err (100) > clipped (25)


def test_lambda_update():
    lag = LagrangeMultiplier(lambda_init=1.0, lambda_lr=0.1, lambda_max=5.0, threshold=0.0)
    v0 = lag.value
    lag.update(episode_cost=2.0)   # cost above threshold -> lambda rises
    assert lag.value > v0
    lag2 = LagrangeMultiplier(lambda_init=1.0, lambda_lr=0.1, lambda_max=5.0, threshold=0.0)
    lag2.update(episode_cost=-2.0)  # cost below threshold -> lambda falls
    assert lag2.value < 1.0
    lag3 = LagrangeMultiplier(lambda_init=4.9, lambda_lr=10.0, lambda_max=5.0, threshold=0.0)
    lag3.update(episode_cost=100.0)
    assert lag3.value <= 5.0 + 1e-6  # clamped at lambda_max


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} tests passed")

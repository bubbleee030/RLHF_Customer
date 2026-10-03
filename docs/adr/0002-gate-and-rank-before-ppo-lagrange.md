# 0002 — Integrate RM+CM via gate-and-rank first; defer PPO-Lagrange to HPC

Date: 2026-07-07
Status: accepted

## Context

Stage 2 of the Safe RLHF pipeline is complete: a deployable Reward Model
(`reward_output/run_reward_byprompt_20260622_104203/epoch1`, honest by-prompt
accuracy ~0.60) and Cost Model
(`cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/`, sign
accuracy ~0.94+). The project goal is to use both models to align the TAIWAN
AI RAP chatbot. The reference method (PKU safe-rlhf PPO-Lagrange, surveyed in
`docs/reports/07-07/safe_rlhf_ppo_lag_survey_20260707.md`) keeps six model
engines resident and trains three of them; PKU ran a 7B actor on 8×A800-80GB.
Local hardware is 2×V100-32GB (64 GB total, no bf16). The production chatbot
is a 24B model served remotely; the local stand-in actor is
Ministral-3-3B-Instruct.

## Decision

Integrate in two stages:

- **Stage B (now, local):** gate-and-rank at inference time — the frozen
  actor samples N candidates, the CM gates (score > 0 rejected), the RM ranks
  survivors. No model weights change.
- **Stage A (later, HPC):** PPO-Lagrange per the reference implementation,
  gated on (i) an HPC allocation with sufficient GPU memory and (ii) an
  improved RM (more annotation, per the 06-22 report conclusion).

## Alternatives considered

- **PPO-Lag now on V100s:** infeasible full-finetune (≈36 GB weights alone
  for six 3B engines, plus ~48 GB Adam state per trained model); a
  LoRA/offload variant would diverge from the reference and still fight fp16
  instability.
- **PPO-Lag now with current RM:** PPO optimizes hard against the reward
  signal; a ~0.60-accuracy RM invites reward hacking. In gate-and-rank a weak
  RM can only mis-rank among already-safe candidates — it degrades gracefully.

## Consequences

- Both models are exercised end-to-end at serving time now, producing demo
  and evaluation evidence for the mentor report without RL infrastructure.
- Serving cost: every response requires N generations + 2N scoring passes.
- The CM's calibrated sign (3-term loss) is load-bearing in both stages:
  gate boundary now, constraint threshold 0.0 later.
- CONTEXT.md's "foundation for later PPO-Lagrange fine-tuning" remains true;
  this ADR only sequences the path there.

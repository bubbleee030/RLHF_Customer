# Prompt Gate Demo Toggle Design

**Date:** 2026-08-03  
**Status:** Approved for immediate execution

## Objective

Add a Gradio checkbox that temporarily disables prompt-level gating for the PPO
side of the research demo. The checkbox defaults to enabled whenever a frozen
prompt-gate model is loaded. Base generation remains ungated for comparison.

## Behavior

- Checkbox label: `啟用 Prompt Gate（僅 PPO）`.
- With `PROMPT_GATE_MODEL` loaded, the checkbox defaults to `True` and is
  interactive.
- Without a gate model, it defaults to `False`, is non-interactive, and the card
  reports `未載入`.
- PPO + enabled + high risk returns `REFUSAL_FALLBACK` before generation.
- PPO + disabled always generates, while retaining the risk score and reporting
  `已手動關閉`.
- Base always generates and reports `參考模型未套用`.
- The setting applies only to the current comparison click; it does not modify
  the model, frozen threshold, artifact, environment, or another user session.

## Architecture

`scripts/serve/prompt_safety.py` exposes a pure `resolve_prompt_gate` function
that converts the classifier decision plus variant/availability/toggle state
into a displayable effective decision. `demo_customer_ppo.py` consumes that
decision, keeps generation/refusal logic unchanged, and binds the checkbox as a
third input to `compare`.

## Safety and Disclosure

The research-only warning remains visible. Turning the gate off does not disable
the output CM score or degeneration checks. The PPO card explicitly reports the
manual-off state so an ungated answer cannot be mistaken for a gated result.

## Testing

Table-driven pure-function tests cover enabled block, disabled bypass, base
bypass, and missing-model behavior. A focused demo copy test checks the Gradio
checkbox binding and safe default. The full CPU suite and Python compilation
must pass before handoff.

## Self-review

No placeholders or ambiguous state precedence remain. State precedence is:
missing model → base reference → PPO manually disabled → PPO enabled decision.
This workspace is not a Git repository, so no design commit is possible.

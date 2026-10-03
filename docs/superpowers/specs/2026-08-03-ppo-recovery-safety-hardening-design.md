# PPO Recovery and Safety Hardening Design

**Date:** 2026-08-03  
**Status:** Approved direction; written-spec review required before implementation

## Objective

Produce a replacement customer-8B research candidate only if it satisfies both
of these conditions on prompt-disjoint validation data:

1. higher paired helpfulness-RM score than the downloaded base model; and
2. no manually observed safety regression, with zero detected output
   degeneration.

The existing repaired RM, 30-step PPO adapter, evaluation, reports, and demo are
immutable baselines for this recovery cycle. No existing artifact is deleted or
overwritten.

## Diagnosis to Verify

Four falsifiable hypotheses drive the work, in this order:

1. **Unpaired sampling confounds the reported RM regression.** The current
   evaluator seeds once, then samples base and PPO sequentially. Prediction:
   resetting an identical seed for each prompt/variant materially changes the
   measured RM delta while preserving deterministic reruns.
2. **Thirty PPO updates over-optimize the small training objective.** Prediction:
   a ten-update candidate at actor LR `1e-5` and KL coefficient `0.4` has lower
   validation KL and a better paired RM delta than the 30-step adapter.
3. **The cost model has customer-domain false negatives.** This is already
   evidenced by manually unsafe responses receiving negative cost. Prediction:
   hyperparameter-only PPO changes cannot reliably remove those failures.
4. **A prompt-level safety signal complements the response CM.** Prediction: a
   classifier trained on prompt-disjoint existing safety labels catches the
   reviewed high-risk prompts that the CM misses, without blocking the majority
   of normal customer-support prompts.

Each hypothesis is tested independently. A failed hypothesis does not trigger a
stack of simultaneous tuning changes.

## Architecture

The recovery pipeline has three independent units:

### 1. Paired evaluator

The evaluator derives a stable seed from `(global_seed, track, prompt_index,
sample_index)` and resets that seed before generating each variant. Base,
current 30-step PPO, and candidate PPO therefore receive the same stochastic
stream for a prompt/sample. The output records the derived seed so any pair can
be replayed.

Selection uses only the existing 37-prompt RM validation split plus a
prompt-disjoint safety validation subset. The sealed 36-prompt RM test remains
untouched during candidate selection. After selection is frozen, one final
sealed evaluation may be run once for reporting.

### 2. Conservative actor candidate

A fresh run starts from `model/costomer_model`; it does not continue from the
30-step LoRA. The isolated change is:

- 10 PPO updates;
- actor LR `1e-5`;
- critic LR `5e-5`;
- KL coefficient `0.4`;
- max new tokens 224;
- prompt batch 4 / microbatch 1;
- non-reentrant gradient checkpointing;
- final actor adapter only.

Temporary checkpoints and evaluation candidates live under a dedicated
`/dev/shm/reward_model_ppo_recovery_20260803/` directory. The directory path is
validated before use. Nothing is deleted automatically. Only an accepted actor
is copied to persistent workspace storage, and the copy is refused unless at
least 150 MiB remains afterward.

### 3. Prompt safety gate

A lightweight character-ngram classifier uses existing prompt-level safety
labels. Training rows are deduplicated by exact normalized prompt, and every
prompt used by RM validation, RM sealed test, PPO evaluation, or manual review
is excluded from classifier training.

At inference:

1. high-risk prompt probability at or above a threshold returns a fixed,
   concise Traditional Chinese refusal with a safe alternative;
2. otherwise the actor generates candidates;
3. the existing CM remains a response-level proxy and the repaired RM ranks
   candidates that pass it;
4. the UI labels both signals as proxies and retains the research-only warning.

The threshold is selected on safety validation to prioritize recall, subject to
an explicit normal-support false-positive ceiling. If the available labels
cannot meet both constraints, the classifier is not presented as a fix and the
candidate fails the safety gate.

## Data Boundaries

- Actor training: `datasets/reward/cs_within_train.jsonl` only.
- Actor selection: `datasets/reward/cs_within_validation.jsonl` only.
- Final RM confirmation: `datasets/reward/cs_within_test.jsonl`, at most once
  after all choices are frozen.
- Prompt-safety training: existing labeled cost prompts after excluding all
  evaluation/manual-review prompt strings.
- Prompt-safety validation: prompt-disjoint labeled examples plus the frozen
  manually reviewed deadline set.

No example used to choose LR, KL, update count, or safety threshold contributes
to the final sealed RM claim.

## Acceptance Gates

The new actor/pipeline is accepted only if all gates pass:

1. paired mean RM delta versus base is strictly positive on RM validation;
2. prompt-level win rate versus base is greater than 50%;
3. mean sequence KL is below 2.0 and no individual sampled KL exceeds 15;
4. degeneration rate is 0%;
5. prompt safety recall is at least 90% on labeled high-risk validation;
6. false-positive rate is at most 10% on normal customer-support validation;
7. manual review finds no case where the accepted pipeline is less safe than
   base across the frozen review prompts.

If either candidate actor fails the helpfulness gates, base remains the actor.
If the prompt classifier fails its recall/false-positive gates, it is not
enabled. If manual safety fails, no new checkpoint is promoted regardless of
automated scores.

## Failure Handling and Storage

- Non-finite metrics, generation failures, or OOM abort the candidate run.
- The current 30-step adapter remains available as a comparison baseline.
- Persistent disk is checked before every copy; the trainer never saves critics
  for the recovery candidate.
- The pre-existing demo container is not stopped unless local training cannot
  proceed after the already-proven checkpointing configuration.
- No HPC transfer is attempted without a configured host and authentication.

## Testing and Evidence

Implementation follows red-green-refactor:

- deterministic seed derivation and base/PPO seed equality;
- prompt exclusion and prompt-disjoint safety split;
- threshold-selection recall/FPR constraints;
- actor-only checkpoint saving and storage guard;
- end-to-end replay showing identical metrics on two evaluator reruns;
- GPU smoke, ten-update training log, paired validation comparison, degeneration
  analysis, and a recorded manual-review table.

The bilingual mentor reports and demo are updated only after the acceptance
decision. Rejected candidates remain clearly labelled and never replace the
known baseline paths.

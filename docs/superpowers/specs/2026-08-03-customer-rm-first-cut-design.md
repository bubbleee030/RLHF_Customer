# Customer-Model RM First-Cut Design

## Goal

Finish the interrupted Claude Code experiment that tests whether rebuilding the
Reward Model (RM) data with only customer-model candidates improves held-out
within-model ranking accuracy above the current 0.444 baseline, then use the
repaired RM for a deadline-bounded PPO-Lagrange LoRA pass on the downloaded 8B
customer actor.

## Fixed inputs

- `datasets/reward/cs_candidates_shard0.jsonl`
- `datasets/reward/cs_candidates_shard1.jsonl`
- 376 unique prompts, six 8B customer-model candidates per prompt, and 2,256
  non-empty visible answers.
- The `[THINK]` block is never sent to a judge or RM. Only `candidate.answer`
  is used.

## Annotation design

Each judging call ranks all six candidates, so one call yields all 15 pairwise
decisions for a prompt. Candidate presentation order is independently shuffled
on every run. The script supports multiple judge model IDs and multiple runs
per model; the first cut uses two shuffled runs of the already validated
`NVIDIA-Nemotron-3-Super-120B-A12B` judge. This reproduces the measured
swap/presentation-consistency filter: retain a pair only when every successful
run chooses the same winner. Disagreement means a near-tie and the pair is
dropped, not resolved by majority vote.

Calls are checkpointed by `(prompt fingerprint, judge model, run)` and are
safe to resume. A prompt contributes pairs only when all configured runs parse
successfully. Raw judge text is retained in the checkpoint for auditability;
API credentials are read from the environment and never written to output.

## Dataset construction

Prompts, not pairs, are deterministically split before emitting training,
validation, and sealed-test JSONL. Validation selects checkpoints and drives
any mitigation; the sealed test is read once after the configuration is fixed.
This prevents the same prompt or candidate from appearing in multiple splits.
Each retained pair uses the existing trainer schema:
`input`, `chosen`, `rejected`, plus provenance fields for prompt fingerprint,
candidate indices, temperatures, judge models, and agreement count.

The build emits:

- `datasets/reward/cs_within_train.jsonl`
- `datasets/reward/cs_within_validation.jsonl`
- `datasets/reward/cs_within_test.jsonl`
- `results/reward/cs_annotation/summary.json`

## Training design

The RM backbone is Ministral-3-3B-Instruct-2512. LoRA (`r=16`, `alpha=32`) is
applied to attention and MLP projections while the scalar score head remains
trainable. Training uses Bradley-Terry loss, prompt-disjoint evaluation, and
selects the best epoch by validation loss. Early stopping halts after two
consecutive epochs without a loss improvement. Only the best adapter is kept
on persistent disk; disposable intermediates may use container tmpfs. Every
saved checkpoint must be loadable by the existing evaluation path.

## 8B actor pass and demo

The actor pass reuses the validated PPO-Lagrange core and mitigated pilot
recipe: LoRA r=16, `kl_coeff` 0.2–0.3, actor LR 2e-5, KL early-stop at 15, and
at most 30 updates for the deadline run. The customer actor uses its working
`[INST]...[/INST]` format; `[THINK]` remains part of actor generation but is
stripped before RM/CM scoring.

The demo mirrors the earlier comparison: the same prompt shows base and PPO
responses side by side with RM score, CM score, KL/divergence, and degeneration
flags. The tuned model is presented as successful only when held-out output is
coherent, safety does not regress, and detected degeneration is 0%.

## Deadline and storage constraints

- Persistent disk starts with only ~1.1 GB free. No existing checkpoint,
  including `model/costomer_model/global_step120/`, may be removed.
- Keep only the selected RM adapter, final PPO adapters, logs, evaluations, and
  reports. Use container tmpfs for disposable intermediates.
- Run locally on 2×V100 first. The 4×V100 VM or NCHC Nano4 is a fallback when
  connection details become available.

## Success criteria

1. Annotation is resumable and no prompt leaks across train/eval.
2. Every emitted pair has unanimous judge direction across configured runs.
3. The LoRA checkpoint reloads and produces scalar scores.
4. Held-out within-model accuracy is compared with the 0.444 baseline.
5. Above 0.60 is deployment-useful; a smaller but statistically meaningful
   gain validates scaling to 2,000–3,000 real prompts; chance-level performance
   is reported as evidence that 376 prompts remain insufficient.
6. RM ≥0.55 plus sane spot checks permits 30 actor updates. RM 0.50–0.55
   permits only a 10-step feasibility pass. Below 0.50 triggers a
   train/validation-only diagnosis and time-bounded repair.
7. PPO output must have 0% detected degeneration, KL below 15, coherent
   Traditional Chinese spot checks, and safety no worse than the base actor.

## Failure handling

- Malformed or incomplete candidate rows fail validation before API calls.
- Judge HTTP/parse failures retry three times and remain resumable.
- If retained train or eval pairs are empty, dataset construction fails loudly.
- GPU or dependency failures do not trigger deletion or cleanup. The user's
  standing rule remains: ask before removing checkpoints or other large files.

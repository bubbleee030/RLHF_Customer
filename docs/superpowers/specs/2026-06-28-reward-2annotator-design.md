# Spec: 2-annotator helpfulness Reward Model experiment

**Date:** 2026-06-28
**Status:** approved
**Builds on:** `docs/reports/06-22/reward_model_training_report_20260622.md`, `docs/adr/0001-...`

## Question

Against the **human-gold** held-out metric (baseline = 0.60 by-prompt pairwise accuracy),
does a second annotator (user2 = `NVIDIA-Nemotron-3-Super-120B-A12B`, helpfulness-only,
de-biased order, temp 0) **improve** the helpfulness RM —
(a) used as a *filter* on the human labels (agreed-only), or
(b) does *naive merging* (union) hurt?

## Critical context (measured)

Human (user1) vs LLM (user2) agreement is **near chance**: pairwise 1049/2238 = **46.9%**,
full-order match 8.8% (33/373). Disagreements skew toward safety-sensitive prompts
(near-reversals). So user2 is **not** a reliable consensus signal; the honest expected
result is "little/no improvement," which is itself worth recording.

## Data (verified)

- Sources joined on **prompt text** (record_ids differ between the checkpoint and the
  backup): `argilla/backups/latest/records.fixed.json` (R1–R4 text + human ranking) ×
  `argilla/llm_judge_user2_checkpoint.jsonl` (human_ranking + llm_ranking).
- Join is clean: **373/373 matched, 0 missing, 0 human_ranking mismatches**.
- Split = same seed-42 by-prompt split as baseline → **336 train / 37 eval prompts**.
  The eval set is the **existing** `datasets/reward/reward_eval_byprompt.jsonl`
  (220 human pairs / 37 prompts) — confirmed identical → apples-to-apples vs 0.60.

## Train sets (from the 336 train prompts; exact counts)

| variant | rule | pairs | file |
|---|---|---|---|
| (baseline, exists) | all human pairs | 2016 | `reward_train_byprompt.jsonl` |
| **agreed** | human pairs user2 orders the same way | **941** | `reward_train_agreed.jsonl` |
| **union** | all human pairs + all LLM pairs (6+6/prompt) | **4032** | `reward_train_union.jsonl` |

## Components

- **New:** `scripts/reward/prepare_reward_data_2ann.py` — join, build `reward_train_agreed.jsonl`
  + `reward_train_union.jsonl` into `datasets/reward/`. Asserts join integrity
  (abort if human-ranking mismatches > 0). Skips empty/identical responses.
- **Reused unchanged:** `train_reward_model.py`, `run_reward_docker.sh` (via `TRAIN=`/`EVAL=`
  overrides), `eval_reward.py`.

## Training

Same config as baseline: Ministral-3-3B-Instruct-2512, BT loss, max_length 4096,
batch 1 × accum 32, lr 1e-5, 3 epochs, last-token, load-in-half + device_map (2×V100),
per-epoch checkpoint, `--eval-dataset-path datasets/reward/reward_eval_byprompt.jsonl`.
Two runs (sequential, both GPUs each): `reward_output/run_reward_{agreed,union}_<ts>/`.
Launched via a detached watcher (setsid) so they survive SSH drop.

## Eval & report

Best-epoch human-holdout accuracy: **baseline 0.60 vs agreed vs union**. Verify each
best checkpoint reloads/reproduces (`eval_reward.py` → `results/reward/`). Write a new
report `docs/reports/06-28/reward_2annotator_experiment_20260628.md` (+ 繁中 if requested).

## Testing

Smoke-test tiny subsets of each train file (1 epoch, max_len 256) before the full runs,
as in the baseline. Verify pair counts (941 / 4032) and that the eval file is unchanged.

## Out of scope

Agreement-weighted training; treating LLM as gold; PPO-Lagrange wiring.

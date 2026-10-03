# Reward model trained with the in-repo custom trainer, not PKU safe-rlhf's DeepSpeed pipeline

**Status:** accepted

We train the helpfulness Reward Model by reusing the repo's proven custom trainer (`scripts/train_cost_model_v2.py`, the same code that trained the cost model), adapted to a pure Bradley-Terry loss — rather than running PKU-Alignment/safe-rlhf's literal `safe_rlhf.values.reward` DeepSpeed pipeline. We still *utilize* PKU's method: the score-model head/class is their code verbatim (`shortcuts/safe_rlhf_score_model___init__.py`) and the loss is their reward formula `-log σ(R(y_w) − R(y_l))`.

## Why

- The working Docker image `cost-model-trainer:v2` ships **transformers 5.7 / torch 2.4**, which is required to load the backbone **Ministral-3-3B-Instruct-2512** (a Dec-2025 Mistral3 architecture). PKU safe-rlhf pins old transformers (~4.38) that **cannot load this model**, so running their pipeline literally would force heavy patching of their code — the opposite of "don't modify their code."
- The real safe-rlhf repo is **not present** in the working image (the on-disk `Dockerfile` that clones it is stale; the rebuilt v2 image dropped it).
- The custom trainer already implements PKU's exact loss terms and already runs this backbone end-to-end on the 2×V100 host, with the May-18 "backbone not saved" bug already fixed.

## Documented deviations / additive changes (per request #2)

These are additions to *our* trainer, made in a **separate** `scripts/reward/train_reward_model.py` (the cost trainer is left untouched to avoid regression):

1. **Reward loss** — pure Bradley-Terry (term 1 only); drop the two ±safety-sign terms used by the cost model.
2. **Reward dataset** — reads `(input, y_w, y_l)` preference pairs (no safety signs); `y_w` mapped to the higher score.
3. **`--eval-dataset-path`** — accept a separate eval file and skip the internal split, enabling a by-prompt holdout.
4. **Per-epoch backbone checkpointing** — save each epoch so the best-by-eval epoch can be kept.

Input text format is kept identical to the cost model: `"User: {prompt}\nAssistant: {response}"`.

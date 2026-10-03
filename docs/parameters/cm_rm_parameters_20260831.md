# Cost Model and Reward Model — complete parameter inventory

**Date:** 2026-08-31
**Scope:** every tunable input to the CM and RM training pipelines, as the code actually
reads them today. Produced ahead of the Airflow migration so the DAG's parameter list can
be derived from verified values rather than from memory.

Tables in this document are generated directly from the trainers' `argparse` definitions by
AST parsing, and the historical run columns are read from each run's `arguments.json`. They
are not hand-transcribed.

---

## How to read these tables

A parameter's effective value passes through **three** layers, and they disagree more often
than you would expect:

| Layer | Where it lives | Notes |
|---|---|---|
| 1. `argparse` default | `parser.add_argument(..., default=)` | What you get running the trainer directly with no flags. Several are stale — see Finding 2. |
| 2. Wrapper value | `scripts/retrain_ministral3b_docker.sh`, `scripts/reward/run_reward_docker.sh` | What production actually passes. Overrides layer 1 for almost every parameter. |
| 3. Historical run | `<run_dir>/arguments.json` | What a specific completed run used. The source of truth for reproducing a published number. |

**The Airflow DAG should take its defaults from layer 2 or 3, never layer 1.**

---

## Findings

### Finding 1 — the RM has two divergent production configurations (needs a decision)

There is no single "the RM config". Two real, completed RM runs used materially different
hyperparameters, and the design spec documents only the second one:

| Parameter | June run — helpfulness RM | August run — customer CS RM | `run_reward_docker.sh` default |
|---|---|---|---|
| `max_length` | 4096 | 576 | **4096** |
| `batch_size` × `grad_accum` | 1 × 32 | 2 × 16 | **1 × 32** |
| `learning_rate` | 1e-5 | 5e-5 | **1e-5** |
| `weight_decay` | 1e-6 | 0.01 | **1e-6** |
| `epochs` | 3 | 5 | **3** |
| `regularization` | 0.001 | 0.01 | **0.001** |
| `lora_r` | *(absent — full FT)* | 16 | **0 (full FT)** |
| `early_stopping_patience` | *(absent)* | 2 | **0 (disabled)** |

Evidence: `reward_output/run_reward_byprompt_20260622_104203/arguments.json` and
`reward_output/run_reward_cs_within_20260803/arguments.json`.

**Why this matters.** Section 6 of
`docs/superpowers/specs/2026-08-25-airflow-cm-rm-training-pipeline-design.md` presents the
August column as the `rm_train` defaults. But running `bash scripts/reward/run_reward_docker.sh`
with no environment overrides reproduces the **June** column instead. Anyone building the DAG
from that spec section would get defaults matching neither the wrapper nor a reproducible run.

The June run's `arguments.json` has no `lora_r` key at all — that run predates the LoRA flags
being added to the RM trainer, so it is full fine-tuning by absence rather than by `lora_r=0`.

**Decision needed:** which of the two configurations is `rm_train`'s default? They train
different models for different purposes, so the honest answer may be that the DAG needs a
`preset` parameter (`helpfulness` | `customer_cs`) rather than one set of numbers.

### Finding 2 — the CM trainer's `argparse` defaults are stale

`scripts/train_cost_model_v2.py` still carries defaults from its DeBERTa era. They are
harmless today only because the wrapper overrides all of them:

| Flag | `argparse` default | What production uses |
|---|---|---|
| `--model-name-or-path` | `microsoft/deberta-v3-large` | `mistralai/Ministral-3-3B-Instruct-2512` |
| `--max-length` | `512` | `4096` |
| `--batch-size` | `4` | `1` |
| `--gradient-accumulation-steps` | `8` | `32` |

Anyone invoking the trainer directly — which is exactly what an Airflow task doing its own
`docker run` would do — silently gets a DeBERTa-shaped configuration. These defaults should be
corrected to the production values before the DAG bypasses the shell wrapper.

The RM trainer does not have this problem; its defaults already name Ministral-3-3B.

### Finding 3 — no dead parameters

Every one of the 62 flags is referenced in its trainer's body. Nothing needs deleting.

### Finding 4 — naming asymmetries between the two trainers

The two trainers grew separately and their flags do not line up. This matters because the
Airflow migration is meant to give them a shared parameter schema:

| Concern | Cost Model | Reward Model |
|---|---|---|
| Best-checkpoint selection | `--save-best` (str, comma-separated metrics) | `--save-best-only` (flag) |
| Per-epoch checkpoints | *(none — always saves)* | `--save-each-epoch` (`--/--no-`) |
| Half precision | `--fp16` **and** `--bf16` | `--bf16` only |
| Loss shape | `--loss-type {sequence-wise,token-wise}` | *(none — Bradley-Terry only)* |
| Score normalization | `--normalize-score-during-training`, `--normalizer-momentum` | *(none)* |
| Early stopping | *(none)* | `--early-stopping-patience`, `--early-stopping-min-delta` |
| Eval predictions | `--save-eval-predictions` | *(none)* |

Seven capabilities exist on exactly one side. The DAG cannot expose one schema for both
without either adding the missing flags or marking them model-specific.

---

## Cost Model — `scripts/train_cost_model_v2.py`

32 command-line parameters. Entry point in production is
`scripts/retrain_ministral3b_docker.sh` → `scripts/trainer.py` → `train_cost_model_v2.main()`.

| Flag | Type | argparse default | Wrapper sets | run 20260625 (best) | Purpose |
|---|---|---|---|---|---|
| `--model-name-or-path` | str | `microsoft/deberta-v3-large` | `$MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | HuggingFace id or local path of the backbone to attach the score head to. |
| `--dataset-path` | path | **required** | `$DATASET` | `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl` | Training data (JSONL). Must be train-only when --eval-dataset-path is also given; the trainer does not subtract eval rows. |
| `--output-dir` | path | **required** | `$RUN_DIR` | `cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best` | Run directory: checkpoints, arguments.json, training_log.json, run_manifest.json. |
| `--max-length` | int | `512` | `$MAX_LENGTH` | `4096` | Tokenizer truncation length. Dominates VRAM; the single biggest cost knob. |
| `--batch-size` | int | `4` | `$BATCH_SIZE` | `1` | Micro-batch per optimizer step. Effective batch = batch_size x gradient_accumulation_steps. |
| `--gradient-accumulation-steps` | int | `8` | `$GRAD_ACCUM` | `32` | Steps accumulated before an optimizer update. |
| `--learning-rate` | float | `1e-05` | `$LR` | `1e-05` | Peak LR after warmup. |
| `--weight-decay` | float | `1e-06` | `1e-6` | `1e-06` | AdamW weight decay. |
| `--epochs` | int | `3` | `$EPOCHS` | `3` | Full passes over the training set. |
| `--warmup-ratio` | float | `0.1` | `0.1` | `0.1` | Fraction of total steps spent warming the LR up from 0. |
| `--regularization` | float | `0.001` | `0.001` | `0.001` | L2 regularization weight on scores |
| `--eval-split-ratio` | float | `0.1` | `0.1` | `0.1` | Fraction of data for evaluation |
| `--eval-dataset-path` | path | `None` | — | — | Explicit eval split produced by scripts/cost/split_cost_dataset.py. When given, --eval-split-ratio is ignored and no in-trainer split occurs. Preferred path; the internal split is retained only for backward compatibility with historical runs. |
| `--log-steps` | int | `10` | `10` | `10` | Optimizer steps between log lines. |
| `--seed` | int | `42` | `$SEED` | `42` | Seed for shuffling, split, and init. |
| `--pooling` | str | `mean` | `last-token` | `last-token` | Pooling method for the sequence Choices: ['mean', 'cls', 'last-token']. |
| `--loss-type` | str | `sequence-wise` | `sequence-wise` | `sequence-wise` | PKU cost loss variant to optimize Choices: ['sequence-wise', 'token-wise']. |
| `--concat-forward` | flag | `False` | — | `False` | Run safer/unsafer pairs in a single concatenated forward pass |
| `--normalize-score-during-training` | flag | `False` | — | `False` | Apply running z-score normalization to sequence/token scores during training |
| `--normalizer-momentum` | float | `0.9` | — | `0.9` | EMA momentum for running score normalization |
| `--gradient-checkpointing` | flag | `False` | — | `False` | Trade compute for VRAM by recomputing activations in the backward pass. |
| `--fp16` | flag | `False` | — | `False` | Use autocast FP16 (best for encoder models like DeBERTa) |
| `--bf16` | flag | `False` | — | `False` | Use BF16 (autocast or native, best for newer decoder models like Gemma/Qwen) |
| `--load-in-half` | flag | `False` | `set` | `True` | Load backbone in native half precision (FP16 or BF16) |
| `--device-map` | flag | `False` | `set` | `True` | Use device_map='auto' to shard large models across all GPUs (skips DataParallel) |
| `--adafactor` | flag | `False` | — | `False` | Use Adafactor optimizer (memory-efficient; recommended for large decoder models) |
| `--save-eval-predictions` | flag | `False` | `set` | `True` | Save per-sample eval predictions to eval_predictions.jsonl in output-dir |
| `--save-backbone` | flag (--x/--no-x) | `True` | `set` | `True` | Persist the fine-tuned backbone via save_pretrained (default: on). REQUIRED for full fine-tunes — without it the trained model is lost and only score_head.pt remains. Pass --no-save-backbone only if you deliberately want to discard backbone weights. |
| `--save-best` | str | `` | — | `pairwise,loss` | Comma-separated metric(s) to keep the BEST per-epoch checkpoint for, e.g. 'pairwise,loss'. Each is saved to <output_dir>/best-<metric>/ whenever it improves. Metrics: pairwise (eval_accuracy, higher=better), loss (eval_loss, lower), sign (eval_accuracy_sign, higher). When set, the final last-epoch backbone is NOT written to the run-dir root (keep best, not latest). |
| `--lora-r` | int | `0` | `$LORA_R` | — | LoRA rank. 0 (default) means full fine-tuning. |
| `--lora-alpha` | int | `32` | `$LORA_ALPHA` | — | LoRA scaling factor; effective scale is alpha/r. |
| `--lora-dropout` | float | `0.05` | `$LORA_DROPOUT` | — | Dropout on the LoRA path. |

---

## Reward Model — `scripts/reward/train_reward_model.py`

30 command-line parameters. Entry point in production is
`scripts/reward/run_reward_docker.sh`. The two run columns are the divergent configurations
described in Finding 1.

| Flag | Type | argparse default | Wrapper sets | June (helpfulness) | Aug (customer CS) | Purpose |
|---|---|---|---|---|---|---|
| `--model-name-or-path` | str | `mistralai/Ministral-3-3B-Instruct-2512` | `$MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | `mistralai/Ministral-3-3B-Instruct-2512` | HuggingFace id or local path of the backbone to attach the score head to. |
| `--dataset-path` | path | **required** | `$TRAIN` | `datasets/reward/reward_train_byprompt.jsonl` | `datasets/reward/cs_within_train.jsonl` | Training data (JSONL). Must be train-only when --eval-dataset-path is also given; the trainer does not subtract eval rows. |
| `--eval-dataset-path` | path | `None` | `$EVAL` | `datasets/reward/reward_eval_byprompt.jsonl` | `datasets/reward/cs_within_validation.jsonl` | Separate eval file (e.g. by-prompt holdout). If set, skips the internal split. |
| `--output-dir` | path | **required** | `$RUN_DIR` | `reward_output/run_reward_byprompt_20260622_104203` | `reward_output/run_reward_cs_within_20260803` | Run directory: checkpoints, arguments.json, training_log.json, run_manifest.json. |
| `--max-length` | int | `4096` | `$MAX_LENGTH` | `4096` | `576` | Tokenizer truncation length. Dominates VRAM; the single biggest cost knob. |
| `--batch-size` | int | `1` | `$BATCH_SIZE` | `1` | `2` | Micro-batch per optimizer step. Effective batch = batch_size x gradient_accumulation_steps. |
| `--gradient-accumulation-steps` | int | `32` | `$GRAD_ACCUM` | `32` | `16` | Steps accumulated before an optimizer update. |
| `--learning-rate` | float | `1e-05` | `$LR` | `1e-05` | `5e-05` | Peak LR after warmup. |
| `--weight-decay` | float | `1e-06` | `$WEIGHT_DECAY` | `1e-06` | `0.01` | AdamW weight decay. |
| `--epochs` | int | `3` | `$EPOCHS` | `3` | `5` | Full passes over the training set. |
| `--warmup-ratio` | float | `0.1` | `0.1` | `0.1` | `0.1` | Fraction of total steps spent warming the LR up from 0. |
| `--regularization` | float | `0.001` | `$REGULARIZATION` | `0.001` | `0.01` | L2 penalty on raw score magnitude, keeping outputs from drifting. |
| `--eval-split-ratio` | float | `0.1` | — | `0.1` | `0.1` | Used only when --eval-dataset-path is not given. |
| `--log-steps` | int | `10` | `10` | `10` | `10` | Optimizer steps between log lines. |
| `--seed` | int | `42` | `$SEED` | `42` | `42` | Seed for shuffling, split, and init. |
| `--pooling` | str | `last-token` | `last-token` | `last-token` | `last-token` | How token states collapse to one score. last-token is what production uses. Choices: ['mean', 'cls', 'last-token']. |
| `--concat-forward` | flag | `False` | — | `False` | `False` | Run the chosen and rejected halves of a pair in one forward pass. |
| `--gradient-checkpointing` | flag | `False` | — | `False` | `False` | Trade compute for VRAM by recomputing activations in the backward pass. |
| `--bf16` | flag | `False` | — | `False` | `False` | bfloat16 autocast. Preferred over fp16 where supported. |
| `--load-in-half` | flag | `False` | `set` | `True` | `True` | Load backbone weights already in half precision, halving load-time RAM. |
| `--device-map` | flag | `False` | `set` | `True` | `True` | Let accelerate shard the model across visible GPUs (device_map=auto). |
| `--adafactor` | flag | `False` | — | `False` | `False` | Use Adafactor instead of AdamW to cut optimizer state memory. |
| `--save-each-epoch` | flag (--x/--no-x) | `True` | `set` | `True` | `False` | Save a backbone checkpoint after every epoch (epochN/ subdirs). |
| `--save-backbone` | flag (--x/--no-x) | `True` | `set` | `True` | `False` | Persist full backbone weights. --no-save-backbone keeps only the adapter + score head. |
| `--save-best-only` | flag | `False` | — | — | `True` | Persist only best/ when validation loss improves. |
| `--lora-r` | int | `0` | `$LORA_R` | — | `16` | Enable LoRA with this rank; zero preserves full fine-tuning. |
| `--lora-alpha` | int | `32` | `$LORA_ALPHA` | — | `32` | LoRA scaling factor; effective scale is alpha/r. |
| `--lora-dropout` | float | `0.05` | `$LORA_DROPOUT` | — | `0.05` | Dropout on the LoRA path. |
| `--early-stopping-patience` | int | `0` | `$EARLY_STOPPING_PATIENCE` | — | `2` | Stop after this many epochs without eval-loss improvement; zero disables. |
| `--early-stopping-min-delta` | float | `0.0` | — | — | `0.0` | Minimum improvement that counts as progress for early stopping. |

---

## Shell wrapper environment variables

These are the knobs an operator sets today. The Airflow DAG will replace this layer, so its
parameter list should cover all of them.

### `scripts/retrain_ministral3b_docker.sh` (Cost Model) — 15 variables

| Variable | Default | Effect |
|---|---|---|
| `HF_CACHE` | `${HOME}/.cache/huggingface` | Host HF cache bind-mounted into the container. |
| `IMAGE` | `cost-model-trainer:v2` | Docker image. |
| `MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | → `--model-name-or-path`. |
| `DATASET` | `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl` | → `--dataset-path`. |
| `MAX_LENGTH` | `4096` | → `--max-length`. |
| `BATCH_SIZE` | `1` | → `--batch-size`. |
| `GRAD_ACCUM` | `32` | → `--gradient-accumulation-steps`. |
| `LR` | `1e-5` | → `--learning-rate`. |
| `EPOCHS` | `3` | → `--epochs`. |
| `SEED` | `42` | → `--seed`. |
| `LORA_R` | `0` | → `--lora-r`. Above 0 also triggers `pip install peft` in the container. |
| `LORA_ALPHA` | `32` | → `--lora-alpha`. |
| `LORA_DROPOUT` | `0.05` | → `--lora-dropout`. |
| `RUN_DIR` | `cost_output/run_ministral_3b_instruct_<timestamp>` | Output directory. |
| `EXTRA_FLAGS` | *(empty)* | Raw flags appended to the Python invocation. |

Note: `weight_decay`, `regularization`, `warmup_ratio`, `eval_split_ratio`, `log_steps`,
`pooling`, and `loss_type` are **hardcoded** in this wrapper and cannot be changed without
editing the file or going through `EXTRA_FLAGS`. The DAG should expose them properly.

### `scripts/reward/run_reward_docker.sh` (Reward Model) — 20 variables

| Variable | Default | Effect |
|---|---|---|
| `HF_CACHE` | `${HOME}/.cache/huggingface` | Host HF cache bind mount. |
| `IMAGE` | `cost-model-trainer:v2` | Docker image. |
| `MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | → `--model-name-or-path`. |
| `SPLIT` | `byprompt` | Selects the dataset pair; `bypair` for the CM-comparable run. |
| `TRAIN` | `datasets/reward/reward_train_${SPLIT}.jsonl` | → `--dataset-path`. |
| `EVAL` | `datasets/reward/reward_eval_${SPLIT}.jsonl` | → `--eval-dataset-path`. |
| `MAX_LENGTH` | `4096` | → `--max-length`. |
| `BATCH_SIZE` | `1` | → `--batch-size`. |
| `GRAD_ACCUM` | `32` | → `--gradient-accumulation-steps`. |
| `LR` | `1e-5` | → `--learning-rate`. |
| `EPOCHS` | `3` | → `--epochs`. |
| `SEED` | `42` | → `--seed`. |
| `WEIGHT_DECAY` | `1e-6` | → `--weight-decay`. |
| `REGULARIZATION` | `0.001` | → `--regularization`. |
| `LORA_R` | `0` | → `--lora-r`; above 0 also triggers `pip install peft`. |
| `LORA_ALPHA` | `32` | → `--lora-alpha`. |
| `LORA_DROPOUT` | `0.05` | → `--lora-dropout`. |
| `EARLY_STOPPING_PATIENCE` | `0` | → `--early-stopping-patience`. |
| `RUN_DIR` | `reward_output/run_reward_${SPLIT}_<timestamp>` | Output directory. |
| `EXTRA_FLAGS` | *(empty)* | Raw flags appended to the invocation. |

`SMOKE=1` overrides `TRAIN`, `EVAL`, `MAX_LENGTH=256`, `GRAD_ACCUM=2`, `EPOCHS=1` and
redirects `RUN_DIR`, giving a minutes-long wiring test instead of a multi-hour run. The CM
wrapper has no equivalent — worth adding, since the DAG needs a cheap validation path on both
sides.

`HOST_UID` / `HOST_GID` (both defaulting to `1000`) are used only to `chown` the run directory
back to the host user after the container exits.

---

## Recommended Airflow parameter defaults

Derived from layer 2 (wrapper) values, which are the ones that actually reproduce today's runs:

- **`cm_train`** — take every value from the CM wrapper table above. Additionally promote the
  seven currently-hardcoded flags (`weight_decay`, `regularization`, `warmup_ratio`,
  `eval_split_ratio`, `log_steps`, `pooling`, `loss_type`) to DAG parameters.
- **`rm_train`** — do **not** pick one of the two configurations arbitrarily. Expose a
  `preset` parameter with `helpfulness` (June column) and `customer_cs` (August column) as the
  two values, each expanding to a full parameter set. This preserves both reproducible runs.
- Before the DAG invokes the trainers directly rather than through the shell wrappers, fix the
  four stale CM `argparse` defaults listed in Finding 2.

## Reproducing this document

```bash
python3 scripts/lib/param_inventory.py
```

Regenerates both tables from the current source. Rerun it after any change to either
trainer's `argparse` block so the inventory cannot drift from the code.

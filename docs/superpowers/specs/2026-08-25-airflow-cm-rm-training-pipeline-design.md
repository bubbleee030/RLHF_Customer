# Airflow migration — Cost Model and Reward Model training pipelines

Date: 2026-08-25
Status: design approved, implementation not started
Scope: CM and RM training pipelines. PPO-Lagrange is documented as the downstream
consumer but is not migrated in this round.

## Goal

Turn the CM and RM training pipelines into parameterized Airflow DAGs, so that a run
is defined by its parameters rather than by which shell variables somebody remembered
to export. Secondary goal: make the data-handling defects that are currently invisible
(CM split leakage, ad-hoc RM split) into explicit, inspectable DAG tasks.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Deliverable this round | Design + mentor report only | Airflow is not installed; vm2 disk is at 100% (463 MB free). Nothing that writes a checkpoint can run today. |
| Upstream boundary | Include candidate generation | Lets the whole RM pipeline be re-run against a new prompt pool or actor by changing params. |
| CM split fix | `split_strategy` parameter, default `by_prompt` | Reproduces old runs (`by_pair`) and produces honest new ones. Directly demonstrates the parameterization goal. |
| Execution model | SSHOperator → existing shell wrappers | Scripts are already env-var driven; preserves the exact `cost-model-trainer:v2` image whose numerical parity was proven on vm3. |
| DAG topology | Two DAGs (`cm_train`, `rm_train`), `schedule=None` | Different inputs and cadences. These are manually-triggered experiments, not cron jobs. |
| Export format | `external` / `integrated` / `both`, default `both` | Different consumers need different layouts (see §4). |
| Training mode | `lora` / `full_ft` | Free on RM; requires porting LoRA into the CM trainer (see §5). |
| Report language | Bilingual EN + zh-TW | Matches the 08-03 mentor report convention. |

---

## 1. Current pipeline, as actually run

Verified against the code on 2026-08-25, not against documentation.

### 1.1 Cost Model

| # | Stage | Entry point | Behaviour |
|---|---|---|---|
| 0 | Sources | `datasets/cost/cost_model_dataset_{pointwise,pairwise}.jsonl` | — |
| 1 | Build pairs | `scripts/rebuild_cost_dataset.py` | Harmful prompts: every safe × unsafe response pair. Benign: safe × safe. SHA-256 triple dedupe. Drops `safer_sign >= unsafer_sign` contradictions. Output `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl`, **1,461 pairs**. |
| 2 | Split | **inside** `scripts/train_cost_model_v2.py:769-776` | Pair-level. `random.Random(seed).shuffle(indices)`, `n_eval = int(n_total * eval_split_ratio)`. 90/10 → 1,315 train / 146 eval. |
| 3 | Train | `scripts/retrain_ministral3b_docker.sh` → `scripts/trainer.py` → `scripts/train_cost_model_v2.py` | Base `mistralai/Ministral-3-3B-Instruct-2512`, **full fine-tune**, 3,429,009,409 trainable params. `max_length` 4096, batch 1 × grad-accum 32, lr 1e-5, weight decay 1e-6, score regularization 0.001, 3 epochs, warmup 0.1, AdamW + cosine, last-token pooling, sequence-wise loss, `load_in_half`, `device_map=auto` across 2×V100. ~1h56m wall time. Format `User: {prompt}\nAssistant: {answer}`. |
| 4 | Select | `--save-best pairwise,loss` | `best-pairwise/` (epoch 1, eval loss 0.745, pair acc 70.55%, sign acc 97.60%); `best-loss/` (epoch 3, eval loss 0.678, pair acc 65.07%, sign acc 99.32%). **`best-loss` is what PPO and the demos use.** |
| 5 | Export | `Ministral3ForCostModel` integrated package | 8-file runtime folder → `s3://cpft/ft/` and MinIO (`rclone` remote `minio_server`). |

Loss is the PKU safe-RLHF 3-part objective (pairwise safe > unsafe cost ranking, safe
cost pushed negative, unsafe cost pushed positive) plus L2 score regularization.
Decision rule: `cost < 0` means safe.

### 1.2 Reward Model

| # | Stage | Entry point | Behaviour |
|---|---|---|---|
| 0 | Prompt pool | `test/prompts/prompt_pool_20260323_144720.jsonl` | 376 prompts. |
| 1 | Generate | `scripts/reward/gen_cs_candidates.py` | 376 prompts × 6 temperatures (0.3/0.5/0.7/0.9/1.0/1.1), top-p 0.95, `max_new_tokens` 512, format `[INST]{prompt}[/INST]`, FP16, sharded across 2 GPUs via `SHARD`/`NUM_SHARDS`. **2,256 generations.** `[THINK]` blocks stripped before any scoring or judging. |
| 2 | Annotate | `scripts/reward/annotate_cs_candidates.py` | Judge `NVIDIA-Nemotron-3-Super-120B-A12B` via NCHC Medusa. Each call ranks all 6 candidates (15 pairwise decisions). Run **twice, independently shuffled**, per prompt. A pair is kept only if both runs agree. 369/376 prompts usable (7 judge-refused). **4,358 of 5,535 possible pairs retained (78.74%).** |
| 3 | Split | ad hoc | Prompt-disjoint, seed 42 → 296 train / 37 validation / 36 sealed test prompts (3,473 / 463 / 422 pairs). Zero prompt leakage across splits. The second-stage 73 → 37/36 division was done manually. |
| 4 | Train | `scripts/reward/run_reward_docker.sh` → `scripts/reward/train_reward_model.py` | **As run for the August customer-CS RM** (the wrapper's own defaults differ — see §6): LoRA r=16 / alpha=32 / dropout=0.05 on a frozen backbone, pure Bradley-Terry loss, `max_length` 576, batch 2 × grad-accum 16, lr 5e-5, weight decay 0.01, regularization 0.01, 5 epochs with early-stopping patience 2. Best = epoch 1 (val loss 0.5362, val acc 0.7451). |
| 5 | Sealed test | `scripts/reward/eval_reward.py` | **0.7915** pairwise accuracy on the 422-pair sealed set, vs the previous RM's 0.5474 on the identical set (+24.4 pp, paired-exact p ≈ 5.1e-14). |
| 6 | Export | `Ministral3ForRewardModel` integrated package | → S3 and MinIO. |

### 1.3 Downstream consumer (not migrated this round)

`scripts/ppo_lag/train_ppo_lag.py` consumes `RM_DIR` and `CM_DIR` as **frozen scorers**.
PPO trains only the actor LoRA plus two separate value critics
(`reward_critic_adapter`, `cost_critic_adapter`). Those critics are value functions and
must never be labelled or uploaded as "the RM" or "the CM". PPO produces no new RM or CM.

---

## 2. Defects confirmed in the code

These are findings from reading the source on 2026-08-25, and each one shapes the design.

1. **CM's split is not a step.** It happens inside the trainer at
   `train_cost_model_v2.py:769-776`, shuffling *pair* indices. Because the same prompt's
   responses appear in many pairs, eval pairs share responses with training pairs — the
   leakage that makes CM eval accuracy optimistic. Lifting the split out is a real code
   change, not just DAG wiring. It is also *why* the leakage went unnoticed for so long:
   there was no artifact to inspect.

2. **Stale default paths.** `retrain_ministral3b_docker.sh` defaults
   `DATASET=datasets/cost_dataset_for_safe_rlhf_clean.jsonl`, which **does not exist**;
   the real file is `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl`.
   `rebuild_cost_dataset.py` has the same stale defaults (`./datasets/...` instead of
   `./datasets/cost/...`). Every successful run must have passed an override. These
   defaults must not be copied into the DAG.

3. **CM has no LoRA support.** `train_cost_model_v2.py` contains no PEFT code at all, and
   `retrain_ministral3b_docker.sh` never installs `peft`. See §5.

4. **RM's 3-way split is undocumented in code.** The 296/37/36 prompt division exists only
   as output files; there is no deterministic function that reproduces it.

---

## 3. DAG architecture

Two DAGs, `schedule=None`, triggered manually with parameters.

```
cm_train                              rm_train
────────                              ────────
preflight                             preflight
   |                                     |
build_pairs                           generate_candidates   [pool: gpu]
   |                                     |
split            <- NEW task          annotate               [API, retry-heavy, resumable]
   |                                     |
train            [pool: gpu]          build_consensus_pairs
   |                                     |
evaluate                              split                  <- formalized 3-way
   |                                     |
select_checkpoint                     train                  [pool: gpu]
   |                                     |
sealed_test      <- hard gate         evaluate_validation
   |                                     |
export_package                        reload_smoke_test
   |                                     |
publish -> Dataset(cm_integrated)     sealed_test            <- hard gate
                                         |
                                      export_package
                                         |
                                      publish -> Dataset(rm_integrated)
```

Every task is an `SSHOperator` invoking an existing script with environment variables
injected from `params`. No training code is rewritten to accommodate Airflow.

**GPU serialization.** All GPU tasks share an Airflow Pool (`gpu_vm2`, 2 slots). This
prevents two runs from colliding on the same V100s — the failure mode that caused the
step-12 OOM on 2026-08-03, when an unrelated demo container held 8.5 GB.

**Sealed test as a hard gate.** `sealed_test` runs once, after `select_checkpoint`, and
its result never feeds back into selection. This encodes the discipline already followed
manually, and makes accidental tuning on the sealed set structurally impossible.

**The `split` task's output contract is binding on `train`.** When the CM trainer is given
both `--dataset-path` and `--eval-dataset-path`, it trusts that the former is train-only —
it does not subtract eval rows. The DAG must therefore pass the split task's
`train.jsonl`, never the full dataset, alongside its `eval.jsonl`. Passing the full dataset
plus an eval split would train on the eval rows and silently recreate the exact leakage the
split task exists to remove. The `train` task asserts that its two inputs come from the same
split directory and that their row counts sum to the manifest's `n_total`.

**`reload_smoke_test` (RM only).** Loads the just-saved checkpoint from disk in a fresh
process and re-scores a fixed pair set. This catches LoRA reload bugs — the class of bug
that produced the `score.bias UNEXPECTED` confusion — before the checkpoint is published.

**Chaining.** Airflow Datasets link the DAGs: publishing a new RM or CM can optionally
trigger a downstream PPO DAG. PPO must also be runnable against *pinned* RM/CM paths
without retraining either, so the dataset trigger is opt-in, never mandatory.

---

## 4. Export step

`export_format`: `external` | `integrated` | `both` (default `both`).

These are not alternatives; they serve different consumers.

| | **external** (native run output) | **integrated** (derived repackage) |
|---|---|---|
| RM layout | `adapter_model.safetensors` (99 MB LoRA) + `score_head.pt` + tokenizer. Requires the base model at load time. | Merged, 6.4 GiB, `score.weight`/`score.bias` FP32 baked into safetensors, class `Ministral3ForRewardModel`. |
| CM layout | `model.safetensors` (6.86 GB full FT) + `score_head.pt` + tokenizer. | 6.4 GiB, class `Ministral3ForCostModel`. |
| Loader | `ScoreModel` in `scripts/serve/gate_rank.py:81-115` — `AutoModel` + external `nn.Linear` head + last-token pooling. | `AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)`. |
| Consumers | **PPO, gate-and-rank, demos, `eval_before_after.py`** | Mentor handoff, `s3://cpft/ft/`, MinIO |

**Format-dependent parity tolerance.** The gate differs by format because the work differs:

- **CM integrated** repackages the score head only; no weight surgery. Gate:
  `max_abs <= 1e-5`, ranking 20/20, sign 40/40. Achieved on vm3: **0.0**.
- **RM integrated** must merge the LoRA into the BF16 backbone. BF16 refolding noise is
  unavoidable and irreducible. Gate: `max_abs <= 0.125`, ranking 20/20. Achieved on vm3
  GPU: **0.0834**. The same check on CPU failed at **0.1336** — therefore **the RM parity
  gate must run on GPU**, and a CPU result is not evidence of a broken merge.

**Guard.** If `export_format=integrated` alone, the DAG emits a warning that no
PPO-consumable artifact was produced. It does not fail the run — a delivery-only run is
legitimate — but it will not silently leave PPO with nothing to point at.

`publish_targets` controls upload: `[]`, `["s3"]`, `["s3","minio"]`. MinIO uses
`rclone copy` (never `rclone sync`, which would delete unrelated objects under the shared
`intern/models/reward_model/` prefix), preceded by `--dry-run` and followed by
`rclone check`.

---

## 5. Training mode

`training_mode`: `lora` | `full_ft`.

| | RM (`train_reward_model.py`) | CM (`train_cost_model_v2.py`) |
|---|---|---|
| LoRA | Supported today (`--lora-r`, PEFT at L104-106 and L224, save path at L346) | **Not supported.** No PEFT code anywhere in the file. |
| Full FT | Supported (`lora_r=0`, the default) | The only mode that exists |
| Docker wrapper | `run_reward_docker.sh` conditionally runs `pip install peft` when `LORA_R > 0` | `retrain_ministral3b_docker.sh` never installs `peft` |

The parameter is free on the RM side. On the CM side it requires:

1. Porting the LoRA path from `train_reward_model.py` into `train_cost_model_v2.py`
   (the pattern is directly reusable: `LoraConfig` / `get_peft_model`, plus the
   `save_backbone or lora_r > 0` save branch).
2. Adding the conditional `pip install peft` to `retrain_ministral3b_docker.sh`.

**Why it is worth doing.** CM full fine-tune is a 3.43B-parameter run producing a
**6.86 GB** checkpoint in ~1h56m on 2×V100. A LoRA CM adapter would be roughly 100 MB.
On a machine that has spent this entire project between 60 MiB and 1 GiB of free disk,
that is the difference between retaining several CM runs and retaining none.

**Caveat that must appear in the mentor report.** CM's headline 99.32% sign accuracy is a
full fine-tune result. A LoRA CM is a different model and requires re-validation before it
replaces anything in PPO. This is an experiment the DAG makes cheap, not a drop-in swap.

---

## 6. Parameterization

The two DAGs share a parameter *schema* but not parameter *values*. The block below shows
`cm_train`'s defaults; `rm_train` differs where the pipelines genuinely differ, and those
differences are listed immediately after. Neither DAG should inherit the other's numbers.

```python
# cm_train defaults
params = {
    # data
    "pointwise_file": "datasets/cost/cost_model_dataset_pointwise.jsonl",
    "pairwise_file":  "datasets/cost/cost_model_dataset_pairwise.jsonl",
    "split_strategy": "by_prompt",      # by_prompt | by_pair
    "split_seed": 42,
    "eval_ratio": 0.1,                  # 2-way train/eval split

    # training
    "training_mode": "full_ft",         # lora | full_ft
    "max_length": 4096,
    "batch_size": 1,
    "grad_accum": 32,
    "lr": 1e-5,
    "epochs": 3,
    "weight_decay": 1e-6,
    "regularization": 0.001,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,   # used when training_mode=lora
    "early_stopping_patience": 0,

    # selection and output
    "save_best": "pairwise,loss",
    "select_metric": "loss",            # which best-* becomes THE published checkpoint
    "export_format": "both",            # external | integrated | both
    "publish_targets": [],              # [] | ["s3"] | ["s3","minio"]

    # execution
    "target_host": "vm2",
    "gpu_pool": "gpu_vm2",
    "image": "cost-model-trainer:v2",
    "smoke": False,
}
```

**`rm_train` has two real configurations, not one.** *(Corrected 2026-08-31 — this
section previously presented the August column below as the sole `rm_train` default, which
was misleading. See `docs/parameters/cm_rm_parameters_20260831.md` Finding 1.)*

Two completed RM runs used materially different hyperparameters, and
`scripts/reward/run_reward_docker.sh` run with no environment overrides reproduces the
**June** column — not the August one:

| Param | June run (helpfulness RM) | August run (customer CS RM) | Wrapper default |
|---|---|---|---|
| `max_length` | 4096 | 576 | 4096 |
| `batch_size` x `grad_accum` | 1 x 32 | 2 x 16 | 1 x 32 |
| `lr` | 1e-5 | 5e-5 | 1e-5 |
| `weight_decay` | 1e-6 | 0.01 | 1e-6 |
| `epochs` | 3 | 5 | 3 |
| `regularization` | 0.001 | 0.01 | 0.001 |
| `lora_r` | *(absent -- full FT)* | 16 | 0 (full FT) |
| `early_stopping_patience` | *(absent)* | 2 | 0 (disabled) |

Evidence: `reward_output/run_reward_byprompt_20260622_104203/arguments.json` and
`reward_output/run_reward_cs_within_20260803/arguments.json`.

**Therefore `rm_train` must expose a `preset` parameter** (`helpfulness` | `customer_cs`),
each expanding to a full parameter set, rather than hardcoding either column. Picking one
arbitrarily would silently make the other historical run unreproducible.

**The August (customer CS) column, contrasted against CM.** These are not stylistic
variations; each reflects a real property of that RM run and must not be silently
unified with CM's values.

| Param | CM | RM | Why they differ |
|---|---|---|---|
| `max_length` | 4096 | 576 | RM scores single prompt+response pairs; CM must fit longer safety-relevant content. |
| `batch_size` × `grad_accum` | 1 × 32 | 2 × 16 | Same effective batch of 32; RM's shorter sequences allow a larger micro-batch. |
| `lr` | 1e-5 | 5e-5 | RM trains a LoRA adapter over a frozen backbone; CM full-fine-tunes 3.43B params. |
| `epochs` | 3 | 5 | RM relies on early stopping to pick the epoch; CM runs a fixed schedule. |
| `early_stopping_patience` | 0 | 2 | As above. |
| `weight_decay` | 1e-6 | 0.01 | Full FT vs LoRA regularization regimes. |
| `regularization` | 0.001 | 0.01 | Score-magnitude L2, tuned per model. |
| `training_mode` | `full_ft` | `lora` | RM was designed as LoRA; CM as full FT. |
| `split_strategy` | `by_prompt` (was `by_pair`) | `by_prompt` (already) | Only CM changes behaviour here. |
| `eval_ratio` | `0.1` (2-way) | 3-way: 296/37/36 prompts | RM additionally holds out a sealed test set. |

`rm_train` also carries generation and annotation parameters that CM has no analogue for:
`prompt_pool`, `n_candidates` (6), `temps` (0.3–1.1), `max_new_tokens` (512),
`judge_model`, `runs_per_prompt` (2), `judge_workers` (4), `judge_timeout` (240s).

`smoke=True` reuses the `SMOKE=1` path already present in `run_reward_docker.sh`, so DAG
wiring can be validated in minutes without occupying a V100 for two hours.

**XCom discipline.** XCom carries only URIs, checksums, counts, and metrics. Never models,
never datasets. Artifacts live on disk or object storage; XCom passes the path plus its
SHA-256.

**Run manifest.** Every run writes `run_manifest.json` containing: source file hashes
(this workspace has no functioning git repository, so file hashes replace a commit SHA),
Docker image digest, base model revision, dataset checksums, split manifest, resolved
dtype, GPU count and model, seed, fully resolved parameter set, selected checkpoint and
the reason it was selected.

---

## 7. Failure handling

**Retries by failure class, not one global setting.**

| Task class | Retries | Rationale |
|---|---|---|
| `annotate` | 10, exponential backoff | Flaky external API. The 2026-08-18 run already required an adaptive 1200 → 2400 → 4800 reasoning-token fallback when Nemotron exhausted its budget. |
| GPU training | **0** | A 2-hour job that hit OOM should not silently restart into the same wall. Fail loudly, let a human look. |
| `preflight`, `split`, `export` | 1 | Cheap and usually deterministic. |

**Resumability where it is expensive.** `generate_candidates` and `annotate` key completed
work on `(prompt, variant, seed, config-hash)` and skip it — the supervisor pattern proven
in the 2026-08-18 policy evaluation, where a 5-hour run survived two recoverable failures.

**Preflight as a real task.** Checks free disk against a threshold, GPU free memory, Docker
image digest match, source file hashes against the manifest, and `NCHC_API_KEY` presence.
Given the disk situation, this is the highest-value task in either DAG: it converts a
two-hour failure into a two-second one.

**Encoded gotchas.** The `cost-model-trainer:v2` entrypoint behaviour (`ENTRYPOINT` is
`/bin/bash` with no `CMD`, so appending a command yields `/bin/bash bash` and the second
token is executed as a script) is handled once in the shared `run_stage()` helper, so it
cannot be rediscovered a fourth time. Correct form: pass `--entrypoint /bin/bash` explicitly.

**Immutability.** Task writes go to run-scoped directories. Source datasets and existing
checkpoints are read-only. A gate failure preserves its evidence and stops; tolerances are
never auto-loosened, and source checkpoints are never modified to make a gate pass.

---

## 8. Pre-migration code changes required

These are prerequisites, not part of the DAG itself. Ordered by dependency.

| # | Change | Files | Blocking? |
|---|---|---|---|
| 1 | Lift the CM train/eval split out of the trainer into a standalone, deterministic step with `by_prompt` and `by_pair` strategies | `scripts/train_cost_model_v2.py`, new `scripts/cost/split_cost_dataset.py` | Yes — blocks `split_strategy` |
| 2 | Formalize the RM 3-way split as a deterministic function with a written manifest | new `scripts/reward/split_reward_dataset.py` | Yes — blocks reproducible RM runs |
| 3 | Fix stale default paths | `scripts/retrain_ministral3b_docker.sh`, `scripts/rebuild_cost_dataset.py` | Yes — silent breakage otherwise |
| 4 | Port LoRA support into the CM trainer + conditional `pip install peft` | `scripts/train_cost_model_v2.py`, `scripts/retrain_ministral3b_docker.sh` | Only for `training_mode=lora` on CM |
| 5 | Extract a shared `run_stage()` Docker helper encoding the entrypoint fix | new `scripts/lib/run_stage.sh` | No, but prevents recurrence |
| 6 | Emit `run_manifest.json` from every training entry point | both trainers | No, but required for reproducibility claims |

---

## 9. Testing

- **Unit**: split functions (determinism given a seed; `by_prompt` produces zero prompt
  overlap; `by_pair` reproduces the historical 1,315/146 division exactly), parameter
  resolution (params → env mapping), manifest serialization.
- **Integration**: `smoke=True` end-to-end run of each DAG on tiny fixtures, asserting task
  order, gate enforcement, and artifact layout.
- **Regression**: a `by_pair` + `full_ft` CM run must reproduce the existing checkpoint's
  metrics within tolerance. This is the proof that the migration preserved behaviour, and
  it is the only way to distinguish "the new number is honest" from "the migration broke
  something".
- **Gate tests**: RM integrated-export parity must be exercised on GPU; a CPU-only run is
  expected to exceed tolerance and must not be treated as a failure of the merge.

---

## 10. Out of scope

- Migrating PPO-Lagrange to Airflow. PPO is documented as the downstream consumer; its
  DAG follows once CM and RM are proven.
- Installing or operating Airflow. No Airflow is installed on any host today.
- Resolving vm2's disk exhaustion. It blocks execution but is an infrastructure task,
  not a pipeline-design one. Preflight surfaces it rather than solving it.
- Retraining CM to fix its known recall gap on generated text (14–55% against
  Nemotron-judged violations). That is a data problem, tracked separately.

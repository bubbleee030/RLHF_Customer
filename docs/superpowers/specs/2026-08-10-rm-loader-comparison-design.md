# Reward Model Loader Comparison Design

**Date:** 2026-08-10  
**Status:** Approved for immediate execution

## Objective

Provide one standalone diagnostic script that scores the same prompt-response
pair while independently switching:

1. the reward-model backbone load layout (`lora` or merged `full`), and
2. the input serialization (`training` or tokenizer `chat` template).

The diagnostic must use the reward model exactly as trained: a language-model
backbone, last-non-padding-token pooling, and the external
`score_head.pt` (`Linear(hidden_size, 1, bias=True)`). It must not use
`AutoModelForSequenceClassification`, whose Ministral score layer has no bias
and therefore reports our saved `score.bias` as unexpected.

## Interface

The script lives at `scripts/reward/test_rm_loading.py` and accepts:

- `--checkpoint`: LoRA adapter directory or merged full-backbone directory.
- `--load-mode {lora,full}`: explicit backbone loading path.
- `--template-mode {training,chat}`: explicit input serialization.
- `--base-model`: optional LoRA base-model override; otherwise read from
  `reward_model_config.json`.
- `--device`: explicit Torch device, defaulting to `cuda:0` when available.
- `--dtype {float16,bfloat16,float32}`.
- `--max-length`, `--prompt`, and `--response`.

The output reports the resolved loader inputs, backbone class/device/dtype,
score-head weight and bias shapes, saved bias value, rendered input, token
count, pooled token index, and scalar score.

## Loading Behavior

### LoRA

Load the tokenizer from the checkpoint, load the base with `AutoModel`, extract
its language-model backbone, apply the adapter through
`PeftModel.from_pretrained`, then call `merge_and_unload`. Require adapter
weights, `reward_model_config.json` (unless `--base-model` is supplied), and
`score_head.pt`.

### Full

Load the tokenizer and merged backbone directly from the checkpoint through
`AutoModel.from_pretrained`. Require `score_head.pt`. Reject a directory that
contains LoRA adapter weights so a mistaken mode cannot look successful.

### Shared score path

Infer `hidden_size`, instantiate `nn.Linear(hidden_size, 1, bias=True)` in
FP32, strictly load the external state dict, pool the last token selected by
the attention mask, and produce one scalar. The score head remains FP32.

## Input Serialization

- `training`: exact training string `User: {prompt}\nAssistant: {response}`.
- `chat`: `tokenizer.apply_chat_template(messages, tokenize=False)` using one
  user and one assistant message.

Serialization is independent of load mode, giving four directly comparable
experiments.

## Failure Policy

There is no try/except fallback between model architectures. Missing files,
missing base metadata, hidden-size mismatch, and wrong load layout fail loudly
with actionable messages. This prevents the mentor-style situation in which
loading technically succeeds after silently ignoring `score.bias`.

## Testing

CPU-only unit tests cover argument defaults, exact training and chat text,
load-layout validation, dtype resolution, external bias-preserving score-head
construction, and last-non-padding-token pooling. A GPU smoke command then
loads the real VM2 LoRA RM and prints a scalar.

## Self-review

The two experimental axes remain independent, and both paths share the same
tokenization-to-scalar implementation. `AutoModelForSequenceClassification`
is deliberately absent. No output artifact is overwritten, and the script is
read-only with respect to checkpoints.

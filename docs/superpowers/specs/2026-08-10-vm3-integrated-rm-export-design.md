# VM3 Exact-Score Integrated Reward Model Export Design

**Date:** 2026-08-10  
**Status:** Approved in conversation; awaiting written-spec review

## Objective

Export a new self-contained Hugging Face reward-model directory on VM3. The
export must integrate the trained `Linear(3072, 1, bias=True)` reward head into
the model safetensors so consumers do not separately load `score_head.pt`.
Scores must match the existing VM3 merged-backbone plus external-head loader,
not the original VM2 LoRA runtime.

## Source and Destination

Read-only sources on VM3:

- Merged backbone:
  `/home/ubuntu/merged_reward_model_ministral3b_20260807_114934`
- External head:
  `score_head.pt` inside that source directory
- Frozen parity pairs:
  `VALIDATION_REPORT.json` inside the source directory

New destination:

- `/home/ubuntu/merged_reward_model_ministral3b_20260810_integrated_exact`

The exporter must stop if the destination exists. It must not overwrite,
delete, rename, or modify the source merged model, original LoRA checkpoint,
S3 staging package, validation reports, Docker image, or caches.

## Architecture

Create `Ministral3ForRewardModel`, a Transformers-compatible custom model with:

1. the existing `Ministral3Model` backbone;
2. last-non-padding-token pooling using `attention_mask`;
3. `score = nn.Linear(config.get_text_config().hidden_size, 1, bias=True)`;
4. an FP32 score module and FP32 pooled hidden state;
5. `logits` returned through a standard sequence-classification output.

The class must preserve the source head's exact `weight` and `bias` tensors.
The saved model shards must contain both `score.weight` and `score.bias`.

## Hugging Face Loading Contract

The destination includes the custom modeling source and `auto_map` metadata so
the mentor can load it without manually constructing a head:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
)
```

The custom class must keep `score` in FP32 even when the backbone is loaded as
FP16. The tokenizer remains loadable through `AutoTokenizer.from_pretrained`.
Input serialization remains an independent caller choice: exact RM training
format or tokenizer chat template.

## Export Process

Run inside the already verified `cost-model-trainer:v2` image on VM3. Mount the
source model and exporter code read-only. Mount only a new, prevalidated parent
location as writable for the destination. Before serialization, verify:

- exact Docker image ID;
- VM3 hostname and four-GPU visibility;
- source directory and required files are real files/directories, not symlinks;
- source head shapes are `(1, 3072)` and `(1,)`;
- at least 15 GiB free disk remains available beyond the estimated export;
- destination does not exist.

Load the merged backbone and external head, construct the custom model, copy
the tensors strictly, register the auto class, save with safe serialization,
and save the tokenizer. Do not include `score_head.pt` in the destination.

## Validation Gates

Use the same VM3 GPU, dtype, prompt formatting, max length, and frozen inputs on
both sides.

1. Cold-load the destination through
   `AutoModelForSequenceClassification(..., trust_remote_code=True)`.
2. Confirm no missing or unexpected model keys.
3. Confirm the reloaded score head is FP32 and has bias enabled.
4. Confirm destination safetensors contain `score.weight` and `score.bias`.
5. Confirm destination contains no `score_head.pt`.
6. Compare the existing external-head loader with the integrated model on the
   frozen 20 pairs / 40 scores:
   - maximum absolute score difference `<= 1e-5`;
   - ranking agreement `20/20`.
7. Recheck the mentor pair in both formats:
   - training-format reference: `-0.95753115`;
   - chat-template reference: `-2.97840667`;
   - each within `1e-5` of the external-head loader run from the same process.

If any gate fails, preserve the failed destination for diagnosis under an
explicitly labelled non-delivery path; do not present or upload it as the final
model, and do not weaken tolerances after seeing results.

## Deliverables

- New integrated model directory on VM3.
- Export script and parity-validation script on VM3.
- Machine-readable validation report with source/destination paths, hashes,
  environment versions, per-score differences, maximum difference, ranking
  agreement, head dtype/shape/bias, and pass/fail status.
- Mentor loading example documenting `trust_remote_code=True` and the required
  input-format choice.

## Self-review

The reference implementation and score baseline are explicit. The destination
is new and non-overwriting. Exact-score preservation includes FP32 head and
bias, not only ranking parity. `trust_remote_code=True` is disclosed as part of
the loading contract. Every success claim maps to a measurable validation
gate, and no gate depends on modifying source artifacts.

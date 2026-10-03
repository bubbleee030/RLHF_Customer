#!/usr/bin/env bash
# Run J, then the safety-aware RM rebuild.
#
# RUN J — same as Run H (augmented CM, threshold -6.309, actor_lr 1e-4) with ONE
# change: lambda_max 5.0 -> 1.0.
#
# Rationale, from measurement rather than taste. Joining 4,890 blinded judge
# labels to their RM/CM scores gives:
#     RM(full_unsafe_compliance) - RM(safe_refusal) = +2.121   (RM pays to violate)
#     CM(safe_refusal) - CM(full_unsafe_compliance) = -3.790   (CM pays to refuse)
# so a safe refusal only outscores unsafe compliance once lambda > 0.560.
# lambda_max 5.0 is the safe-rlhf default, never tuned here -- Run H trained at
# 8.9x the force required and diverged at step 111, whereas Run G (identical but
# with an inert constraint) survived 400 steps. lambda is the destabiliser.
# 1.0 leaves a 1.79x margin over the crossover without the hammer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pick_pair() {
  mapfile -t u < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${u[0]:-9}" -lt 1000 ] && [ "${u[1]:-9}" -lt 1000 ] && { echo "0,1"; return 0; }
  [ "${u[2]:-9}" -lt 1000 ] && [ "${u[3]:-9}" -lt 1000 ] && { echo "2,3"; return 0; }
  return 1
}
while ! GPUS=$(pick_pair); do sleep 120; done
echo "$(date -Is) Run J on GPUs $GPUS (lambda_max=1.0)"

env ACTOR_MODEL=/backup/model/costomer_model \
    RM_DIR=/backup/reward_output/run_reward_cs_within_20260803/best \
    CM_DIR=cost_output/cmE_augmented_fullft_20260829/best-loss \
    PROMPT_FORMAT=customer_inst PPO_PROMPT_FILE=datasets/reward/cs_within_train.jsonl \
    LORA_R=16 LORA_ALPHA=32 PROMPT_BATCH_SIZE=4 MICRO_BATCH_SIZE=1 MAX_NEW_TOKENS=224 \
    MAX_UPDATES=400 EPOCHS=20 GRADIENT_CHECKPOINTING=1 SAVE_EVERY=50 SAVE_EPOCHS=0 \
    SAVE_CRITICS=0 EARLY_STOP_KL=15.0 CRITIC_LR=5e-5 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    ACTOR_LR=1e-4 KL_COEFF=0.2 THRESHOLD=-6.309 LAMBDA_MAX=1.0 \
    RUN_DIR=ppo_output/runJ_lambdacap_20260830 \
  bash run_exp_docker.sh 'pip install --quiet peft 2>/dev/null; python3 scripts/ppo_lag/train_ppo_lag.py' \
  >> logs/runJ_lambdacap_20260830.log 2>&1
echo "$(date -Is) Run J finished"

# ---------------------------------------------------------------------------
# Safety-aware RM rebuild. The existing RM saw ONLY benign customer-service
# helpfulness pairs, so it never learned that refusing a harmful request beats
# complying -- hence the +2.121. The augmented CM corpus already holds 4,861
# (safe refusal, unsafe compliance) pairs on 979 HARMFUL prompts, audit-verified,
# in exactly the shape the RM trainer consumes.
#
# The two prompt sets are DISJOINT (benign vs harmful), so this adds a region the
# RM has never seen rather than overwriting helpfulness. The original pairs are
# retained deliberately as the guard against "refuse everything" generalising
# onto benign prompts -- the 2026-08-03 safety gate died at 56.8% over-block.
echo "$(date -Is) building safety-aware RM dataset"
python3 - <<'PY'
import json, hashlib, random
random.seed(42)

def load(p): return [json.loads(l) for l in open(p)]

help_train = load('datasets/reward/cs_within_train.jsonl')
help_val   = load('datasets/reward/cs_within_validation.jsonl')
safe_train = load('datasets/augment/cm_pairs_train_strict_20260830.jsonl')
safe_val   = load('datasets/augment/cm_pairs_val_strict_20260830.jsonl')

def to_rm(rows, source):
    out = []
    for r in rows:
        out.append({
            "input": r["input"],
            "chosen": r["answer"],          # audit-verified safe_refusal
            "rejected": r["other_answer"],  # audit-verified unsafe compliance
            "prompt_fingerprint": hashlib.sha256(r["input"].encode()).hexdigest()[:16],
            "pair": "safety",
            "source": source,
        })
    return out

tr = help_train + to_rm(safe_train, "augmented_cm")
va = help_val   + to_rm(safe_val,   "augmented_cm")
random.shuffle(tr)

# Prompt-disjointness is the lesson from the CM leak; assert it rather than assume.
ptr = {r["input"] for r in tr}
pva = {r["input"] for r in va}
overlap = ptr & pva
assert not overlap, f"{len(overlap)} prompts leak between train and val"

import pathlib
d = pathlib.Path('datasets/reward/rm_safety_aware_20260830'); d.mkdir(parents=True, exist_ok=True)
for name, rows in (("train", tr), ("validation", va)):
    with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({
    "train_pairs": len(tr), "val_pairs": len(va),
    "train_helpfulness": len(help_train), "train_safety": len(safe_train),
    "safety_fraction": round(len(safe_train)/len(tr), 3),
    "train_prompts": len(ptr), "val_prompts": len(pva), "prompt_overlap": 0,
}, indent=2))
PY
echo "$(date -Is) RM dataset built -> datasets/reward/rm_safety_aware_20260830"

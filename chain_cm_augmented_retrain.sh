#!/usr/bin/env bash
# Retrain the CM on the augmented dataset, then measure the payoff on the SAME
# 270-response red-team OOD set used for the old and honest CMs.
#
# This is the measurement that tests the professor's hypothesis directly:
#   old CM    (leaked by_pair split)      -> 20.14% unsafe recall
#   honest CM (by_prompt, same data)      -> 35.25%   (leak fix alone)
#   augmented CM (by_prompt + new prompts)-> ?        (does more data close it?)
#
# Hyperparameters are IDENTICAL to the honest-CM run, so the only variable is the
# training data. Anything else would confound the answer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

AUG=datasets/cost/cm_augmented_20260829
RUN=cost_output/cmE_augmented_fullft_20260829

# 1) wait for the build to produce a dataset
while [ ! -s "$AUG/train.jsonl" ] || [ ! -s "$AUG/eval.jsonl" ]; do
  sleep 120
done
echo "$(date -Is) augmented dataset ready: train=$(wc -l < "$AUG/train.jsonl") eval=$(wc -l < "$AUG/eval.jsonl")"

# 2) wait for a free GPU pair. Run C2 holds 2,3; Run D holds 0,1.
#    Poll for two GPUs reporting near-zero used memory.
pick_free_pair() {
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "${used[2]:-99999}" -lt 1000 ] && [ "${used[3]:-99999}" -lt 1000 ]; then echo "2,3"; return 0; fi
  if [ "${used[0]:-99999}" -lt 1000 ] && [ "${used[1]:-99999}" -lt 1000 ]; then echo "0,1"; return 0; fi
  return 1
}
while ! GPUS=$(pick_free_pair); do sleep 180; done
echo "$(date -Is) using GPUs $GPUS"

# A full-FT checkpoint is ~6.9GB; refuse to start without room for it plus slack,
# so we fail loudly here rather than part-way through writing a checkpoint.
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 12 ]; then
  echo "$(date -Is) ABORT: only ${FREE_GB}G free, need >=12G for the checkpoint"
  exit 1
fi

# 3) train — same recipe as the honest-CM run, only the data differs
CUDA_VISIBLE_DEVICES="$GPUS" bash run_exp_docker.sh "python3 scripts/trainer.py \
  --model-name-or-path mistralai/Ministral-3-3B-Instruct-2512 \
  --dataset-path $AUG/train.jsonl \
  --eval-dataset-path $AUG/eval.jsonl \
  --output-dir $RUN \
  --max-length 4096 --batch-size 1 --gradient-accumulation-steps 32 \
  --learning-rate 1e-5 --weight-decay 1e-6 --regularization 0.001 \
  --epochs 3 --warmup-ratio 0.1 --log-steps 10 --seed 42 \
  --pooling last-token --loss-type sequence-wise --load-in-half --device-map \
  --save-eval-predictions --save-backbone --save-best loss \
  --lora-r 0" >> logs/cmE_augmented_20260829.log 2>&1
echo "$(date -Is) CM retrain finished"

# 4) the payoff: same OOD harness, same 270 responses, comparable numbers
if [ -d "$RUN/best-loss" ]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" bash run_exp_docker.sh "python3 scripts/infer_redteam.py \
    --run-dir $RUN/best-loss \
    --redteam-file datasets/cost/redteam_result.json \
    --output results/redteam_augmentedcm.json" >> logs/ood_eval_augmented_20260829.log 2>&1
  echo "$(date -Is) OOD eval done"
  python3 - <<'PY'
import json
rows = [("OLD CM (leaked)", "results/redteam_oldcm.json"),
        ("HONEST CM", "results/redteam_honestcm.json"),
        ("AUGMENTED CM", "results/redteam_augmentedcm.json")]
print(f'{"model":<18}{"unsafe recall":>14}{"safe recall":>13}{"accuracy":>10}{"caught":>9}')
for name, path in rows:
    try:
        s = json.load(open(path))["summary"]
    except Exception:
        print(f"{name:<18}{'(missing)':>14}"); continue
    c = s["confusion"]["Y"]
    print(f'{name:<18}{s["recall_unsafe_Y"]:>14.4f}{s["recall_safe_N"]:>13.4f}'
          f'{s["accuracy"]:>10.4f}{c["unsafe"]:>6}/139')
PY
else
  echo "$(date -Is) no best-loss checkpoint; skipping OOD eval"
fi

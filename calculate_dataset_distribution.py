import json
import random
from collections import Counter

dataset_path = "/home/ubuntu/reward_model/datasets/cost_dataset_for_safe_rlhf_clean.jsonl"

samples = []
skipped = 0

with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
        try: row = json.loads(line)
        except:
            skipped += 1
            continue
            
        prompt = str(row.get("input", "")).strip()
        answer = str(row.get("answer", "")).strip()
        other = str(row.get("other_answer", "")).strip()
        safer = row.get("safer", None)
        is_safe = row.get("is_safe", None)
        is_other_safe = row.get("is_other_safe", None)
        
        if not prompt or not answer or not other:
            skipped += 1
            continue
            
        if not isinstance(safer, bool) or not isinstance(is_safe, bool) or not isinstance(is_other_safe, bool):
            skipped += 1
            continue

        if safer:
            safer_response = answer
            unsafer_response = other
            safer_safety_sign = 2 * int(is_safe) - 1
            unsafer_safety_sign = 2 * int(is_other_safe) - 1
        else:
            safer_response = other
            unsafer_response = answer
            safer_safety_sign = 2 * int(is_other_safe) - 1
            unsafer_safety_sign = 2 * int(is_safe) - 1
            
        if safer_safety_sign < unsafer_safety_sign:
            skipped += 1
            continue
            
        if safer_response == unsafer_response:
            skipped += 1
            continue
            
        pair_type = "unknown"
        if safer_safety_sign == 1 and unsafer_safety_sign == -1:
            pair_type = "Safe vs Unsafe"
        elif safer_safety_sign == 1 and unsafer_safety_sign == 1:
            pair_type = "Safe vs Safe"
        elif safer_safety_sign == -1 and unsafer_safety_sign == -1:
            pair_type = "Unsafe vs Unsafe"
            
        samples.append(pair_type)

n_total = len(samples)
indices = list(range(n_total))
random.Random(42).shuffle(indices)

n_eval = max(1, int(n_total * 0.1))
eval_indices = indices[:n_eval]
train_indices = indices[n_eval:]

train_samples = [samples[i] for i in train_indices]
eval_samples = [samples[i] for i in eval_indices]

train_counter = Counter(train_samples)
eval_counter = Counter(eval_samples)

print(f"Total samples: {n_total}")
print(f"Train samples: {len(train_samples)}")
print("Train distribution:")
for k, v in train_counter.items():
    print(f"  {k}: {v} ({v/len(train_samples)*100:.1f}%)")
    
print(f"\nEval samples: {len(eval_samples)}")
print("Eval distribution:")
for k, v in eval_counter.items():
    print(f"  {k}: {v} ({v/len(eval_samples)*100:.1f}%)")

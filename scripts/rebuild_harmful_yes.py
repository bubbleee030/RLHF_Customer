#!/usr/bin/env python3
import json
from pathlib import Path

def rebuild_harmful_yes():
    harmful_prompts = {}

    # 1. Load the original harmful_yes.jsonl to keep metadata if it exists
    old_file = Path("datasets/harmful_yes.jsonl")
    if old_file.exists():
        with open(old_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                prompt = data.get("x", "").strip()
                if prompt:
                    harmful_prompts[prompt] = data

    print(f"Loaded {len(harmful_prompts)} original harmful prompts with metadata.")

    # 2. Extract from pointwise dataset
    pointwise_file = Path("datasets/cost_model_dataset_pointwise.jsonl")
    new_pointwise = 0
    if pointwise_file.exists():
        with open(pointwise_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    prompt = data.get("x", "").strip()
                    if data.get("is_harmful") == "Yes" and prompt:
                        if prompt not in harmful_prompts:
                            harmful_prompts[prompt] = {"x": prompt, "is_harmful": "Yes"}
                            new_pointwise += 1
                except json.JSONDecodeError:
                    pass
    
    print(f"Found {new_pointwise} NEW harmful prompts from pointwise dataset.")

    # 3. Extract from pairwise dataset
    pairwise_file = Path("datasets/cost_model_dataset_pairwise.jsonl")
    new_pairwise = 0
    if pairwise_file.exists():
        with open(pairwise_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    prompt = data.get("x", "").strip()
                    if data.get("is_harmful") == "Yes" and prompt:
                        if prompt not in harmful_prompts:
                            harmful_prompts[prompt] = {"x": prompt, "is_harmful": "Yes"}
                            new_pairwise += 1
                except json.JSONDecodeError:
                    pass

    print(f"Found {new_pairwise} NEW harmful prompts from pairwise dataset.")

    # 4. Save to clean file
    out_file = Path("datasets/harmful_yes_clean.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for prompt, data in harmful_prompts.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"✅ Successfully wrote {len(harmful_prompts)} unique harmful prompts to {out_file}")

if __name__ == "__main__":
    rebuild_harmful_yes()

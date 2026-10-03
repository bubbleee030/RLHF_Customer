import json

harmful_yes_path = "/Users/bubble/Antigravity/reward_model/datasets/harmful_yes.jsonl"
full_prompts_path = "/Users/bubble/Antigravity/reward_model/test/prompts/GeneratedPromptsFull.jsonl"

with open(harmful_yes_path, 'r', encoding='utf-8') as f:
    harmful_yes = [json.loads(line) for line in f if line.strip()]
    
with open(full_prompts_path, 'r', encoding='utf-8') as f:
    full_prompts = [json.loads(line) for line in f if line.strip()]

hy_set = {item['x'].strip() for item in harmful_yes if 'x' in item}
fp_dict = {item['prompt'].strip(): item for item in full_prompts if 'prompt' in item}
fp_set = set(fp_dict.keys())

intersection = hy_set.intersection(fp_set)
print(f"harmful_yes items: {len(hy_set)}")
print(f"GeneratedPromptsFull items: {len(fp_set)}")
print(f"Intersection: {len(intersection)}")

if len(intersection) < len(hy_set):
    print("Some items in harmful_yes are NOT in GeneratedPromptsFull.")
    # Check severity in harmful_yes
    has_severity = sum(1 for item in harmful_yes if 'severity' in item)
    print(f"Items already having severity in harmful_yes: {has_severity}")
else:
    print("All items in harmful_yes are in GeneratedPromptsFull.")

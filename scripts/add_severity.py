import json

harmful_yes_path = "/Users/bubble/Antigravity/reward_model/datasets/harmful_yes.jsonl"
full_prompts_path = "/Users/bubble/Antigravity/reward_model/test/prompts/GeneratedPromptsFull.jsonl"

with open(full_prompts_path, 'r', encoding='utf-8') as f:
    full_prompts = [json.loads(line) for line in f if line.strip()]

fp_dict = {item['prompt'].strip(): item['severity'] for item in full_prompts if 'prompt' in item and 'severity' in item}

new_harmful_yes = []
with open(harmful_yes_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        prompt_text = data.get('x', '').strip()
        if prompt_text in fp_dict:
            data['severity'] = fp_dict[prompt_text]
        else:
            data['severity'] = "Unknown"
        new_harmful_yes.append(data)

with open(harmful_yes_path, 'w', encoding='utf-8') as f:
    for data in new_harmful_yes:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
print("Successfully added severity!")

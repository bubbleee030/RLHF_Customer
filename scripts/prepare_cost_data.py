import json
import argparse
import os
from collections import defaultdict

def parse_reward_dataset(input_file):
    results = []
    skipped = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"Skipping malformed line: {e}")
                skipped += 1
                continue
            
            # 從原資料中提取資訊
            prompt = data.get('x', '')
            y_w = data.get('y_w', '')
            y_l = data.get('y_l', '')
            is_harmful_str = data.get('is_harmful', 'No')
            
            is_harmful = (is_harmful_str == 'Yes')
            
            # 轉換為 Safe-RLHF 預期的格式
            # (參考 safe_rlhf.datasets.safety_preference.SafetyPreferenceDataset)
            out_data = {
                'input': prompt,
                'answer': y_w,          # y_w 被預設為較安全或較好的回答
                'other_answer': y_l,
                'safer': True,          # `answer`(y_w) 相對 `other_answer`(y_l) 更安全/較佳
                'is_safe': True,        # `answer`(y_w) 本身是否安全 -> 是
                'is_other_safe': not is_harmful  # 如果原始 prompt 是有害的(Yes)，則 y_l 通常是不安全的回應；否則均為安全(No)
            }
            results.append(out_data)
    print(f"✅ Processed {len(results)} pairs from {input_file}")
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} malformed lines.")
    return results

def parse_cost_dataset(input_file):
    results = []
    skipped = 0
    groups = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                groups[data['x']].append(data)
            except Exception as e:
                skipped += 1
                
    for x, items in groups.items():
        is_harmful_str = items[0].get('is_harmful', 'No')
        is_harmful = (is_harmful_str == 'Yes')
        
        safe_ones = [d for d in items if d.get('s') == 1]
        unsafe_ones = [d for d in items if d.get('s') == -1]
        
        if is_harmful and safe_ones and unsafe_ones:
            # Pair safe and unsafe for harmful prompts
            for s_item in safe_ones:
                for u_item in unsafe_ones:
                    results.append({
                        'input': x,
                        'answer': s_item['y'],
                        'other_answer': u_item['y'],
                        'safer': True,
                        'is_safe': True,
                        'is_other_safe': False
                    })
        elif not is_harmful and len(safe_ones) >= 2:
            # For harmless prompts, pair the safe responses together randomly (pair offset by 1)
            for i in range(0, len(safe_ones) - 1, 2):
                results.append({
                    'input': x,
                    'answer': safe_ones[i]['y'],
                    'other_answer': safe_ones[i+1]['y'],
                    'safer': True,
                    'is_safe': True,
                    'is_other_safe': True
                })

    print(f"✅ Processed {len(results)} logical pairs from {input_file} (point-wise grouping)")
    if skipped > 0:
        print(f"⚠️ Skipped {skipped} malformed lines.")
    return results

def main():
    parser = argparse.ArgumentParser(description="Merge and convert datasets to safe-rlhf pairwise format")
    parser.add_argument("--reward_file", type=str, default="./datasets/cost_model_dataset_pairwise.jsonl")
    parser.add_argument("--cost_file", type=str, default="./datasets/cost_model_dataset_pointwise.jsonl")
    parser.add_argument("--output_file", type=str, default="./datasets/cost_dataset_for_safe_rlhf.jsonl")
    args = parser.parse_args()

    all_pairs = []
    
    if os.path.exists(args.reward_file):
        all_pairs.extend(parse_reward_dataset(args.reward_file))
    else:
        print(f"⚠️ Warning: {args.reward_file} not found.")

    if os.path.exists(args.cost_file):
        all_pairs.extend(parse_cost_dataset(args.cost_file))
    else:
        print(f"⚠️ Warning: {args.cost_file} not found.")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:  # Fix the FileNotFoundError if output_file is just a file name with no directory
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in all_pairs:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    print(f"🎉 Successfully wrote {len(all_pairs)} TOTAL combined paired examples to {args.output_file}")

if __name__ == "__main__":
    main()

import json
from collections import Counter
import os

def analyze_distribution(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    main_cat_counter = Counter()
    sub_cat_counter = Counter()
    severity_counter = Counter()
    total_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                main_cat = data.get('category', 'Unknown')
                sub_cat = data.get('sub_category', 'Unknown')
                severity = data.get('severity', 'Unknown')
                
                main_cat_counter[main_cat] += 1
                sub_cat_counter[sub_cat] += 1
                severity_counter[severity] += 1
                total_count += 1
            except json.JSONDecodeError:
                print("Error parsing a line in JSONL.")

    print("=" * 50)
    print(f"Total entries: {total_count}")
    print("=" * 50)
    
    print("\n[Main Category Distribution]")
    for cat, count in main_cat_counter.most_common():
        percentage = (count / total_count) * 100
        print(f"{cat}: {count} ({percentage:.1f}%)")

    print("\n[Sub Category Distribution]")
    for subcat, count in sub_cat_counter.most_common():
        percentage = (count / total_count) * 100
        print(f"{subcat}: {count} ({percentage:.1f}%)")
    
    print("\n[Severity Distribution]")
    for severity, count in severity_counter.most_common():
        percentage = (count / total_count) * 100
        print(f"{severity}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    filepath = "/Users/bubble/Antigravity/reward_model/datasets/harmful_yes.jsonl"
    analyze_distribution(filepath)

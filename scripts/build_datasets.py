import json
import itertools
import os

def build_datasets():
    records_file = "records_api.json"
    if not os.path.exists(records_file):
        print(f"Error: {records_file} not found. Please run fetch_api.py first.")
        return

    with open(records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    pointwise_cost_model_data = []    # Format: (x, y, s, is_harmful)
    pairwise_cost_model_data = []  # Format: (x, y_w, y_l, is_harmful)

    for record in records:
        # skip completed records cuz the other not yet done
        # if record.get("status") != "completed":
        #     continue 

        fields = record.get("fields", {})
        prompt = fields.get("prompt", "")
        
        # Responses
        resps = {
            "R1": fields.get("R1"),
            "R2": fields.get("R2"),
            "R3": fields.get("R3"),
            "R4": fields.get("R4")
        }

        # User submitted responses
        responses_list = record.get("responses", [])
        user_response = next((r for r in responses_list if r.get("status") == "submitted"), None)
        
        if not user_response:
            continue

        values = user_response.get("values", {})
        is_harmful = values.get("is_harmful", {}).get("value")

        # --- 1. Pointwise Cost Model Dataset ---
        # Safe = +1, Harmful = -1
        for rx in ["R1", "R2", "R3", "R4"]:
            safety_key = f"safety_{rx.lower()}"
            user_saf = values.get(safety_key, {}).get("value")
            
            if user_saf and resps.get(rx):
                # Safe -> +1, Harmful -> -1
                s_val = 1 if user_saf == "Safe" else -1
                pointwise_cost_model_data.append({
                    "x": prompt,
                    "y": resps[rx],
                    "s": s_val,
                    "is_harmful": is_harmful,
                    "response_id": rx
                })

        # --- 2. Pairwise Cost Model Dataset ---
        helpfulness_ranking = values.get("helpfulness_ranking", {}).get("value", [])
        if helpfulness_ranking:
            # 排序根據 rank 遞增 (1, 2, 3, 4)
            sorted_ranking = sorted(helpfulness_ranking, key=lambda x: x.get("rank", 99))
            user_ranking = [item["value"] for item in sorted_ranking]
            
            # 如果 ranking 完全等於預設的 ["R1", "R2", "R3", "R4"]，視為 Tie 不加入
            if user_ranking == ["R1", "R2", "R3", "R4"]:
                continue
                
            for i in range(len(user_ranking)):
                for j in range(i + 1, len(user_ranking)):
                    better_rx = user_ranking[i]
                    worse_rx = user_ranking[j]
                    
                    y_w = resps.get(better_rx)
                    y_l = resps.get(worse_rx)
                    
                    if y_w and y_l:
                        pairwise_cost_model_data.append({
                            "x": prompt,
                            "y_w": y_w,
                            "y_l": y_l,
                            "is_harmful": is_harmful,
                            "winner": better_rx,
                            "loser": worse_rx
                        })

    # Save outputs
    pointwise_cost_file = "cost_model_dataset_pointwise.jsonl"
    pairwise_cost_file = "cost_model_dataset_pairwise.jsonl"
    
    with open(pointwise_cost_file, "w", encoding="utf-8") as f:
        for item in pointwise_cost_model_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(pairwise_cost_file, "w", encoding="utf-8") as f:
        for item in pairwise_cost_model_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset generated!")
    print(f"- Pointwise Cost Model Data: {len(pointwise_cost_model_data)} records -> {pointwise_cost_file}")
    print(f"- Pairwise Cost Model Data: {len(pairwise_cost_model_data)} pairs -> {pairwise_cost_file}")

if __name__ == "__main__":
    build_datasets()

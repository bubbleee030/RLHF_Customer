import os
import argilla as rg
import json
import sys

# 連線
API_URL = "https://your-argilla-server.hf.space"
API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
DATASET_NAME = "TAIWAN_AI_RAP_Safety_Alignment"

try:
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

dataset = client.datasets(DATASET_NAME)
if not dataset:
    print("Dataset not found!")
    sys.exit(1)

print("Fetching records via SDK...")
records = list(dataset.records(with_suggestions=True, with_responses=True))
print(f"Total records fetched via SDK: {len(records)}")

out_data = []

for r in records:
    rec_dict = {
        "id": str(r.id),
        "status": r.status,
        "fields": r.fields,
        "responses": {},
        "suggestions": {}
    }
    
    # Extract responses correctly
    if r.responses:
        # r.responses is a dictionary or list, depending on SDK version
        for key, resp in r.responses.items():
            if isinstance(resp, list):
                rec_dict["responses"][key] = [
                    {"value": user_resp.value, "user_id": str(user_resp.user_id)} 
                    for user_resp in resp
                ]
            else:
                rec_dict["responses"][key] = {
                    "value": resp.value, "user_id": str(resp.user_id)
                }

    if r.suggestions:
        # r.suggestions is list or dict
        if isinstance(r.suggestions, dict):
            for key, sug in r.suggestions.items():
                rec_dict["suggestions"][key] = {"value": getattr(sug, "value", sug)}
        elif isinstance(r.suggestions, list):
            for sug in r.suggestions:
                # Based on suggestions structure, which might be a list of Suggestion objects
                rec_dict["suggestions"][sug.question_name] = {"value": sug.value}
                
    out_data.append(rec_dict)

output_file = "/home/ubuntu/reward_model/records_api.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)
print(f"Saved into {output_file}")

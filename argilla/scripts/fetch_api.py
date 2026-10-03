import json
import requests
import os

API_URL = "https://your-argilla-server.hf.space"
API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
DATASET_NAME = "TAIWAN_AI_RAP_Safety_Alignment"

headers = {
    "X-Argilla-API-Key": API_KEY,
    "Content-Type": "application/json"
}

print(f"Connecting to {API_URL}...")

try:
    # 1. Fetch datasets directly for the user
    datasets_res = requests.get(f"{API_URL}/api/v1/me/datasets", headers=headers)
    datasets_res.raise_for_status()
    datasets = datasets_res.json().get("items", [])
    
    dataset = next((d for d in datasets if d["name"] == DATASET_NAME), None)
    if not dataset:
        raise ValueError(f"Dataset {DATASET_NAME} not found for this user.")
        
    dataset_id = dataset["id"]
    print(f"Dataset ID: {dataset_id}")

    # 2. Fetch records with pagination
    all_records = []
    offset = 0
    limit = 100

    while True:
        print(f"Fetching records offset={offset}...")
        res = requests.get(f"{API_URL}/api/v1/datasets/{dataset_id}/records?offset={offset}&limit={limit}&include=responses&include=suggestions", headers=headers)
        res.raise_for_status()
        items = res.json().get("items", [])
        if not items:
            break
        all_records.extend(items)
        offset += limit
        
    print(f"Total records fetched: {len(all_records)}")

    output_file = "/home/ubuntu/reward_model/records_api.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"Saved correctly to {output_file}")
except Exception as e:
    print(f"Error: {e}")

#!/usr/bin/env python3
"""
Disaster-recovery re-upload of TAIWAN_AI_RAP_Helpfulness after the Argilla
Space reset wiped the live dataset AND the annotator user.

Recreates:
  - user1 (annotator, password 12345678) + workspace membership
  - the dataset from backup settings
  - all 373 submitted human rankings, re-attributed to user1

Source: argilla/backups/TAIWAN_AI_RAP_Helpfulness_20260525_192914
  - records.fixed.json     : records with CORRECT ranking order (records.json is buggy)
  - .argilla/settings.json : dataset settings

Suggestions are intentionally NOT restored (they were the corrupted rank=null
field removed previously; the clean approved state had 0 suggestions).
"""
import os
import json
from pathlib import Path

import argilla as rg

API_URL = "https://your-argilla-server.hf.space"
API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
DATASET_NAME = "TAIWAN_AI_RAP_Helpfulness"
WORKSPACE = "argilla"
TARGET_USERNAME = "user1"
TARGET_PASSWORD = "12345678"
QUESTION = "helpfulness_ranking"

BACKUP = Path(__file__).resolve().parent.parent / "backups" / "TAIWAN_AI_RAP_Helpfulness_20260525_192914"
SETTINGS_JSON = BACKUP / ".argilla" / "settings.json"
RECORDS_JSON = BACKUP / "records.fixed.json"


def build_settings(raw: dict) -> rg.Settings:
    fields = [
        rg.TextField(
            name=f["name"],
            title=f.get("title"),
            use_markdown=f["settings"].get("use_markdown", False),
            required=f.get("required", True),
        )
        for f in raw["fields"]
    ]
    q = next(q for q in raw["questions"] if q["name"] == QUESTION)
    questions = [
        rg.RankingQuestion(
            name=q["name"],
            title=q.get("title"),
            description=q.get("description"),
            values=[o["value"] for o in q["settings"]["options"]],
            required=q.get("required", True),
        )
    ]
    dist = raw.get("distribution", {})
    return rg.Settings(
        guidelines=raw.get("guidelines"),
        distribution=rg.TaskDistribution(min_submitted=dist.get("min_submitted", 2)),
        fields=fields,
        questions=questions,
    )


def ensure_user(client: rg.Argilla):
    user = client.users(TARGET_USERNAME)
    if user is None:
        print(f"Creating user '{TARGET_USERNAME}'...")
        user = rg.User(username=TARGET_USERNAME, password=TARGET_PASSWORD).create()
    else:
        print(f"User '{TARGET_USERNAME}' already exists.")
    ws = client.workspaces(WORKSPACE)
    try:
        user.add_to_workspace(ws)
        print(f"Added '{TARGET_USERNAME}' to workspace '{WORKSPACE}'.")
    except Exception as e:
        print(f"(workspace add skipped: {e})")
    return user.id


def main():
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)

    target_uid = ensure_user(client)
    print(f"Target user id = {target_uid}")

    settings_raw = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
    records_raw = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(records_raw)} records from backup")

    existing = client.datasets(name=DATASET_NAME, workspace=WORKSPACE)
    if existing is not None:
        print(f"Deleting existing online dataset '{DATASET_NAME}'...")
        existing.delete()

    dataset = rg.Dataset(name=DATASET_NAME, workspace=WORKSPACE, settings=build_settings(settings_raw))
    dataset.create()
    print(f"Recreated dataset '{DATASET_NAME}'")

    rg_records = []
    for r in records_raw:
        fields = r["fields"]
        responses = []
        for resp in (r.get("responses") or {}).get(QUESTION, []):
            val = resp.get("value")
            if isinstance(val, list) and len(val) == 4:
                # Ranking responses need explicit ranks (plain list -> rank None -> 422)
                ranked = [{"value": v, "rank": i + 1} for i, v in enumerate(val)]
                responses.append(
                    rg.Response(QUESTION, ranked, user_id=target_uid, status="submitted")
                )
        rg_records.append(
            rg.Record(
                fields={k: fields[k] for k in ("prompt", "R1", "R2", "R3", "R4")},
                responses=responses or None,
            )
        )

    n_resp = sum(1 for x in rg_records if x.responses)
    print(f"Prepared {len(rg_records)} records ({n_resp} with a {TARGET_USERNAME} response)")
    dataset.records.log(rg_records)
    print("Upload complete.")


if __name__ == "__main__":
    main()

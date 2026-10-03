#!/usr/bin/env python3
"""
Restore a TAIWAN_AI_RAP_Helpfulness backup over the live online dataset,
re-attributing the backed-up submitted annotations to a target user (user1).

Source backup: argilla/backups/TAIWAN_AI_RAP_Helpfulness_20260525_192914
  - records.fixed.json : SDK records w/ fields + suggestions + (corrected) responses
  - .argilla/settings.json : dataset settings (guidelines/fields/question/distribution)
"""
import os
import json
from pathlib import Path

import argilla as rg

API_URL = "https://your-argilla-server.hf.space"
API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
DATASET_NAME = "TAIWAN_AI_RAP_Helpfulness"
WORKSPACE = "argilla"
TARGET_USERNAME = "user1"  # annotations get re-attributed to this user

BACKUP = Path(__file__).resolve().parent.parent / "backups" / "TAIWAN_AI_RAP_Helpfulness_20260525_192914"
SETTINGS_JSON = BACKUP / ".argilla" / "settings.json"
RECORDS_JSON = BACKUP / "records.fixed.json"
QUESTION = "helpfulness_ranking"


def build_settings(raw: dict) -> rg.Settings:
    fields = []
    for f in raw["fields"]:
        fields.append(
            rg.TextField(
                name=f["name"],
                title=f.get("title"),
                use_markdown=f["settings"].get("use_markdown", False),
                required=f.get("required", True),
            )
        )
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


def main():
    client = rg.Argilla(api_url=API_URL, api_key=API_KEY)
    target_user = client.users(TARGET_USERNAME)
    if target_user is None:
        raise SystemExit(f"Target user '{TARGET_USERNAME}' not found")
    target_uid = target_user.id
    print(f"Target user '{TARGET_USERNAME}' id = {target_uid}")

    settings_raw = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
    records_raw = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(records_raw)} records from backup")

    # --- delete existing online dataset ---
    existing = client.datasets(name=DATASET_NAME, workspace=WORKSPACE)
    if existing is not None:
        print(f"Deleting existing online dataset '{DATASET_NAME}'...")
        existing.delete()

    # --- recreate from backup settings ---
    dataset = rg.Dataset(name=DATASET_NAME, workspace=WORKSPACE, settings=build_settings(settings_raw))
    dataset.create()
    print(f"Recreated dataset '{DATASET_NAME}'")

    # --- build records: fields + suggestion + response(user1, submitted) ---
    rg_records = []
    for r in records_raw:
        fields = r["fields"]
        suggestions = []
        sug = (r.get("suggestions") or {}).get(QUESTION)
        if sug and isinstance(sug.get("value"), list):
            suggestions.append(rg.Suggestion(QUESTION, sug["value"]))

        responses = []
        for resp in (r.get("responses") or {}).get(QUESTION, []):
            val = resp.get("value")
            if isinstance(val, list) and len(val) == 4:
                # Ranking responses need explicit ranks; the SDK does NOT
                # auto-assign ranks from a plain ordered list (rank -> None -> 422).
                ranked = [{"value": v, "rank": i + 1} for i, v in enumerate(val)]
                responses.append(
                    rg.Response(QUESTION, ranked, user_id=target_uid, status="submitted")
                )

        rg_records.append(
            rg.Record(
                fields={k: fields[k] for k in ("prompt", "R1", "R2", "R3", "R4")},
                suggestions=suggestions or None,
                responses=responses or None,
            )
        )

    n_resp = sum(1 for x in rg_records if x.responses)
    print(f"Prepared {len(rg_records)} records ({n_resp} with a {TARGET_USERNAME} response)")
    dataset.records.log(rg_records)
    print("Upload complete.")


if __name__ == "__main__":
    main()

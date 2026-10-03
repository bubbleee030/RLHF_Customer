#!/usr/bin/env python3
"""
上傳 Helpfulness-only 標注任務到 Argilla。

資料來源：argilla/records.json（Safety Alignment 備份）
目標資料集：TAIWAN_AI_RAP_Helpfulness

本次標注只評估 Helpfulness，不評估 Safety。
標注者只需對同一 Prompt 的 R1~R4 依有用程度排序。
"""

import os
import json
import sys
from pathlib import Path

import argilla as rg

# ── 連線設定 ────────────────────────────────────────────────────
ARGILLA_API_URL = os.environ.get("ARGILLA_API_URL", "https://your-argilla-server.hf.space")
ARGILLA_API_KEY = os.environ["ARGILLA_API_KEY"]  # required; no default
DATASET_NAME = "TAIWAN_AI_RAP_Helpfulness"
WORKSPACE_NAME = "argilla"

# ── 資料來源 ────────────────────────────────────────────────────
RECORDS_FILE = Path(__file__).resolve().parent.parent / "records.json"

GUIDELINES = """## Helpfulness 標注指引

本次標注只評估「有用性 (Helpfulness)」，**不評估安全性**。

請比較同一使用者問題下的四個模型回覆，並排出哪一個對使用者更有幫助。

### 有幫助的回答通常具備以下特徵：
1. **更直接回答**使用者的問題
2. 提供**更充分、具體、可操作**的資訊
3. **較少明顯錯誤**或違反常識的資訊
4. 若拒絕回答，仍有**清楚說明拒答的理由**、違反的政策與替代性資訊
5. **表達更清楚、有邏輯**、容易閱讀

### ⚠️ 注意事項
- Safety **不納入**本次評估，請勿考量回答是否安全、合規或是否應該拒答
- 請只問自己：「哪個回答更滿足使用者原本的需求？」
- 排名 1 = 最有幫助，排名 4 = 最沒有幫助
"""


def connect(api_url: str, api_key: str) -> rg.Argilla:
    try:
        client = rg.Argilla(api_url=api_url, api_key=api_key)
        print(f"✅ 成功連線至 Argilla: {api_url}")
        return client
    except Exception as e:
        print(f"❌ 連線失敗: {e}")
        sys.exit(1)


def create_dataset(client: rg.Argilla) -> rg.Dataset:
    settings = rg.Settings(
        guidelines=GUIDELINES,
        distribution=rg.TaskDistribution(min_submitted=1),
        fields=[
            rg.TextField(name="prompt", title="[User Prompt] 原始問題"),
            rg.TextField(name="R1", title="Response 1"),
            rg.TextField(name="R2", title="Response 2"),
            rg.TextField(name="R3", title="Response 3"),
            rg.TextField(name="R4", title="Response 4"),
        ],
        questions=[
            rg.RankingQuestion(
                name="helpfulness_ranking",
                title="請根據「對使用者的幫助程度」排出名次（最上方 = 最有幫助）",
                description="只評估 Helpfulness，不考慮 Safety。",
                values=["R1", "R2", "R3", "R4"],
                required=True,
            ),
        ],
    )

    # 若已存在則先刪除
    try:
        existing = client.datasets(name=DATASET_NAME, workspace=WORKSPACE_NAME)
        if existing:
            print(f"⚠️  發現已存在 '{DATASET_NAME}'，先刪除...")
            existing.delete()
    except Exception:
        pass

    dataset = rg.Dataset(name=DATASET_NAME, workspace=WORKSPACE_NAME, settings=settings)
    dataset.create()
    print(f"✅ Dataset '{DATASET_NAME}' 建立完成")
    return dataset


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        print(f"❌ 找不到資料檔案: {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"📄 讀取 {len(records)} 筆記錄 from {path}")
    return records


def build_rg_records(raw_records: list[dict]) -> list[rg.Record]:
    rg_records = []
    skipped = 0

    for r in raw_records:
        fields = r.get("fields", {})
        prompt = fields.get("prompt", "").strip()
        r1 = fields.get("R1", "").strip()
        r2 = fields.get("R2", "").strip()
        r3 = fields.get("R3", "").strip()
        r4 = fields.get("R4", "").strip()

        if not all([prompt, r1, r2, r3, r4]):
            skipped += 1
            continue

        # 沿用舊有 LLM suggestion 的 helpfulness_ranking（若有）
        suggestions = []
        old_sug = r.get("suggestions", {})
        ranking_sug = old_sug.get("helpfulness_ranking", {})
        ranking_value = ranking_sug.get("value")
        if isinstance(ranking_value, list) and len(ranking_value) == 4:
            suggestions.append(rg.Suggestion("helpfulness_ranking", ranking_value))

        record = rg.Record(
            fields={"prompt": prompt, "R1": r1, "R2": r2, "R3": r3, "R4": r4},
            suggestions=suggestions if suggestions else None,
        )
        rg_records.append(record)

    if skipped:
        print(f"⚠️  跳過 {skipped} 筆缺少欄位的記錄")
    print(f"✅ 準備上傳 {len(rg_records)} 筆記錄")
    return rg_records


def main():
    client = connect(ARGILLA_API_URL, ARGILLA_API_KEY)
    raw_records = load_records(RECORDS_FILE)
    dataset = create_dataset(client)
    rg_records = build_rg_records(raw_records)

    print("📤 上傳中...")
    dataset.records.log(rg_records)
    print(f"🎉 成功推送 {len(rg_records)} 筆記錄至 Argilla！")
    print(f"   前往 {ARGILLA_API_URL} 開始標注")


if __name__ == "__main__":
    main()

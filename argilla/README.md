# Argilla 標注工作流

Argilla 伺服器：`https://your-argilla-server.hf.space`

---

## 資料來源說明

| 檔案 | 來源 | 說明 |
|---|---|---|
| `argilla/records.json` | 舊 VM (`reward_model0`) + Argilla SDK `dataset.to_disk()` | Safety Alignment 標注備份，373 筆，含 LLM suggestions |
| `records_api.json` (root) | Argilla REST API `fetch_api.py` | 同樣 373 筆，**含人工標注結果**（submitted responses） |
| `test/prompts/response_set_20260323_130841_judged.jsonl` | 舊 VM，Step 14 LLM Judge 輸出 | 上傳 Argilla 的原始來源（prompt + R1~R4 + LLM label） |
| `test/prompts/prompt_pool_20260323_144720.jsonl` | 舊 VM，Step 12 混合 | 188 Harmful + 188 Normal，共 376 筆（已隨機混合） |
| `docs/reports/` | 舊 VM，`test/reports/` | 5 份規劃與進度報告（Taxonomy v5、Response Gen、HPC 等） |

---

## 目錄結構

```
argilla/
├── .argilla/           # Safety Alignment dataset.json + settings.json
├── records.json        # 373 筆 Safety Alignment 備份（from Argilla SDK）
└── scripts/
    ├── fetch_api.py          # 用 REST API 抓回 records → records_api.json
    ├── test_argilla_fetch.py # 用 SDK 抓回記錄（除錯用）
    ├── test_api_routes.py    # 測試 Argilla API 連線
    ├── auto_backup.py        # 定期自動備份 Argilla 資料集到 backups/
    ├── backup.sh             # 備份 shell wrapper（呼叫 auto_backup.py）
    └── upload_helpfulness.py # 上傳 Helpfulness-only 標注任務 ⬅ Round 2

docs/
├── Cell6_Cell9_Modification_Notes.md  # Notebook 修改說明（sub-topic 機制）
└── reports/
    ├── Safety Alignment Plan: Preference Dataset Taxonomy v5.md
    ├── Safety Alignment Plan: Harmful Prompt Augmentation Progress Report v1.md
    ├── Safety Alignment Plan: Harmful Prompt Augmentation Progress Report v2.md
    ├── Safety Alignment Plan: Response Generation Progress Report v1.md
    └── Safety Alignment Plan: Reward and Cost Model Training Progress Report v1.md
```

---

## Round 1：Safety Alignment 標注（已完成）

### 資料集名稱
`TAIWAN_AI_RAP_Safety_Alignment`

### 問題設計
| Question | Type | 說明 |
|---|---|---|
| `helpfulness_ranking` | Ranking | R1~R4 整體品質排序 |
| `safety_r1~r4` | Label | 每個回答 Safe / Harmful |

### 上傳流程（參考 test.ipynb Step 14-15）
1. **Step 14**：用 `JUDGE_MODEL`（Llama-3.3-Nemotron-Super-49B）對每筆 prompt 的 4 個回答做 LLM-as-Judge 預標注，輸出 `_judged.jsonl`
2. **Step 15**：用 Argilla SDK 建立 Dataset + Settings，將 judged 結果作為 `rg.Suggestion` 推送

```python
import argilla as rg

client = rg.Argilla(
    api_url="https://your-argilla-server.hf.space",
    api_key="<ARGILLA_API_KEY>"
)

# 建立 Dataset
dataset = rg.Dataset(name="TAIWAN_AI_RAP_Safety_Alignment", workspace="argilla", settings=settings)
dataset.create()

# 上傳 records
dataset.records.log(rg_records)
```

### 下載備份
```bash
# 用 SDK 備份到本機
python argilla/scripts/auto_backup.py \
  --api-url https://your-argilla-server.hf.space \
  --api-key <KEY> \
  --dataset TAIWAN_AI_RAP_Safety_Alignment \
  --workspace argilla

# 用 REST API 抓 records
python argilla/scripts/fetch_api.py
```

---

## Round 2：Helpfulness 標注（進行中）

### 資料集名稱
`TAIWAN_AI_RAP_Helpfulness`

### 評估原則
- **只評估 Helpfulness，不評估 Safety**
- 問自己：「哪個回答更滿足使用者原本的需求？」
- 有幫助的回答特徵：
  - 更直接回答使用者的問題
  - 提供更充分、具體、可操作的資訊
  - 較少明顯錯誤或違反常識的資訊
  - 若拒絕回答，有清楚說明拒答理由、政策與替代資訊
  - 表達更清楚、有邏輯、容易閱讀

### 問題設計
| Question | Type | 說明 |
|---|---|---|
| `helpfulness_ranking` | Ranking | R1~R4 依 Helpfulness 排序（1=最有幫助） |

### 上傳流程
```bash
python argilla/scripts/upload_helpfulness.py
```

腳本會：
1. 讀取 `argilla/records.json`（舊 Safety 標注資料集，保留 prompt + R1~R4）
2. 建立新的 `TAIWAN_AI_RAP_Helpfulness` Dataset（只有 helpfulness_ranking）
3. 沿用舊的 LLM suggestion（ranking）作為初始建議
4. 推送全部 373 筆記錄

### 標注完成後：取回資料
```bash
# 修改 fetch_api.py 的 DATASET_NAME 再執行
DATASET_NAME="TAIWAN_AI_RAP_Helpfulness" python argilla/scripts/fetch_api.py
```

取回的 JSON 可直接用 `scripts/build_datasets.py` 建立 pairwise 訓練資料。

---

---

## 日常操作

### 備份資料集

```bash
cd /home/ubuntu/reward_model

# 單次備份（預設備份 Safety Alignment）
bash argilla/scripts/backup.sh backup

# 備份 Helpfulness 資料集
python argilla/scripts/auto_backup.py \
  --dataset TAIWAN_AI_RAP_Helpfulness \
  --api-key <KEY>

# 定期自動備份（每 120 分鐘）
bash argilla/scripts/backup.sh schedule 120

# 列出所有備份
bash argilla/scripts/backup.sh list
```

備份輸出：
```
argilla/backups/
├── TAIWAN_AI_RAP_Safety_Alignment_20260325_143022/
│   ├── records.json          # SDK 備份（原始格式）
│   ├── records_api.json      # REST API 備份（含正確 Ranking 順序）
│   ├── records.fixed.json    # 修正 Ranking 序列化 bug 後的版本
│   ├── ranking_mismatch_report.json
│   └── backup_metadata.json
└── latest/                   # 永遠指向最新備份
```

> **注意**：`auto_backup.py` 會自動比較 hash，若資料無變動則跳過重複備份，最多保留 5 份（可用 `--max-backups` 調整）。

---

### 查看標注進度

#### 方法一：Argilla Web UI
前往 `https://your-argilla-server.hf.space`，登入後點選對應資料集，右上角可看到：
- **Pending**：尚未標注的筆數
- **Submitted**：已完成的筆數
- **Progress bar**：整體完成比例

#### 方法二：REST API 查詢

```bash
# 查詢目前資料集狀態
curl -H "X-Argilla-API-Key: <KEY>" \
  "https://your-argilla-server.hf.space/api/v1/me/datasets" | python3 -m json.tool

# 計算已標注筆數（records with responses）
python3 - <<'EOF'
import requests, json
API_URL = "https://your-argilla-server.hf.space"
API_KEY = "<KEY>"
DATASET_NAME = "TAIWAN_AI_RAP_Helpfulness"
headers = {"X-Argilla-API-Key": API_KEY}

datasets = requests.get(f"{API_URL}/api/v1/me/datasets", headers=headers).json()["items"]
ds = next(d for d in datasets if d["name"] == DATASET_NAME)
ds_id = ds["id"]

all_records, offset = [], 0
while True:
    r = requests.get(f"{API_URL}/api/v1/datasets/{ds_id}/records?offset={offset}&limit=100&include=responses", headers=headers).json()
    items = r.get("items", [])
    if not items: break
    all_records.extend(items)
    offset += 100

submitted = sum(1 for r in all_records if r.get("responses"))
print(f"Total: {len(all_records)}  Submitted: {submitted}  Pending: {len(all_records)-submitted}")
EOF
```

---

### 新增標注者帳號

#### 方法：Argilla Web UI（最簡單）
1. 前往 `https://your-argilla-server.hf.space` → 以 admin 身份登入
2. 左側選單 → **Settings** → **Users**
3. 點 **+ Add User**，填入 username / password / role（`annotator`）
4. 新使用者前往同一 URL 登入即可開始標注

#### 方法：REST API

```bash
# 建立新使用者（role: "annotator" 或 "admin"）
curl -X POST \
  -H "X-Argilla-API-Key: <ADMIN_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"username":"annotator01","password":"mypassword","role":"annotator","first_name":""}' \
  "https://your-argilla-server.hf.space/api/v1/users"

# 列出所有使用者
curl -H "X-Argilla-API-Key: <ADMIN_KEY>" \
  "https://your-argilla-server.hf.space/api/v1/users" | python3 -m json.tool
```

> 新增完 user 後，記得在 Argilla Web UI 的 **Settings → Workspaces → argilla** 下將該使用者加入 workspace，否則看不到資料集。

---

## 相關文件

| 文件 | 說明 |
|---|---|
| `docs/reports/Safety Alignment Plan: Preference Dataset Taxonomy v5.md` | Harm Category 定義、Severity Levels、Seed Prompts 設計 |
| `docs/reports/Harmful Prompt Augmentation Progress Report v2.md` | Prompt 擴增結果（373 筆 harmful + 188 normal 的來由） |
| `docs/reports/Response Generation Progress Report v1.md` | 4 個模型回答生成流程說明 |
| `docs/reports/Reward and Cost Model Training Progress Report v1.md` | Cost Model 訓練（DeBERTa，71.9% pairwise accuracy） |
| `docs/Cell6_Cell9_Modification_Notes.md` | Notebook Cell 6/9 的 sub-topic 約束機制說明 |
| `test.ipynb` | 完整端到端流程（Step 1~15） |

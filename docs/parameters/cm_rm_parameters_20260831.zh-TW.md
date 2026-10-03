# Cost Model 與 Reward Model — 完整參數清單

**日期：** 2026-08-31
**範圍：** CM 與 RM 訓練流程目前程式碼實際讀取的所有可調輸入。此文件在 Airflow 遷移之前
產出，讓 DAG 的參數清單能夠建立在已驗證的數值上，而不是憑記憶填寫。

本文件的表格由 AST 解析兩支 trainer 的 `argparse` 定義自動生成，歷史執行欄位則直接讀取各次
執行的 `arguments.json`，並非人工謄寫。

---

## 如何閱讀這些表格

一個參數的最終生效值會經過**三層**，而這三層彼此不一致的情況比預期更常見：

| 層級 | 位置 | 說明 |
|---|---|---|
| 1. `argparse` 預設 | `parser.add_argument(..., default=)` | 直接執行 trainer 且不帶任何旗標時得到的值。其中數個已經過時 — 見發現 2。 |
| 2. wrapper 設定值 | `scripts/retrain_ministral3b_docker.sh`、`scripts/reward/run_reward_docker.sh` | production 實際傳入的值。幾乎所有參數都會覆蓋第 1 層。 |
| 3. 歷史執行值 | `<run_dir>/arguments.json` | 某次已完成執行實際使用的值。要重現已發表的數字時，這才是真正的依據。 |

**Airflow DAG 的預設值應取自第 2 或第 3 層，絕不可取自第 1 層。**

---

## 主要發現

### 發現 1 — RM 存在兩套分歧的 production 設定（需要決策）

並不存在單一的「RM 設定」。兩次真實完成的 RM 執行使用了實質不同的超參數，而設計規格書
只記載了其中第二套：

| 參數 | 六月執行 — helpfulness RM | 八月執行 — 客服 CS RM | `run_reward_docker.sh` 預設 |
|---|---|---|---|
| `max_length` | 4096 | 576 | **4096** |
| `batch_size` × `grad_accum` | 1 × 32 | 2 × 16 | **1 × 32** |
| `learning_rate` | 1e-5 | 5e-5 | **1e-5** |
| `weight_decay` | 1e-6 | 0.01 | **1e-6** |
| `epochs` | 3 | 5 | **3** |
| `regularization` | 0.001 | 0.01 | **0.001** |
| `lora_r` | *（不存在 — full FT）* | 16 | **0（full FT）** |
| `early_stopping_patience` | *（不存在）* | 2 | **0（停用）** |

證據：`reward_output/run_reward_byprompt_20260622_104203/arguments.json` 與
`reward_output/run_reward_cs_within_20260803/arguments.json`。

**為何重要。** `docs/superpowers/specs/2026-08-25-airflow-cm-rm-training-pipeline-design.md`
第 6 節將八月那一欄呈現為 `rm_train` 的預設值。但實際執行
`bash scripts/reward/run_reward_docker.sh` 且不帶任何環境變數覆蓋時，重現的是**六月**那一欄。
若有人依該節建立 DAG，得到的預設值既不符合 wrapper，也無法重現任何一次執行。

六月執行的 `arguments.json` 完全沒有 `lora_r` 這個鍵 — 該次執行早於 LoRA 旗標被加入 RM
trainer，因此它是「因為參數不存在」而成為 full fine-tuning，而非因為 `lora_r=0`。

**需要決策：** 哪一套才是 `rm_train` 的預設？兩者訓練的是目的不同的模型，因此誠實的答案
可能是 DAG 需要一個 `preset` 參數（`helpfulness` | `customer_cs`），而不是單一組數值。

### 發現 2 — CM trainer 的 `argparse` 預設值已經過時

`scripts/train_cost_model_v2.py` 仍保留 DeBERTa 時期的預設值。目前之所以無害，只是因為
wrapper 把它們全部覆蓋掉了：

| 旗標 | `argparse` 預設 | production 實際使用 |
|---|---|---|
| `--model-name-or-path` | `microsoft/deberta-v3-large` | `mistralai/Ministral-3-3B-Instruct-2512` |
| `--max-length` | `512` | `4096` |
| `--batch-size` | `4` | `1` |
| `--gradient-accumulation-steps` | `8` | `32` |

任何直接呼叫 trainer 的人 — 而這正是 Airflow task 自行執行 `docker run` 時會做的事 — 會
無聲地得到一組 DeBERTa 形狀的設定。在 DAG 繞過 shell wrapper 之前，這些預設值應先修正為
production 值。

RM trainer 沒有這個問題，其預設值已指向 Ministral-3-3B。

### 發現 3 — 沒有任何無用參數

全部 62 個旗標在各自 trainer 的程式主體中都有被引用，沒有需要刪除的參數。

### 發現 4 — 兩支 trainer 之間的命名不對稱

兩支 trainer 是各自演化的，旗標並未對齊。這一點很重要，因為 Airflow 遷移的目的正是要讓
兩者共用同一套參數 schema：

| 關注點 | Cost Model | Reward Model |
|---|---|---|
| 最佳 checkpoint 選擇 | `--save-best`（字串，逗號分隔指標） | `--save-best-only`（旗標） |
| 每個 epoch 存檔 | *（無 — 一律儲存）* | `--save-each-epoch`（`--/--no-`） |
| 半精度 | `--fp16` **與** `--bf16` | 僅 `--bf16` |
| 損失形狀 | `--loss-type {sequence-wise,token-wise}` | *（無 — 僅 Bradley-Terry）* |
| 分數正規化 | `--normalize-score-during-training`、`--normalizer-momentum` | *（無）* |
| Early stopping | *（無）* | `--early-stopping-patience`、`--early-stopping-min-delta` |
| Eval 預測輸出 | `--save-eval-predictions` | *（無）* |

有七項能力只存在於其中一邊。DAG 無法在不補齊缺少的旗標、或不將它們標記為模型專屬的情況下，
對兩者提供單一 schema。

---

## Cost Model — `scripts/train_cost_model_v2.py`

共 32 個命令列參數。production 進入點為
`scripts/retrain_ministral3b_docker.sh` → `scripts/trainer.py` → `train_cost_model_v2.main()`。

| 參數 | 型別 | argparse 預設 | wrapper 設定 | run 20260625 (best) | 用途 |
|---|---|---|---|---|---|
| `--model-name-or-path` | str | `microsoft/deberta-v3-large` | `$MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | 作為骨幹（backbone）的 HuggingFace 模型 id 或本地路徑，score head 會接在其上。 |
| `--dataset-path` | path | **required** | `$DATASET` | `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl` | 訓練資料（JSONL）。同時給定 --eval-dataset-path 時，此檔必須只含訓練資料；trainer 不會自動扣除 eval 列。 |
| `--output-dir` | path | **required** | `$RUN_DIR` | `cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best` | 執行目錄：checkpoints、arguments.json、training_log.json、run_manifest.json。 |
| `--max-length` | int | `512` | `$MAX_LENGTH` | `4096` | tokenizer 截斷長度。主導 VRAM 用量，是成本影響最大的單一參數。 |
| `--batch-size` | int | `4` | `$BATCH_SIZE` | `1` | 每個 optimizer step 的 micro-batch。有效批次 = batch_size x gradient_accumulation_steps。 |
| `--gradient-accumulation-steps` | int | `8` | `$GRAD_ACCUM` | `32` | 累積多少 step 後才更新一次 optimizer。 |
| `--learning-rate` | float | `1e-05` | `$LR` | `1e-05` | warmup 之後的峰值學習率。 |
| `--weight-decay` | float | `1e-06` | `1e-6` | `1e-06` | AdamW 的 weight decay。 |
| `--epochs` | int | `3` | `$EPOCHS` | `3` | 完整走過訓練集的次數。 |
| `--warmup-ratio` | float | `0.1` | `0.1` | `0.1` | 總步數中用於將學習率自 0 暖身的比例。 |
| `--regularization` | float | `0.001` | `0.001` | `0.001` | 對原始分數量值的 L2 懲罰，避免輸出漂移。 |
| `--eval-split-ratio` | float | `0.1` | `0.1` | `0.1` | 後備的內部保留比例，僅在未提供 --eval-dataset-path 時生效。 |
| `--eval-dataset-path` | path | `None` | — | — | 外部評估集。設定後 trainer 會跳過自身的內部切分。 |
| `--log-steps` | int | `10` | `10` | `10` | 每隔幾個 optimizer step 輸出一行紀錄。 |
| `--seed` | int | `42` | `$SEED` | `42` | shuffle、切分與初始化所用的亂數種子。 |
| `--pooling` | str | `mean` | `last-token` | `last-token` | 如何將 token 狀態壓成單一分數。production 使用 last-token。 Choices: ['mean', 'cls', 'last-token']. |
| `--loss-type` | str | `sequence-wise` | `sequence-wise` | `sequence-wise` | sequence-wise 對整段序列給一次分數；token-wise 則平均每個 token 的分數。 Choices: ['sequence-wise', 'token-wise']. |
| `--concat-forward` | flag | `False` | — | `False` | 將 pair 的 chosen 與 rejected 兩半併在同一次 forward 計算。 |
| `--normalize-score-during-training` | flag | `False` | — | `False` | 追蹤 running mean/var 並對輸出分數做正規化。 |
| `--normalizer-momentum` | float | `0.9` | — | `0.9` | 上述 running normalizer 的 EMA 動量。 |
| `--gradient-checkpointing` | flag | `False` | — | `False` | 在 backward 時重算 activation，以計算量換取 VRAM。 |
| `--fp16` | flag | `False` | — | `False` | 半精度 autocast。僅 CM 有，RM trainer 沒有此參數。 |
| `--bf16` | flag | `False` | — | `False` | bfloat16 autocast。在支援的硬體上優於 fp16。 |
| `--load-in-half` | flag | `False` | `set` | `True` | 直接以半精度載入骨幹權重，載入時記憶體需求減半。 |
| `--device-map` | flag | `False` | `set` | `True` | 交由 accelerate 將模型切分到可見的多張 GPU（device_map=auto）。 |
| `--adafactor` | flag | `False` | — | `False` | 改用 Adafactor 取代 AdamW，以降低 optimizer state 記憶體。 |
| `--save-eval-predictions` | flag | `False` | `set` | `True` | 輸出每筆樣本的 eval 分數，供錯誤分析。 |
| `--save-backbone` | flag (--x/--no-x) | `True` | `set` | `True` | 保存完整骨幹權重。--no-save-backbone 則只留 adapter 與 score head。 |
| `--save-best` | str | `` | — | `pairwise,loss` | 以逗號分隔的指標清單，據以追蹤 best-* checkpoint（CM）。 |
| `--lora-r` | int | `0` | `$LORA_R` | — | LoRA rank。設為 0 會完全停用 LoRA，該次執行即成為 full fine-tuning。 |
| `--lora-alpha` | int | `32` | `$LORA_ALPHA` | — | LoRA 縮放係數；實際縮放為 alpha/r。 |
| `--lora-dropout` | float | `0.05` | `$LORA_DROPOUT` | — | LoRA 路徑上的 dropout。 |

---

## Reward Model — `scripts/reward/train_reward_model.py`

共 30 個命令列參數。production 進入點為 `scripts/reward/run_reward_docker.sh`。
兩個執行欄位即為發現 1 所述的兩套分歧設定。

| 參數 | 型別 | argparse 預設 | wrapper 設定 | June (helpfulness) | Aug (customer CS) | 用途 |
|---|---|---|---|---|---|---|
| `--model-name-or-path` | str | `mistralai/Ministral-3-3B-Instruct-2512` | `$MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | `mistralai/Ministral-3-3B-Instruct-2512` | 作為骨幹（backbone）的 HuggingFace 模型 id 或本地路徑，score head 會接在其上。 |
| `--dataset-path` | path | **required** | `$TRAIN` | `datasets/reward/reward_train_byprompt.jsonl` | `datasets/reward/cs_within_train.jsonl` | 訓練資料（JSONL）。同時給定 --eval-dataset-path 時，此檔必須只含訓練資料；trainer 不會自動扣除 eval 列。 |
| `--eval-dataset-path` | path | `None` | `$EVAL` | `datasets/reward/reward_eval_byprompt.jsonl` | `datasets/reward/cs_within_validation.jsonl` | 外部評估集。設定後 trainer 會跳過自身的內部切分。 |
| `--output-dir` | path | **required** | `$RUN_DIR` | `reward_output/run_reward_byprompt_20260622_104203` | `reward_output/run_reward_cs_within_20260803` | 執行目錄：checkpoints、arguments.json、training_log.json、run_manifest.json。 |
| `--max-length` | int | `4096` | `$MAX_LENGTH` | `4096` | `576` | tokenizer 截斷長度。主導 VRAM 用量，是成本影響最大的單一參數。 |
| `--batch-size` | int | `1` | `$BATCH_SIZE` | `1` | `2` | 每個 optimizer step 的 micro-batch。有效批次 = batch_size x gradient_accumulation_steps。 |
| `--gradient-accumulation-steps` | int | `32` | `$GRAD_ACCUM` | `32` | `16` | 累積多少 step 後才更新一次 optimizer。 |
| `--learning-rate` | float | `1e-05` | `$LR` | `1e-05` | `5e-05` | warmup 之後的峰值學習率。 |
| `--weight-decay` | float | `1e-06` | `$WEIGHT_DECAY` | `1e-06` | `0.01` | AdamW 的 weight decay。 |
| `--epochs` | int | `3` | `$EPOCHS` | `3` | `5` | 完整走過訓練集的次數。 |
| `--warmup-ratio` | float | `0.1` | `0.1` | `0.1` | `0.1` | 總步數中用於將學習率自 0 暖身的比例。 |
| `--regularization` | float | `0.001` | `$REGULARIZATION` | `0.001` | `0.01` | 對原始分數量值的 L2 懲罰，避免輸出漂移。 |
| `--eval-split-ratio` | float | `0.1` | — | `0.1` | `0.1` | 後備的內部保留比例，僅在未提供 --eval-dataset-path 時生效。 |
| `--log-steps` | int | `10` | `10` | `10` | `10` | 每隔幾個 optimizer step 輸出一行紀錄。 |
| `--seed` | int | `42` | `$SEED` | `42` | `42` | shuffle、切分與初始化所用的亂數種子。 |
| `--pooling` | str | `last-token` | `last-token` | `last-token` | `last-token` | 如何將 token 狀態壓成單一分數。production 使用 last-token。 Choices: ['mean', 'cls', 'last-token']. |
| `--concat-forward` | flag | `False` | — | `False` | `False` | 將 pair 的 chosen 與 rejected 兩半併在同一次 forward 計算。 |
| `--gradient-checkpointing` | flag | `False` | — | `False` | `False` | 在 backward 時重算 activation，以計算量換取 VRAM。 |
| `--bf16` | flag | `False` | — | `False` | `False` | bfloat16 autocast。在支援的硬體上優於 fp16。 |
| `--load-in-half` | flag | `False` | `set` | `True` | `True` | 直接以半精度載入骨幹權重，載入時記憶體需求減半。 |
| `--device-map` | flag | `False` | `set` | `True` | `True` | 交由 accelerate 將模型切分到可見的多張 GPU（device_map=auto）。 |
| `--adafactor` | flag | `False` | — | `False` | `False` | 改用 Adafactor 取代 AdamW，以降低 optimizer state 記憶體。 |
| `--save-each-epoch` | flag (--x/--no-x) | `True` | `set` | `True` | `False` | 每個 epoch 結束後寫出一個 checkpoint（RM）。 |
| `--save-backbone` | flag (--x/--no-x) | `True` | `set` | `True` | `False` | 保存完整骨幹權重。--no-save-backbone 則只留 adapter 與 score head。 |
| `--save-best-only` | flag | `False` | — | — | `True` | 只保留最佳 checkpoint，而非每個 epoch 都存（RM）。 |
| `--lora-r` | int | `0` | `$LORA_R` | — | `16` | LoRA rank。設為 0 會完全停用 LoRA，該次執行即成為 full fine-tuning。 |
| `--lora-alpha` | int | `32` | `$LORA_ALPHA` | — | `32` | LoRA 縮放係數；實際縮放為 alpha/r。 |
| `--lora-dropout` | float | `0.05` | `$LORA_DROPOUT` | — | `0.05` | LoRA 路徑上的 dropout。 |
| `--early-stopping-patience` | int | `0` | `$EARLY_STOPPING_PATIENCE` | — | `2` | 容忍幾個 epoch 沒有進步才停止。0 表示停用 early stopping。 |
| `--early-stopping-min-delta` | float | `0.0` | — | — | `0.0` | 要被視為有進步所需的最小改善幅度。 |

---

## Shell wrapper 環境變數

這是目前操作者實際會設定的旋鈕。Airflow DAG 將取代這一層，因此其參數清單應涵蓋以下全部項目。

### `scripts/retrain_ministral3b_docker.sh`（Cost Model）— 15 個變數

| 變數 | 預設 | 作用 |
|---|---|---|
| `HF_CACHE` | `${HOME}/.cache/huggingface` | 掛載進容器的主機 HF 快取。 |
| `IMAGE` | `cost-model-trainer:v2` | Docker image。 |
| `MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | → `--model-name-or-path`。 |
| `DATASET` | `datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl` | → `--dataset-path`。 |
| `MAX_LENGTH` | `4096` | → `--max-length`。 |
| `BATCH_SIZE` | `1` | → `--batch-size`。 |
| `GRAD_ACCUM` | `32` | → `--gradient-accumulation-steps`。 |
| `LR` | `1e-5` | → `--learning-rate`。 |
| `EPOCHS` | `3` | → `--epochs`。 |
| `SEED` | `42` | → `--seed`。 |
| `LORA_R` | `0` | → `--lora-r`。大於 0 時另會在容器內觸發 `pip install peft`。 |
| `LORA_ALPHA` | `32` | → `--lora-alpha`。 |
| `LORA_DROPOUT` | `0.05` | → `--lora-dropout`。 |
| `RUN_DIR` | `cost_output/run_ministral_3b_instruct_<timestamp>` | 輸出目錄。 |
| `EXTRA_FLAGS` | *（空）* | 附加到 Python 呼叫末端的原始旗標。 |

注意：`weight_decay`、`regularization`、`warmup_ratio`、`eval_split_ratio`、`log_steps`、
`pooling`、`loss_type` 在此 wrapper 中是**寫死的**，除非編輯檔案或透過 `EXTRA_FLAGS`，
否則無法變更。DAG 應將它們正式開放為參數。

### `scripts/reward/run_reward_docker.sh`（Reward Model）— 20 個變數

| 變數 | 預設 | 作用 |
|---|---|---|
| `HF_CACHE` | `${HOME}/.cache/huggingface` | 主機 HF 快取掛載。 |
| `IMAGE` | `cost-model-trainer:v2` | Docker image。 |
| `MODEL` | `mistralai/Ministral-3-3B-Instruct-2512` | → `--model-name-or-path`。 |
| `SPLIT` | `byprompt` | 選擇資料集組合；`bypair` 為可與 CM 比較的版本。 |
| `TRAIN` | `datasets/reward/reward_train_${SPLIT}.jsonl` | → `--dataset-path`。 |
| `EVAL` | `datasets/reward/reward_eval_${SPLIT}.jsonl` | → `--eval-dataset-path`。 |
| `MAX_LENGTH` | `4096` | → `--max-length`。 |
| `BATCH_SIZE` | `1` | → `--batch-size`。 |
| `GRAD_ACCUM` | `32` | → `--gradient-accumulation-steps`。 |
| `LR` | `1e-5` | → `--learning-rate`。 |
| `EPOCHS` | `3` | → `--epochs`。 |
| `SEED` | `42` | → `--seed`。 |
| `WEIGHT_DECAY` | `1e-6` | → `--weight-decay`。 |
| `REGULARIZATION` | `0.001` | → `--regularization`。 |
| `LORA_R` | `0` | → `--lora-r`；大於 0 時另會觸發 `pip install peft`。 |
| `LORA_ALPHA` | `32` | → `--lora-alpha`。 |
| `LORA_DROPOUT` | `0.05` | → `--lora-dropout`。 |
| `EARLY_STOPPING_PATIENCE` | `0` | → `--early-stopping-patience`。 |
| `RUN_DIR` | `reward_output/run_reward_${SPLIT}_<timestamp>` | 輸出目錄。 |
| `EXTRA_FLAGS` | *（空）* | 附加到呼叫末端的原始旗標。 |

`SMOKE=1` 會覆蓋 `TRAIN`、`EVAL`、`MAX_LENGTH=256`、`GRAD_ACCUM=2`、`EPOCHS=1` 並改寫
`RUN_DIR`，把數小時的訓練變成數分鐘的接線測試。CM wrapper 沒有對應機制 — 建議補上，
因為 DAG 兩邊都需要一條便宜的驗證路徑。

`HOST_UID` / `HOST_GID`（皆預設 `1000`）僅用於容器結束後把執行目錄 `chown` 回主機使用者。

---

## 建議的 Airflow 參數預設值

以下取自第 2 層（wrapper）數值，因為那才是能重現今日執行結果的一組：

- **`cm_train`** — 全部取自上方 CM wrapper 表格。此外應將目前寫死的七個旗標
  （`weight_decay`、`regularization`、`warmup_ratio`、`eval_split_ratio`、`log_steps`、
  `pooling`、`loss_type`）提升為 DAG 參數。
- **`rm_train`** — **不要**任意選擇兩套設定其中之一。應開放一個 `preset` 參數，
  取值為 `helpfulness`（六月欄）與 `customer_cs`（八月欄），各自展開為完整參數組，
  以同時保留兩次可重現的執行。
- 在 DAG 直接呼叫 trainer（而非透過 shell wrapper）之前，先修正發現 2 所列的四個過時
  CM `argparse` 預設值。

## 重新產生本文件

```bash
python3 scripts/lib/param_inventory.py --lang zh
```

會依當前程式碼重新生成兩份表格。任何一支 trainer 的 `argparse` 區塊變動後都應重跑，
確保本清單不會與程式碼脫節。

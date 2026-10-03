# Reward / Cost Model 訓練(詞彙表,繁中版)

TAIWAN AI RAP 客服助理的安全對齊價值模型,採用 Safe RLHF(PKU-Alignment)方法論。先訓練兩個偏好模型,再分兩階段與聊天機器人整合:先做推論時的 gate-and-rank,之後做 PPO-Lagrange 微調。

(本檔為 `CONTEXT.md` 的繁中對照版;兩檔同步維護,以英文版為準。)

## 詞彙

**Actor**:
回應正在被對齊的聊天機器人 policy — 負責生成的模型;RM 與 CM 只負責評判。在 PPO-Lagrange 中是被訓練的對象;在 gate-and-rank 中是凍結的生成器。
_避免_:policy model、chatbot model(在正式機與替身之間有歧義)

**Production chatbot(正式聊天機器人)**:
實際部署的 TAIWAN AI RAP 助理:Mistral-Small-3.2-24B SFT(`...rap-nchc-iservice-0.5.0-ep5-fapm50`),經 Medusa API 提供服務。不在本機。
_避免_:the model、our model

**Stand-in actor(替身 actor)**:
本機開發用的 actor:Ministral-3-3B-Instruct-2512,因與 RM/CM backbone 同 tokenizer 家族而選用。
_避免_:test model、small model

**Customer model(客戶模型)**:
客戶自行微調的助理模型,也是 RM/CM 存在的服務對象。目前 checkpoint(`model/costomer_model`)為 8B `mistral_2512` 多模態版本,輸出帶 `[THINK]` 區塊;先前的 24B `...CS-3in1-FAPM50` checkpoint 屬同一血脈,是偏好資料中 R1/R2 候選的來源。
_避免_:production chatbot(那指特定 24B checkpoint,非整條血脈)、Model_A(是槽位標籤,不是模型)

**[THINK] block([THINK] 區塊)**:
客戶模型在答案前輸出的 `[THINK]...[/THINK]` 推理前綴。評分前一律剝除 — RM 與 CM 只看見可見答案,永遠不看推理內容。
_避免_:chain of thought(暗示推理本身被評分;此處是丟棄)

**Visible answer(可見答案)**:
客戶模型回應中 `[/THINK]` 之後的部分 — 使用者實際讀到的文字,也是 RM 與 CM 唯一評分的對象。

**Source signature(來源簽名)**:
回應中任何「由哪個模型撰寫」的風格痕跡,且該痕跡與偏好標籤相關。2026-03 偏好資料即帶有此問題:R1/R2 來自客戶模型,R3/R4 來自參考模型(gpt-oss-20b 或原版 Mistral-Small-3.2-24B,各約半數),而評審有 78% 偏好參考模型。學到此捷徑的 RM 在跨模型配對上表現良好,但在 gate-and-rank 實際面對的同模型配對上只有隨機水準。
_避免_:bias(過於籠統)、overfitting(這是對真實特徵走捷徑,不是記憶訓練集)

**Gate-and-rank**:
推論時整合:actor 生成 N 個候選回應,Cost Model 閘門剔除不安全者(score > 0),Reward Model 對安全存活者排名,回傳排名最高的回應。
_避免_:best-of-N(只描述排名那一半)、rejection sampling(只描述閘門那一半)

**PPO-Lagrange (PPO-Lag)**:
Safe RLHF 的訓練階段:在「預期 cost ≤ 門檻」的約束下微調 actor 以最大化 reward,兩者以學習得到的 Lagrange 乘數平衡。
_避免_:RLHF(不夠明確)、PPO(漏掉約束)

**Lagrange multiplier (λ)**:
PPO-Lagrange actor 目標中權衡安全與有用的學習純量。近期生成平均不安全時上升,安全時下降。
_避免_:safety weight、penalty coefficient(暗示手調;其實是學來的)

**Refusal fallback(拒答備援)**:
當 N 個候選全部未通過 Cost Model 閘門時,gate-and-rank 回傳的固定預寫安全回絕語。保證系統絕不輸出被 CM 標記為不安全的回應。「全數被擋率」作為一級指標追蹤。
_避免_:error message、default answer

**Episode cost**:
Cost Model 對一筆生成的 prompt+response 給出的純量分數。其移動平均與門檻的差驅動 Lagrange 乘數。
_避免_:cost loss、safety score

**Reward Model (RM)**:
對 prompt+response 評*有用性*分數的價值模型。以純 Bradley-Terry 成對損失 `-log σ(R(y_w) − R(y_l))` 訓練 — 不含安全正負號項。
_避免_:helpfulness model、preference model(有歧義)

**Cost Model (CM)**:
對 prompt+response 評*無害性*分數的價值模型(score < 0 = 安全,> 0 = 不安全)。以 PKU 三項損失訓練:Bradley-Terry 排序加上兩個 ±sign 項以校準絕對安全/不安全正負號。
_避免_:safety model、harm model

**Helpfulness ranking(有用性排名)**:
標註者對單一 prompt 的四個候選回應(R1–R4)給出的全序,最佳在前。`TAIWAN_AI_RAP_Helpfulness` 資料集的標註單位。
_避免_:rating、score、label

**Preference pair(偏好對)**:
`(prompt, y_w, y_l)` 三元組,`y_w` 為較受偏好(較有用)的回應,`y_l` 為較不受偏好者。Reward model 訓練資料由每筆有用性排名展開成偏好對而得。
_避免_:comparison、example

**Score head**:
接在 backbone 最後隱藏狀態上的單一線性層(`hidden_size → 1`),輸出純量 reward/cost。逐字複製自 PKU `safe_rlhf` 的 score-model 程式碼。
_避免_:classification head、value head

**Backbone**:
訓練期間全參數微調的預訓練 decoder(此處為 Ministral-3-3B-Instruct-2512)。在最後一個非 pad token 池化後餵入 score head。
_避免_:base model(與 "Base" vs "Instruct" checkpoint 變體混淆)

**Critic(reward critic / cost critic)**:
PPO-Lag 訓練期間用來預測 return 的逐 token 價值估計器;依作者配方由 RM/CM 溫啟動。與 RM/CM 本身不同 — 後者保持凍結、對整段回應評分。
_避免_:把 critic 稱作 "reward model"

**Value head**:
critic 的 `hidden_size → 1` 線性層,套用在**每個** token 位置(由 score head 初始化後參與訓練)。不是 score head — score head 只讀最後一個 token 且保持凍結。

**LoRA adapter**:
注入 attention/MLP 投影層的小型可訓練低秩矩陣;試點中 PPO-Lag 唯一更新的權重。adapter 停用 = 原始凍結模型(作為 PPO 的參考模型,也是評估中的「前」actor)。

**PPO-Lag pilot(試點)**:
2026-07-13 的階段 A 試行:在本機 2×V100 上,以 LoRA 對 3B 替身 actor 執行作者的 PPO-Lagrange,作為 24B 正式版上 HPC 前的配方預演。
_避免_:稱之為「階段 A 完成」— 它是預演,不是正式執行

# 0002 — 先以 gate-and-rank 整合 RM+CM;PPO-Lagrange 延後至 HPC

日期:2026-07-07
狀態:已採納

## 背景

Safe RLHF 管線的階段 2 已完成:可部署的 Reward Model
(`reward_output/run_reward_byprompt_20260622_104203/epoch1`,誠實 by-prompt
準確率 ~0.60)與 Cost Model
(`cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/`,sign
準確率 ~0.94+)。專案目標是用這兩個模型對齊 TAIWAN AI RAP 聊天機器人。
參考方法(PKU safe-rlhf 的 PPO-Lagrange,調查見
`docs/reports/07-07/safe_rlhf_ppo_lag_survey_20260707.md`)需要六個模型引擎
同時駐留、訓練其中三個;PKU 用 8×A800-80GB 跑 7B actor。本機硬體為
2×V100-32GB(共 64 GB,無 bf16)。正式聊天機器人是遠端服務的 24B 模型;
本機替身 actor 為 Ministral-3-3B-Instruct。

## 決策

分兩階段整合:

- **階段 B(現在,本機):**推論時 gate-and-rank — 凍結的 actor 抽樣 N 個
  候選,CM 閘門過濾(score > 0 者剔除),RM 對存活者排名。所有候選都被
  剔除時,回傳固定的拒答模板(refusal fallback),並將「全數被擋率」
  作為一級指標記錄。不改動任何模型權重。
- **階段 A(之後,HPC):**依參考實作執行 PPO-Lagrange,前提為
  (i) 取得足夠 GPU 記憶體的 HPC 配額,以及 (ii) RM 改善
  (需更多標註,見 06-22 報告結論)。

## 考慮過的替代方案

- **現在就在 V100 上跑 PPO-Lag:**全參數微調不可行(六個 3B 引擎光權重
  就 ≈36 GB,加上每個被訓練模型 ~48 GB 的 Adam 狀態);LoRA/offload 變體
  會偏離參考實作,而且仍得對抗 fp16 的不穩定性。
- **用現在的 RM 跑 PPO-Lag:**PPO 會強力最佳化 reward 訊號;~0.60 準確率
  的 RM 會招致 reward hacking。gate-and-rank 中,弱 RM 最多只能在*已經
  安全*的候選之間排錯序 — 退化得優雅。

## 後果

- 兩個模型現在就在服務時被端到端使用,為 mentor 報告產出示範與評估證據,
  無需 RL 基礎設施。
- 服務成本:每個回應需要 N 次生成 + 2N 次評分前向。
- CM 校準過的正負號(三項損失)在兩個階段都是承重結構:現在是閘門邊界,
  之後是約束門檻 0.0。
- CONTEXT.md 所述「作為之後 PPO-Lagrange 微調的基礎」仍然成立;本 ADR
  只是排定到達路徑的順序。

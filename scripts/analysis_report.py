import json
import re
from pathlib import Path
from collections import defaultdict

# 檢查常見中國用語 (PRC Terms)
PRC_TERMS = {
    "視頻", "軟件", "硬件", "屏幕", "支持",
    "內存", "程序", "服務器", "默認", "激活",
    " U盤", "分辨率", "鼠標", "智能手機", "打印機", "信息", "網絡",
    "項目", "接口", "質量"
}

# 判斷是單純拒絕還是帶解釋的拒絕
def analyze_refusal_style(text):
    text = text.replace('\n', '').strip()
    if any(k in text for k in ["抱歉", "無法", "不能", "對不起", "違反"]):
        if len(text) < 40 and not any(k in text for k in ["因為", "導致", "為了", "根據", "規定", "建議"]):
            return "直接拒絕 (Direct Refusal)"
        else:
            return "解釋拒絕 (Explanation Refusal)"
    return "非拒絕 (Non-refusal)"

def analyze():
    import os
    records_file = "records_api.json"
    if not os.path.exists(records_file):
        print(f"Error: {records_file} not found. Please run fetch_api.py first.")
        return
        
    with open(records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 統計指標
    stats = {
        "model_A_prc_terms": 0,
        "model_B_prc_terms": 0,
        "model_A_missing_chars": 0,
        "model_B_missing_chars": 0,
        "preferred_direct_refusal": 0,
        "preferred_explanation": 0,
        "preferred_non_refusal": 0
    }
    
    prc_findings = []
    missing_findings = []

    for record in records:
        fields = record.get("fields", {})
        prompt = fields.get("prompt", "")
        
        # 假設 R1, R2 來自 Model A, R3, R4 來自 Model B
        resps = {
            "R1": ("Model A", fields.get("R1", "")),
            "R2": ("Model A", fields.get("R2", "")),
            "R3": ("Model B", fields.get("R3", "")),
            "R4": ("Model B", fields.get("R4", ""))
        }
        
        for r_id, (model_name, r_text) in resps.items():
            if not r_text:
                continue
                
            # 1. 檢查中國用語
            found_prc = [term for term in PRC_TERMS if term in r_text]
            if found_prc:
                if model_name == "Model A": stats["model_A_prc_terms"] += 1
                else: stats["model_B_prc_terms"] += 1
                prc_findings.append({
                    "model": model_name,
                    "prompt": prompt,
                    "response": r_text,
                    "terms": found_prc
                })
                
            # 2. 檢查漏字或亂碼
            if "\ufffd" in r_text:
                if model_name == "Model A": stats["model_A_missing_chars"] += 1
                else: stats["model_B_missing_chars"] += 1
                missing_findings.append({
                    "model": model_name,
                    "prompt": prompt,
                    "response": r_text
                })

        # 3. Labeler 偏好分析
        responses_list = record.get("responses", [])
        
        # 尋找已完成的人工標註 (過濾掉未回答的狀況)
        user_response = next((r for r in responses_list if r.get("status") == "submitted"), None)
        
        # 如果 user 沒回答，我們這邊只關注「標註者有送出」的
        if user_response:
            values = user_response.get("values", {})
            helpfulness_ranking = values.get("helpfulness_ranking", {}).get("value", [])
            
            if helpfulness_ranking and len(helpfulness_ranking) > 0:
                # helpfulness_ranking 是一組 dict: [{"value": "R4", "rank": 1}, ...]
                # 我們要找 rank 最小的(第一名) 或者是陣列排序後的第一個
                top_choice_obj = min(helpfulness_ranking, key=lambda x: x.get("rank", 99))
                top_choice = top_choice_obj.get("value")
                top_text = resps.get(top_choice, ("", ""))[1]
                
                style = analyze_refusal_style(top_text)
                if style == "直接拒絕 (Direct Refusal)":
                    stats["preferred_direct_refusal"] += 1
                elif style == "解釋拒絕 (Explanation Refusal)":
                    stats["preferred_explanation"] += 1
                elif style == "非拒絕 (Non-refusal)":
                    stats["preferred_non_refusal"] += 1

    # 產出 Report
    report = f"""# RLHF Safety Alignment 分析報表

## 1. 數量與趨勢分析

### 簡體中國用語 (PRC Terms) 偵測
- **Model A (客服模型)** 出現次數: {stats['model_A_prc_terms']}
- **Model B** 出現次數: {stats['model_B_prc_terms']}

### 漏字與亂碼 () 偵測
- **Model A (客服模型)** 出現次數: {stats['model_A_missing_chars']}
- **Model B** 出現次數: {stats['model_B_missing_chars']}

---

## 2. Labeler 偏好分析 (拒答風格)
在使用者標註為 `completed` (已送出) 的回答中，被排在**第一名**的生成結果風格分佈如下：
- **直接拒絕 (Direct Refusal)** (較簡短，無特定解釋): {stats['preferred_direct_refusal']} 次
- **解釋拒絕 (Explanation Refusal)** (較長，帶有原因說明): {stats['preferred_explanation']} 次
- **非拒絕 (順從回答)**: {stats['preferred_non_refusal']} 次

> **分析結論**：由此可看出 Labeler 在面對潛在有害問題時，是不是更傾向模型給予充分的拒絕解釋，或者認為簡短拒絕較安全。

---

## 3. 具體案例列舉

### 包含簡體用語的案例 (最多顯示 5 筆)
"""
    for item in prc_findings[:5]:
        report += f"- **模型**: {item['model']}\\n"
        report += f"  - **Prompt**: {item['prompt'][:50]}...\\n"
        report += f"  - **命中用語**: `{', '.join(item['terms'])}`\\n"
        report += f"  - **Response Snippet**: {item['response'][:100]}...\\n\\n"

    report += "### 產生漏字/亂碼的案例 (最多顯示 5 筆)\\n"
    if missing_findings:
        for item in missing_findings[:5]:
            report += f"- **模型**: {item['model']}\\n"
            report += f"  - **Prompt**: {item['prompt'][:50]}...\\n"
            report += f"  - **Response Snippet**: {item['response'][:100]}...\\n\\n"
    else:
        report += "無發現明顯亂碼。\\n"

    with open("analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("分析報表已生成：analysis_report.md")

if __name__ == "__main__":
    analyze()

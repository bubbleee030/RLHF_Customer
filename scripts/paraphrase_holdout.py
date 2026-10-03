#!/usr/bin/env python3
"""
Paraphrase eval_dataset.jsonl using Gemma 4 API to create holdout evaluation data.

Usage:
    cd /home/ubuntu/reward_model
    python3 scripts/paraphrase_holdout.py
"""
import json
import time
import requests
import os
from pathlib import Path

API_KEY = os.environ["NCHC_API_KEY"]  # required; no default
BASE_URL = os.environ.get("NCHC_BASE_URL", "https://your-openai-compatible-endpoint/v1")
MODEL = "gemma-4-31B-it"

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

INPUT_FILE = Path("datasets/eval_dataset.jsonl")
OUTPUT_FILE = Path("datasets/eval_dataset_holdout_gemma4.jsonl")
PROGRESS_FILE = Path("datasets/eval_dataset_holdout_gemma4_progress.jsonl")


def call_api(messages: list, max_tokens: int = 2048, temperature: float = 0.7) -> str | None:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [Attempt {attempt+1}/3] API error: {e}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return None


def build_paraphrase_prompt(sample: dict) -> str:
    inp = sample["input"]
    ans = sample["answer"]
    other = sample["other_answer"]

    # For very long responses, truncate but note it
    MAX = 1200
    ans_trunc = ans[:MAX] + ("\n[...以下省略...]" if len(ans) > MAX else "")
    other_trunc = other[:MAX] + ("\n[...以下省略...]" if len(other) > MAX else "")

    return f"""請將以下三個欄位改寫成不同的措辭，但必須保留完全相同的語意與安全屬性。

規則：
1. 使用繁體中文（台灣用語），句型自然
2. 改變措辭/句型，但語意和安全性質絕對不變
3. 若有程式碼，可改變變數名稱或範例值，但邏輯/語言不變
4. 若是危險、違法或有害內容，改寫後仍必須保有相同的危害性（不要加入拒絕語句）
5. 只回傳 JSON，不要其他文字

原始內容：
INPUT: {inp}

ANSWER_A: {ans_trunc}

ANSWER_B: {other_trunc}

輸出格式（嚴格 JSON，key 名稱完全一致）：
{{
  "input": "<改寫後的使用者提問>",
  "answer": "<改寫後的回答A>",
  "other_answer": "<改寫後的回答B>"
}}"""


def parse_json_output(text: str) -> dict | None:
    """Try to extract JSON from the API response."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find JSON block in markdown
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.rindex("```")
            try:
                return json.loads(text[start:end].strip())
            except Exception:
                pass
    # Try to find { ... } block
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        pass
    return None


def main():
    with open(INPUT_FILE) as f:
        samples = [json.loads(l) for l in f]

    # Load existing progress
    done_indices = set()
    completed_results = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            for line in f:
                row = json.loads(line)
                idx = row["_eval_idx"]
                done_indices.add(idx)
                completed_results[idx] = row
        print(f"Resuming: {len(done_indices)}/{len(samples)} already done")

    total = len(samples)
    failed = []

    with open(PROGRESS_FILE, "a") as prog_f:
        for i, sample in enumerate(samples):
            idx = sample["_eval_idx"]
            if idx in done_indices:
                continue

            print(f"[{i+1}/{total}] idx={idx}  is_safe={sample['is_safe']}, is_other_safe={sample['is_other_safe']}")
            print(f"  input: {sample['input'][:60]}...")

            prompt = build_paraphrase_prompt(sample)
            messages = [{"role": "user", "content": prompt}]

            raw = call_api(messages, max_tokens=2048, temperature=0.8)

            if raw is None:
                print(f"  ❌ API failed, using original")
                result = {k: sample[k] for k in ("input", "answer", "other_answer")}
                result["_paraphrased"] = False
                result["_paraphrase_failed"] = True
                failed.append(idx)
            else:
                parsed = parse_json_output(raw)
                if parsed and "input" in parsed and "answer" in parsed and "other_answer" in parsed:
                    result = {
                        "input": parsed["input"],
                        "answer": parsed["answer"],
                        "other_answer": parsed["other_answer"],
                        "_paraphrased": True,
                        "_paraphrase_failed": False,
                    }
                    print(f"  ✅ input: {parsed['input'][:60]}...")
                else:
                    print(f"  ⚠️  JSON parse failed, using original. Raw: {raw[:100]}")
                    result = {k: sample[k] for k in ("input", "answer", "other_answer")}
                    result["_paraphrased"] = False
                    result["_paraphrase_failed"] = True
                    failed.append(idx)

            # Preserve all original label fields
            for key in ("safer", "is_safe", "is_other_safe", "_eval_idx"):
                result[key] = sample[key]

            prog_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            prog_f.flush()
            done_indices.add(idx)

            # Rate limiting
            time.sleep(1.0)

    # Merge progress into final output (ordered by _eval_idx)
    all_results = {}
    with open(PROGRESS_FILE) as f:
        for line in f:
            row = json.loads(line)
            all_results[row["_eval_idx"]] = row

    with open(OUTPUT_FILE, "w") as f:
        for idx in sorted(all_results.keys()):
            f.write(json.dumps(all_results[idx], ensure_ascii=False) + "\n")

    n_success = sum(1 for r in all_results.values() if r.get("_paraphrased"))
    print(f"\n{'='*60}")
    print(f"完成！成功改寫: {n_success}/{total}")
    print(f"失敗回退原始: {len(failed)}")
    print(f"輸出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

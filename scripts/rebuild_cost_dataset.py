#!/usr/bin/env python3
"""Rebuild cost_dataset_for_safe_rlhf.jsonl from source data with quality fixes.

Fixes applied:
1. Skip pairs where answer == other_answer (identical text)
2. Deduplicate (input, answer, other_answer) triples
3. Validate safer_sign >= unsafer_sign (no label contradictions)
4. Print detailed data quality report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def _hash_triple(inp: str, ans: str, other: str) -> str:
    """Create a deterministic hash for deduplication."""
    raw = f"{inp}\x00{ans}\x00{other}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_pointwise(input_file: Path) -> tuple[list[dict], dict]:
    """Parse pointwise cost dataset and create pairwise samples.

    For harmful prompts: pair safe (s=1) responses with unsafe (s=-1) responses.
    For non-harmful prompts: pair consecutive safe responses together.
    """
    stats = {
        "total_records": 0,
        "unique_prompts": 0,
        "harmful_prompts": 0,
        "non_harmful_prompts": 0,
        "safe_responses": 0,
        "unsafe_responses": 0,
        "pairs_created": 0,
        "identical_text_skipped": 0,
        "no_contrast_skipped": 0,
    }

    groups: dict[str, list[dict]] = defaultdict(list)

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                groups[data["x"]].append(data)
                stats["total_records"] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    stats["unique_prompts"] = len(groups)
    results = []

    for x, items in groups.items():
        is_harmful_str = items[0].get("is_harmful", "No")
        is_harmful = is_harmful_str == "Yes"

        safe_ones = [d for d in items if d.get("s") == 1]
        unsafe_ones = [d for d in items if d.get("s") == -1]
        stats["safe_responses"] += len(safe_ones)
        stats["unsafe_responses"] += len(unsafe_ones)

        if is_harmful:
            stats["harmful_prompts"] += 1
            if safe_ones and unsafe_ones:
                for s_item in safe_ones:
                    for u_item in unsafe_ones:
                        safe_text = s_item["y"].strip()
                        unsafe_text = u_item["y"].strip()

                        # Fix 1: Skip identical text pairs
                        if safe_text == unsafe_text:
                            stats["identical_text_skipped"] += 1
                            continue

                        results.append({
                            "input": x,
                            "answer": safe_text,
                            "other_answer": unsafe_text,
                            "safer": True,
                            "is_safe": True,
                            "is_other_safe": False,
                        })
                        stats["pairs_created"] += 1
            else:
                stats["no_contrast_skipped"] += 1
        else:
            stats["non_harmful_prompts"] += 1
            if len(safe_ones) >= 2:
                for i in range(0, len(safe_ones) - 1, 2):
                    text_a = safe_ones[i]["y"].strip()
                    text_b = safe_ones[i + 1]["y"].strip()

                    # Fix 1: Skip identical text pairs
                    if text_a == text_b:
                        stats["identical_text_skipped"] += 1
                        continue

                    results.append({
                        "input": x,
                        "answer": text_a,
                        "other_answer": text_b,
                        "safer": True,
                        "is_safe": True,
                        "is_other_safe": True,
                    })
                    stats["pairs_created"] += 1

    return results, stats


def parse_pairwise(input_file: Path) -> tuple[list[dict], dict]:
    """Parse pairwise cost dataset (y_w / y_l format)."""
    stats = {
        "total_records": 0,
        "pairs_created": 0,
        "identical_text_skipped": 0,
    }

    results = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total_records"] += 1

            prompt = data.get("x", "").strip()
            y_w = data.get("y_w", "").strip()
            y_l = data.get("y_l", "").strip()
            is_harmful_str = data.get("is_harmful", "No")
            is_harmful = is_harmful_str == "Yes"

            if not prompt or not y_w or not y_l:
                continue

            # Fix 1: Skip identical text pairs
            if y_w == y_l:
                stats["identical_text_skipped"] += 1
                continue

            # For pairwise data:
            # y_w is the "better" (winner) response
            # y_l is the "worse" (loser) response
            # If the prompt is harmful:
            #   - y_w (winner) is more likely a safe refusal → is_safe=True
            #   - y_l (loser) is more likely an unsafe compliance → is_other_safe=False
            # If not harmful:
            #   - Both are likely safe, y_w is just better quality
            results.append({
                "input": prompt,
                "answer": y_w,
                "other_answer": y_l,
                "safer": True,
                "is_safe": True,
                "is_other_safe": not is_harmful,
            })
            stats["pairs_created"] += 1

    return results, stats


def deduplicate(pairs: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by (input, answer, other_answer) triple."""
    seen = set()
    deduped = []
    dup_count = 0

    for pair in pairs:
        h = _hash_triple(pair["input"], pair["answer"], pair["other_answer"])
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        deduped.append(pair)

    return deduped, dup_count


def validate_safety_signs(pairs: list[dict]) -> tuple[list[dict], int]:
    """Validate that safer_sign >= unsafer_sign.

    safer_sign  = 2*int(is_safe) - 1       → +1 if safe, -1 if unsafe
    unsafer_sign = 2*int(is_other_safe) - 1 → +1 if safe, -1 if unsafe

    If safer=True, then answer is the safer one.
    The safer answer's sign must be >= the unsafer answer's sign.
    i.e., is_safe >= is_other_safe (True >= False, True >= True, but NOT False < True)
    """
    valid = []
    contradiction_count = 0

    for pair in pairs:
        is_safe = pair["is_safe"]
        is_other_safe = pair["is_other_safe"]
        safer = pair["safer"]

        if safer:
            safer_sign = 2 * int(is_safe) - 1
            unsafer_sign = 2 * int(is_other_safe) - 1
        else:
            safer_sign = 2 * int(is_other_safe) - 1
            unsafer_sign = 2 * int(is_safe) - 1

        if safer_sign < unsafer_sign:
            contradiction_count += 1
            continue

        valid.append(pair)

    return valid, contradiction_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild cost dataset for safe-rlhf with quality fixes"
    )
    parser.add_argument(
        "--pointwise-file",
        type=Path,
        default=Path("./datasets/cost/cost_model_dataset_pointwise.jsonl"),
    )
    parser.add_argument(
        "--pairwise-file",
        type=Path,
        default=Path("./datasets/cost/cost_model_dataset_pairwise.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("./datasets/cost/cost_dataset_for_safe_rlhf_clean.jsonl"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("REBUILDING COST DATASET FOR SAFE-RLHF")
    print("=" * 60)

    all_pairs: list[dict] = []

    # --- Pointwise source ---
    if args.pointwise_file.exists():
        pw_pairs, pw_stats = parse_pointwise(args.pointwise_file)
        all_pairs.extend(pw_pairs)
        print(f"\n📊 Pointwise Source: {args.pointwise_file}")
        for k, v in pw_stats.items():
            print(f"   {k}: {v}")
    else:
        print(f"⚠️  Pointwise file not found: {args.pointwise_file}")

    # --- Pairwise source ---
    if args.pairwise_file.exists():
        pr_pairs, pr_stats = parse_pairwise(args.pairwise_file)
        all_pairs.extend(pr_pairs)
        print(f"\n📊 Pairwise Source: {args.pairwise_file}")
        for k, v in pr_stats.items():
            print(f"   {k}: {v}")
    else:
        print(f"⚠️  Pairwise file not found: {args.pairwise_file}")

    print(f"\n--- Pre-dedup total: {len(all_pairs)} pairs ---")

    # --- Deduplication ---
    all_pairs, dup_count = deduplicate(all_pairs)
    print(f"🔄 Duplicates removed: {dup_count}")
    print(f"   After dedup: {len(all_pairs)} pairs")

    # --- Validation ---
    all_pairs, contradiction_count = validate_safety_signs(all_pairs)
    print(f"❌ Contradictions removed: {contradiction_count}")
    print(f"   After validation: {len(all_pairs)} pairs")

    # --- Summary statistics ---
    both_safe = sum(1 for p in all_pairs if p["is_safe"] and p["is_other_safe"])
    safe_unsafe = sum(1 for p in all_pairs if p["is_safe"] and not p["is_other_safe"])
    unique_prompts = len(set(p["input"] for p in all_pairs))

    print(f"\n{'=' * 60}")
    print(f"FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"   Total pairs: {len(all_pairs)}")
    print(f"   Unique prompts: {unique_prompts}")
    print(f"   Both safe (is_safe=T, is_other_safe=T): {both_safe}")
    print(f"   Safe vs Unsafe (is_safe=T, is_other_safe=F): {safe_unsafe}")

    # --- Write output ---
    out_dir = args.output_file.parent
    if out_dir != Path("."):
        out_dir.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✅ Written to: {args.output_file}")

    # --- Final verification ---
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION")
    print(f"{'=' * 60}")

    # Re-check: no identical pairs
    identical = sum(
        1 for p in all_pairs if p["answer"].strip() == p["other_answer"].strip()
    )
    print(f"   Identical answer/other_answer: {identical} (should be 0)")

    # Re-check: no duplicates
    seen = set()
    dups = 0
    for p in all_pairs:
        h = _hash_triple(p["input"], p["answer"], p["other_answer"])
        if h in seen:
            dups += 1
        seen.add(h)
    print(f"   Duplicate triples: {dups} (should be 0)")

    # Re-check: no contradictions
    contras = 0
    for p in all_pairs:
        safer_sign = 2 * int(p["is_safe"]) - 1
        unsafer_sign = 2 * int(p["is_other_safe"]) - 1
        if p["safer"] and safer_sign < unsafer_sign:
            contras += 1
    print(f"   Label contradictions: {contras} (should be 0)")

    assert identical == 0, "Still have identical pairs!"
    assert dups == 0, "Still have duplicate triples!"
    assert contras == 0, "Still have label contradictions!"
    print("\n✅ All checks passed!")


if __name__ == "__main__":
    main()

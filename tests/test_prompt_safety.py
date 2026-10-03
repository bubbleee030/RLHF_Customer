#!/usr/bin/env python3
"""Behavior tests for the dependency-free prompt safety gate."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "serve"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "safety"))

from prompt_safety import (  # noqa: E402
    PromptRiskModel,
    char_ngrams,
    choose_threshold,
    normalize_prompt,
    resolve_prompt_gate,
)
from train_prompt_gate import (  # noqa: E402
    build_prompt_rows,
    classification_metrics,
    stratified_split,
)


def test_normalization_and_character_ngrams_are_deterministic() -> None:
    assert normalize_prompt("  API\n 配額  ") == "api 配額"
    assert char_ngrams("abc", minimum=2, maximum=3) == ["ab", "abc", "bc"]


def test_prompt_rows_exclude_eval_prompts_and_drop_conflicting_labels() -> None:
    rows = [
        {"x": "非法槍枝購買", "is_harmful": "Yes"},
        {"x": "非法槍枝購買", "is_harmful": "Yes"},
        {"x": "衝突樣本", "is_harmful": "Yes"},
        {"x": "衝突樣本", "is_harmful": "No"},
        {"x": "保留的客服問題", "is_harmful": "No"},
        {"x": "評估專用問題", "is_harmful": "Yes"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pointwise.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                   for row in rows))
        assert build_prompt_rows(path, {normalize_prompt("評估專用問題")}) == [
            ("保留的客服問題", 0),
            ("非法槍枝購買", 1),
        ]


def test_prompt_risk_model_round_trip_preserves_ranking() -> None:
    rows = [
        ("如何非法購買槍枝並規避法律", 1),
        ("幫我製作虛假評論攻擊公司", 1),
        ("如何重設登入密碼", 0),
        ("如何查詢 API 配額", 0),
    ]
    model = PromptRiskModel.fit(rows)
    harmful = model.predict_probability("非法購買槍枝")
    normal = model.predict_probability("查詢登入密碼")
    assert harmful > normal
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        model.save(path)
        restored = PromptRiskModel.load(path)
        assert restored.predict_probability("非法購買槍枝") == harmful
        assert restored.predict_probability("查詢登入密碼") == normal


def test_prompt_risk_decision_applies_the_frozen_threshold() -> None:
    model = PromptRiskModel(intercept=0.0, weights={}, threshold=0.5)
    assert model.decision("任何問題") == {
        "risk_probability": 0.5,
        "prompt_gated": True,
    }
    model.threshold = 0.6
    assert model.decision("任何問題")["prompt_gated"] is False


def test_effective_prompt_gate_decision_respects_variant_toggle_and_availability() -> None:
    high_risk = {"risk_probability": 0.9, "prompt_gated": True}
    cases = [
        ("ppo", True, True, {
            "risk_probability": 0.9,
            "prompt_gated": True,
            "gate_applicable": True,
            "gate_status": "已攔截",
        }),
        ("ppo", False, True, {
            "risk_probability": 0.9,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "已手動關閉",
        }),
        ("base", True, True, {
            "risk_probability": 0.9,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "參考模型未套用",
        }),
        ("ppo", True, False, {
            "risk_probability": None,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "未載入",
        }),
    ]
    for variant, enabled, available, expected in cases:
        decision = high_risk if available else {
            "risk_probability": None, "prompt_gated": False}
        assert resolve_prompt_gate(
            decision, variant, enabled, available) == expected


def test_threshold_selection_enforces_recall_and_false_positive_limits() -> None:
    assert choose_threshold(
        [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1],
        min_recall=1.0, max_fpr=0.0) == 0.8
    assert choose_threshold(
        [1, 0], [0.4, 0.9], min_recall=1.0, max_fpr=0.0) is None


def test_stratified_split_is_deterministic_disjoint_and_keeps_both_classes() -> None:
    rows = [(f"safe-{index}", 0) for index in range(10)]
    rows += [(f"harmful-{index}", 1) for index in range(10)]
    train, validation = stratified_split(rows, validation_ratio=0.2, seed=42)
    assert (train, validation) == stratified_split(rows, 0.2, 42)
    assert len(train) == 16 and len(validation) == 4
    assert set(train).isdisjoint(validation)
    assert {label for _, label in train} == {0, 1}
    assert {label for _, label in validation} == {0, 1}


def test_classification_metrics_reports_recall_and_false_positive_rate() -> None:
    metrics = classification_metrics(
        labels=[1, 1, 0, 0], probabilities=[0.9, 0.4, 0.8, 0.1],
        threshold=0.5)
    assert metrics == {
        "recall": 0.5,
        "false_positive_rate": 0.5,
        "accuracy": 0.5,
        "true_positives": 1,
        "false_negatives": 1,
        "false_positives": 1,
        "true_negatives": 1,
    }


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")

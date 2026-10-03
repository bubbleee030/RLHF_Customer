#!/usr/bin/env python3
"""Dependency-free character-ngram prompt-risk classifier."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def char_ngrams(text: str, minimum: int = 2, maximum: int = 5) -> list[str]:
    compact = normalize_prompt(text)
    return sorted({compact[index:index + size]
                   for size in range(minimum, maximum + 1)
                   for index in range(max(0, len(compact) - size + 1))})


def resolve_prompt_gate(risk_decision: dict, variant: str, enabled: bool,
                        gate_available: bool) -> dict:
    """Resolve the effective demo gate without changing its frozen risk score."""
    if not gate_available:
        return {
            "risk_probability": None,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "未載入",
        }
    probability = risk_decision["risk_probability"]
    if variant != "ppo":
        return {
            "risk_probability": probability,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "參考模型未套用",
        }
    if not enabled:
        return {
            "risk_probability": probability,
            "prompt_gated": False,
            "gate_applicable": False,
            "gate_status": "已手動關閉",
        }
    gated = bool(risk_decision["prompt_gated"])
    return {
        "risk_probability": probability,
        "prompt_gated": gated,
        "gate_applicable": True,
        "gate_status": "已攔截" if gated else "未攔截",
    }


@dataclass
class PromptRiskModel:
    intercept: float
    weights: dict[str, float]
    threshold: float = 0.5
    minimum_ngram: int = 2
    maximum_ngram: int = 5

    @classmethod
    def fit(cls, rows: list[tuple[str, int]], alpha: float = 1.0) -> "PromptRiskModel":
        class_counts = Counter(label for _, label in rows)
        if class_counts[0] == 0 or class_counts[1] == 0:
            raise ValueError("prompt-risk training requires both classes")
        document_counts = {0: Counter(), 1: Counter()}
        for prompt, label in rows:
            document_counts[label].update(set(char_ngrams(prompt)))
        vocabulary = set(document_counts[0]) | set(document_counts[1])
        total_documents = len(rows)
        intercept = math.log(class_counts[1] / total_documents)
        intercept -= math.log(class_counts[0] / total_documents)
        weights: dict[str, float] = {}
        for gram in vocabulary:
            p1 = (document_counts[1][gram] + alpha) / (class_counts[1] + 2 * alpha)
            p0 = (document_counts[0][gram] + alpha) / (class_counts[0] + 2 * alpha)
            intercept += math.log1p(-p1) - math.log1p(-p0)
            weights[gram] = (
                math.log(p1) - math.log1p(-p1)
                - math.log(p0) + math.log1p(-p0)
            )
        return cls(intercept=intercept, weights=weights)

    def predict_probability(self, prompt: str) -> float:
        score = self.intercept + sum(
            self.weights.get(gram, 0.0)
            for gram in set(char_ngrams(
                prompt, self.minimum_ngram, self.maximum_ngram)))
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 700.0)))
        exp_score = math.exp(max(score, -700.0))
        return exp_score / (1.0 + exp_score)

    def decision(self, prompt: str) -> dict:
        probability = self.predict_probability(prompt)
        return {
            "risk_probability": probability,
            "prompt_gated": probability >= self.threshold,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": 1,
            "intercept": self.intercept,
            "weights": dict(sorted(self.weights.items())),
            "threshold": self.threshold,
            "minimum_ngram": self.minimum_ngram,
            "maximum_ngram": self.maximum_ngram,
        }, ensure_ascii=False, separators=(",", ":")))

    @classmethod
    def load(cls, path: Path | str) -> "PromptRiskModel":
        data = json.loads(Path(path).read_text())
        if data.get("format_version") != 1:
            raise ValueError("unsupported prompt-risk model format")
        return cls(
            intercept=float(data["intercept"]),
            weights={key: float(value) for key, value in data["weights"].items()},
            threshold=float(data["threshold"]),
            minimum_ngram=int(data["minimum_ngram"]),
            maximum_ngram=int(data["maximum_ngram"]),
        )


def choose_threshold(labels: list[int], probabilities: list[float],
                     min_recall: float = 0.90,
                     max_fpr: float = 0.10) -> float | None:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("threshold selection requires both classes")
    valid: list[tuple[float, float, float]] = []
    for threshold in sorted(set(probabilities)):
        predicted = [probability >= threshold for probability in probabilities]
        recall = sum(p and label == 1 for p, label in zip(predicted, labels)) / positives
        fpr = sum(p and label == 0 for p, label in zip(predicted, labels)) / negatives
        if recall >= min_recall and fpr <= max_fpr:
            valid.append((recall, fpr, threshold))
    if not valid:
        return None
    return min(valid, key=lambda item: (-item[0], item[1], item[2]))[2]

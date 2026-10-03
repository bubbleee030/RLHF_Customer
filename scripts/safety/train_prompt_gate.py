#!/usr/bin/env python3
"""Train the prompt-risk gate from existing pointwise prompt labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from prompt_safety import (  # noqa: E402
    PromptRiskModel,
    choose_threshold,
    normalize_prompt,
)


def build_prompt_rows(pointwise_path: Path,
                      excluded_prompts: set[str]) -> list[tuple[str, int]]:
    excluded = {normalize_prompt(prompt) for prompt in excluded_prompts}
    labels_by_prompt: dict[str, set[int]] = defaultdict(set)
    display: dict[str, str] = {}
    for line in pointwise_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("x", "")).strip()
        normalized = normalize_prompt(prompt)
        if not normalized or normalized in excluded:
            continue
        raw_label = str(row.get("is_harmful", "")).strip().lower()
        if raw_label not in {"yes", "no"}:
            continue
        labels_by_prompt[normalized].add(int(raw_label == "yes"))
        display.setdefault(normalized, prompt)
    return sorted((display[prompt], next(iter(labels)))
                  for prompt, labels in labels_by_prompt.items()
                  if len(labels) == 1)


def stratified_split(rows: list[tuple[str, int]], validation_ratio: float,
                     seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)
    if set(by_label) != {0, 1}:
        raise ValueError("stratified split requires both classes")
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    for label in (0, 1):
        ordered = sorted(by_label[label], key=lambda row: hashlib.sha256(
            f"{seed}\0{normalize_prompt(row[0])}".encode()).digest())
        n_validation = max(1, round(len(ordered) * validation_ratio))
        n_validation = min(n_validation, len(ordered) - 1)
        validation.extend(ordered[:n_validation])
        train.extend(ordered[n_validation:])
    return sorted(train), sorted(validation)


def classification_metrics(labels: list[int], probabilities: list[float],
                           threshold: float) -> dict:
    predicted = [probability >= threshold for probability in probabilities]
    tp = sum(prediction and label == 1
             for prediction, label in zip(predicted, labels))
    fn = sum(not prediction and label == 1
             for prediction, label in zip(predicted, labels))
    fp = sum(prediction and label == 0
             for prediction, label in zip(predicted, labels))
    tn = sum(not prediction and label == 0
             for prediction, label in zip(predicted, labels))
    return {
        "recall": tp / (tp + fn),
        "false_positive_rate": fp / (fp + tn),
        "accuracy": (tp + tn) / len(labels),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
    }


def load_excluded_prompts(paths: list[Path]) -> set[str]:
    prompts: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
        else:
            payload = json.loads(path.read_text())
            rows = payload.get("records", []) if isinstance(payload, dict) else payload
        for row in rows:
            prompt = row.get("input") or row.get("prompt") or row.get("x")
            if prompt:
                prompts.add(normalize_prompt(prompt))
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--max-fpr", type=float, default=0.1)
    args = parser.parse_args()

    excluded = load_excluded_prompts(args.exclude)
    rows = build_prompt_rows(args.pointwise, excluded)
    train, validation = stratified_split(rows, args.validation_ratio, args.seed)
    model = PromptRiskModel.fit(train)
    labels = [label for _, label in validation]
    probabilities = [model.predict_probability(prompt) for prompt, _ in validation]
    threshold = choose_threshold(
        labels, probabilities, args.min_recall, args.max_fpr)
    accepted = threshold is not None
    model.threshold = threshold if accepted else 0.5
    metrics = classification_metrics(labels, probabilities, model.threshold)
    result = {
        "accepted": accepted,
        "threshold": threshold,
        "requirements": {"min_recall": args.min_recall,
                         "max_false_positive_rate": args.max_fpr},
        "metrics": metrics,
        "num_source_rows": len(rows),
        "num_train": len(train),
        "num_validation": len(validation),
        "num_excluded_prompts": len(excluded),
        "train_class_counts": {
            "safe": sum(label == 0 for _, label in train),
            "harmful": sum(label == 1 for _, label in train),
        },
        "validation_class_counts": {
            "safe": sum(label == 0 for _, label in validation),
            "harmful": sum(label == 1 for _, label in validation),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model.json")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

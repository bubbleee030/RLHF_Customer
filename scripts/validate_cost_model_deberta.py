#!/usr/bin/env python3
"""Validate a trained DeBERTa cost model on a held-out split."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PairSample:
    prompt: str
    safe_response: str
    unsafe_response: str


class PairwiseCostDataset(Dataset[PairSample]):
    def __init__(self, dataset_path: Path) -> None:
        self.samples: list[PairSample] = []
        self.skipped = 0

        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    self.skipped += 1
                    continue

                prompt = str(row.get("input", "")).strip()
                answer = str(row.get("answer", "")).strip()
                other = str(row.get("other_answer", "")).strip()
                safer = row.get("safer", None)

                if not prompt or not answer or not other:
                    self.skipped += 1
                    continue

                safe_response: str | None = None
                unsafe_response: str | None = None

                if isinstance(safer, bool):
                    if safer:
                        safe_response, unsafe_response = answer, other
                    else:
                        safe_response, unsafe_response = other, answer
                else:
                    is_safe = row.get("is_safe", None)
                    is_other_safe = row.get("is_other_safe", None)
                    if isinstance(is_safe, bool) and isinstance(is_other_safe, bool) and is_safe != is_other_safe:
                        if is_safe:
                            safe_response, unsafe_response = answer, other
                        else:
                            safe_response, unsafe_response = other, answer

                if safe_response is None or unsafe_response is None:
                    self.skipped += 1
                    continue

                self.samples.append(
                    PairSample(
                        prompt=prompt,
                        safe_response=safe_response,
                        unsafe_response=unsafe_response,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PairSample:
        return self.samples[idx]


class DebertaCostModel(nn.Module):
    def __init__(self, backbone_path: str | Path) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(str(backbone_path))
        hidden_size = self.backbone.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.score_head(pooled).squeeze(-1)

    def forward(
        self,
        safe_input_ids: torch.Tensor,
        safe_attention_mask: torch.Tensor,
        unsafe_input_ids: torch.Tensor,
        unsafe_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_scores = self._encode(safe_input_ids, safe_attention_mask)
        unsafe_scores = self._encode(unsafe_input_ids, unsafe_attention_mask)
        return safe_scores, unsafe_scores


class PairCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _format_text(prompt: str, response: str) -> str:
        return f"User: {prompt}\nAssistant: {response}"

    def __call__(self, batch: list[PairSample]) -> dict[str, torch.Tensor]:
        safe_texts = [self._format_text(sample.prompt, sample.safe_response) for sample in batch]
        unsafe_texts = [self._format_text(sample.prompt, sample.unsafe_response) for sample in batch]

        safe_encoded = self.tokenizer(
            safe_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        unsafe_encoded = self.tokenizer(
            unsafe_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "safe_input_ids": safe_encoded["input_ids"],
            "safe_attention_mask": safe_encoded["attention_mask"],
            "unsafe_input_ids": unsafe_encoded["input_ids"],
            "unsafe_attention_mask": unsafe_encoded["attention_mask"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DeBERTa cost model")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_subset(
    model: DebertaCostModel,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for batch in pbar:
            safe_input_ids = batch["safe_input_ids"].to(device, non_blocking=True)
            safe_attention_mask = batch["safe_attention_mask"].to(device, non_blocking=True)
            unsafe_input_ids = batch["unsafe_input_ids"].to(device, non_blocking=True)
            unsafe_attention_mask = batch["unsafe_attention_mask"].to(device, non_blocking=True)

            safe_scores, unsafe_scores = model(
                safe_input_ids=safe_input_ids,
                safe_attention_mask=safe_attention_mask,
                unsafe_input_ids=unsafe_input_ids,
                unsafe_attention_mask=unsafe_attention_mask,
            )

            loss = F.softplus(safe_scores - unsafe_scores)
            correct = (unsafe_scores > safe_scores).float()

            total_loss += loss.sum().item()
            total_correct += correct.sum().item()
            total_count += int(loss.numel())

    if total_count == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "num_samples": 0}

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "num_samples": total_count,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    score_head_path = args.model_dir / "score_head.pt"
    if not score_head_path.exists():
        raise FileNotFoundError(f"score_head.pt not found in {args.model_dir}")

    dataset = PairwiseCostDataset(args.dataset_path)
    if len(dataset) == 0:
        raise ValueError("No valid samples in dataset for validation")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    model = DebertaCostModel(backbone_path=args.model_dir)
    score_state = torch.load(score_head_path, map_location="cpu")
    model.score_head.load_state_dict(score_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_max_length = getattr(model.backbone.config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if isinstance(model_max_length, int) and model_max_length > 0:
        effective_max_length = min(effective_max_length, model_max_length)

    n_total = len(dataset)
    indices = list(range(n_total))
    random.Random(args.seed).shuffle(indices)

    val_size = max(1, int(n_total * args.validation_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if not train_indices:
        raise ValueError("Validation split too large; no train subset left")

    collator = PairCollator(tokenizer=tokenizer, max_length=effective_max_length)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    train_metrics = evaluate_subset(model, train_loader, device, "Eval-TrainSubset")
    val_metrics = evaluate_subset(model, val_loader, device, "Eval-Validation")

    summary = {
        "model_dir": str(args.model_dir),
        "dataset_path": str(args.dataset_path),
        "dataset_total_samples": n_total,
        "dataset_skipped_samples": dataset.skipped,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "train_subset": train_metrics,
        "validation_subset": val_metrics,
    }

    out_path = args.model_dir / "validation_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

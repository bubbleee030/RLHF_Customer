#!/usr/bin/env python3
"""Train a DeBERTa-based cost model on pairwise safety preference data.

Expected JSONL fields per row:
- input
- answer
- other_answer
- safer (bool): whether answer is safer than other_answer
Optional fallback fields:
- is_safe, is_other_safe
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


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
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
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
    parser = argparse.ArgumentParser(description="Train DeBERTa cost model on local JSONL data")
    parser.add_argument("--model-name-or-path", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PairwiseCostDataset(args.dataset_path)
    if len(dataset) == 0:
        raise ValueError("No valid training samples were found in dataset.")

    print(f"Loaded samples: {len(dataset)}")
    print(f"Skipped rows: {dataset.skipped}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    model = DebertaCostModel(args.model_name_or_path)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    model.to(device)

    model_max_length = getattr(model.backbone.config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if isinstance(model_max_length, int) and model_max_length > 0:
        effective_max_length = min(effective_max_length, model_max_length)
    if effective_max_length < args.max_length:
        print(
            f"Requested max_length={args.max_length} exceeds model limit; using {effective_max_length} instead.",
        )

    collator = PairCollator(tokenizer=tokenizer, max_length=effective_max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(dataloader) / max(1, args.gradient_accumulation_steps))
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    start_time = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, batch in enumerate(pbar, start=1):
            safe_input_ids = batch["safe_input_ids"].to(device, non_blocking=True)
            safe_attention_mask = batch["safe_attention_mask"].to(device, non_blocking=True)
            unsafe_input_ids = batch["unsafe_input_ids"].to(device, non_blocking=True)
            unsafe_attention_mask = batch["unsafe_attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                safe_scores, unsafe_scores = model(
                    safe_input_ids=safe_input_ids,
                    safe_attention_mask=safe_attention_mask,
                    unsafe_input_ids=unsafe_input_ids,
                    unsafe_attention_mask=unsafe_attention_mask,
                )
                # Encourage higher cost for unsafe responses.
                loss = F.softplus(safe_scores - unsafe_scores).mean()
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if step_idx % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                with torch.no_grad():
                    batch_acc = (unsafe_scores > safe_scores).float().mean().item()
                    display_loss = loss.item() * args.gradient_accumulation_steps
                    current_lr = scheduler.get_last_lr()[0]

                if global_step % args.log_steps == 0 or global_step == 1:
                    pbar.set_postfix(loss=f"{display_loss:.4f}", acc=f"{batch_acc:.4f}", lr=f"{current_lr:.2e}")

                history.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": display_loss,
                        "accuracy": batch_acc,
                        "learning_rate": current_lr,
                    }
                )

        # Flush remaining gradients if dataloader length is not divisible by grad accumulation.
        if len(dataloader) % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

    elapsed = time.time() - start_time

    model.backbone.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), args.output_dir / "cost_model_state.pt")
    torch.save(model.score_head.state_dict(), args.output_dir / "score_head.pt")

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "num_samples": len(dataset),
        "skipped_samples": dataset.skipped,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "total_update_steps": global_step,
        "fp16": bool(args.fp16 and device.type == "cuda"),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "train_seconds": elapsed,
        "last_log": history[-1] if history else None,
        "best_accuracy": max((h["accuracy"] for h in history), default=None),
        "best_loss": min((h["loss"] for h in history), default=None),
    }

    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "arguments.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print("Training complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

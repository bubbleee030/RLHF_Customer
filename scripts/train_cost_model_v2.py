#!/usr/bin/env python3
"""Train a cost model using the PKU-SafeRLHF 3-term loss function.

Following the methodology from:
  PKU-Alignment/safe-rlhf  (safe_rlhf/values/cost/trainer.py)

Loss (sequence-wise):
  L = -log σ(C_unsafe - C_safe)           # Term 1: pairwise ordering
      -log σ(sign_safe * C_safe)           # Term 2: safe → negative cost
      -log σ(sign_unsafe * C_unsafe)       # Term 3: unsafe → positive cost

Where:
  sign_safe  = -safety_sign_safe   (safe → safety_sign=+1 → cost_sign=-1)
  sign_unsafe = -safety_sign_unsafe (unsafe → safety_sign=-1 → cost_sign=+1)

Expected JSONL fields per row:
  - input, answer, other_answer
  - safer (bool), is_safe (bool), is_other_safe (bool)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class PairSample:
    prompt: str
    safer_response: str       # lower cost expected
    unsafer_response: str     # higher cost expected
    safer_safety_sign: int    # +1 if safe, -1 if unsafe
    unsafer_safety_sign: int  # +1 if safe, -1 if unsafe


@dataclass
class RunningScoreNormalizer:
    """EMA normalizer for score stabilization across steps."""

    momentum: float = 0.9
    eps: float = 1e-6
    mean: torch.Tensor | None = None
    var: torch.Tensor | None = None
    initialized: bool = False

    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        values = values.detach().float()
        if mask is not None:
            values = values[mask.bool()]
        values = values.reshape(-1)
        if values.numel() == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_var = batch_var.clamp_min(self.eps)

        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var
            self.initialized = True
            return

        keep = self.momentum
        update = 1.0 - keep
        assert self.mean is not None and self.var is not None
        self.mean = keep * self.mean + update * batch_mean
        self.var = keep * self.var + update * batch_var

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        if not self.initialized or self.mean is None or self.var is None:
            return values
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.var.to(device=values.device, dtype=values.dtype).add(self.eps).sqrt()
        return (values - mean) / std

    def export_state(self) -> dict[str, Any]:
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "initialized": self.initialized,
            "mean": None if self.mean is None else float(self.mean.item()),
            "var": None if self.var is None else float(self.var.item()),
        }


class PairwiseCostDataset(Dataset):
    """Load pairwise cost data with safety signs."""

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
                is_safe = row.get("is_safe", None)
                is_other_safe = row.get("is_other_safe", None)

                if not prompt or not answer or not other:
                    self.skipped += 1
                    continue

                if not isinstance(safer, bool) or not isinstance(is_safe, bool) or not isinstance(is_other_safe, bool):
                    self.skipped += 1
                    continue

                # Determine safer/unsafer assignment
                if safer:
                    safer_response = answer
                    unsafer_response = other
                    safer_safety_sign = 2 * int(is_safe) - 1      # +1 if safe, -1 if unsafe
                    unsafer_safety_sign = 2 * int(is_other_safe) - 1
                else:
                    safer_response = other
                    unsafer_response = answer
                    safer_safety_sign = 2 * int(is_other_safe) - 1
                    unsafer_safety_sign = 2 * int(is_safe) - 1

                # Validate: safer answer's sign must be >= unsafer answer's sign
                if safer_safety_sign < unsafer_safety_sign:
                    self.skipped += 1
                    continue

                # Skip identical responses
                if safer_response == unsafer_response:
                    self.skipped += 1
                    continue

                self.samples.append(
                    PairSample(
                        prompt=prompt,
                        safer_response=safer_response,
                        unsafer_response=unsafer_response,
                        safer_safety_sign=safer_safety_sign,
                        unsafer_safety_sign=unsafer_safety_sign,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PairSample:
        return self.samples[idx]


class PairCollator:
    """Tokenize and collate pairs with safety signs."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _format_text(prompt: str, response: str) -> str:
        return f"User: {prompt}\nAssistant: {response}"

    def __call__(self, batch: list[PairSample]) -> dict[str, torch.Tensor]:
        safer_texts = [self._format_text(s.prompt, s.safer_response) for s in batch]
        unsafer_texts = [self._format_text(s.prompt, s.unsafer_response) for s in batch]

        all_texts = safer_texts + unsafer_texts
        all_encoded = self.tokenizer(
            all_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        safer_input_ids, unsafer_input_ids = all_encoded["input_ids"].chunk(2, dim=0)
        safer_attention_mask, unsafer_attention_mask = all_encoded["attention_mask"].chunk(2, dim=0)

        safer_signs = torch.tensor([s.safer_safety_sign for s in batch], dtype=torch.long)
        unsafer_signs = torch.tensor([s.unsafer_safety_sign for s in batch], dtype=torch.long)

        return {
            "safer_input_ids": safer_input_ids,
            "safer_attention_mask": safer_attention_mask,
            "safer_safety_sign": safer_signs,
            "unsafer_input_ids": unsafer_input_ids,
            "unsafer_attention_mask": unsafer_attention_mask,
            "unsafer_safety_sign": unsafer_signs,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CostModel(nn.Module):
    """Score model with configurable pooling."""

    def __init__(
        self,
        model_name_or_path: str,
        load_in_half: bool = False,
        pooling: str = "mean",
        use_bf16: bool = False,
        use_device_map: bool = False,
        model_revision: str | None = None,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self._use_device_map = use_device_map
        dtype = (torch.bfloat16 if use_bf16 else torch.float16) if load_in_half else torch.float32

        load_kwargs: dict = {"dtype": dtype, "use_safetensors": True}
        if use_device_map:
            load_kwargs["device_map"] = "auto"
        if model_revision is not None:
            load_kwargs["revision"] = model_revision

        # Try AutoModel first; for multimodal models (e.g. Mistral3ForConditionalGeneration)
        # fall back to AutoModelForCausalLM and extract the inner language model.
        try:
            raw = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        except Exception:
            raw = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

        # For multimodal / conditional-generation models, pull out the language model.
        self.backbone = (
            raw.language_model if hasattr(raw, 'language_model')
            else raw.model if hasattr(raw, 'model') and hasattr(raw.model, 'embed_tokens')
            else raw
        )

        # Resolve hidden_size from the backbone or its config hierarchy.
        cfg = self.backbone.config
        hidden_size = (
            getattr(cfg, 'hidden_size', None)
            or getattr(getattr(cfg, 'text_config', None), 'hidden_size', None)
            or getattr(raw.config, 'hidden_size', None)
            or getattr(getattr(raw.config, 'text_config', None), 'hidden_size', None)
        )
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config.")

        if use_device_map:
            # Score head lives on whichever device the backbone's last layer lands on.
            params = list(self.backbone.parameters())
            self._first_device = params[0].device
            self._last_device = params[-1].device
            self.score_head = nn.Linear(hidden_size, 1).float().to(self._last_device)
        else:
            self.score_head = nn.Linear(hidden_size, 1).float()

        nn.init.zeros_(self.score_head.bias)
        # Detect once whether backbone accepts use_cache (decoder models do, encoders don't)
        import inspect
        sig = inspect.signature(self.backbone.forward)
        self._supports_use_cache = 'use_cache' in sig.parameters

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return per-token and sequence-level scores."""
        if self._use_device_map:
            # device_map scatters backbone across GPUs; route inputs to the first device.
            input_ids = input_ids.to(self._first_device)
            attention_mask = attention_mask.to(self._first_device)

        call_kwargs: dict = {'use_cache': False} if self._supports_use_cache else {}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **call_kwargs)
        last_hidden_state = outputs.last_hidden_state  # (B, L, E)

        # With device_map, last_hidden_state lands on _last_device; move attention_mask there too.
        hidden_dev = last_hidden_state.device
        attention_mask_h = attention_mask.to(hidden_dev)

        token_scores = self.score_head(last_hidden_state.float()).squeeze(-1)  # (B, L)

        if self.pooling == "last-token":
            # Find the index of the last non-padding token for each sequence
            end_index = attention_mask_h.long().sum(dim=1) - 1  # (B,)
            end_index = end_index.clamp(min=0)
            batch_size = last_hidden_state.size(0)
            pooled_hidden = last_hidden_state[
                torch.arange(batch_size, device=hidden_dev),
                end_index,
            ]  # (B, E)
        elif self.pooling == "cls":
            # Take the first token
            pooled_hidden = last_hidden_state[:, 0, :]  # (B, E)
        elif self.pooling == "mean":
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask_h.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_hidden = sum_hidden / sum_mask  # (B, E)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Force FP32 for score head to prevent NaN under FP16 on V100
        pooled_hidden = pooled_hidden.float()
        sequence_scores = self.score_head(pooled_hidden).squeeze(-1)  # (B,)
        return {
            "token_scores": token_scores,
            "sequence_scores": sequence_scores,
        }

    def forward(
        self,
        safer_input_ids: torch.Tensor,
        safer_attention_mask: torch.Tensor,
        unsafer_input_ids: torch.Tensor,
        unsafer_attention_mask: torch.Tensor,
        concat_forward: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return pair scores for safer/unsafer samples."""
        if concat_forward:
            combined = self._encode(
                torch.cat([safer_input_ids, unsafer_input_ids], dim=0),
                torch.cat([safer_attention_mask, unsafer_attention_mask], dim=0),
            )
            lower_token_scores, higher_token_scores = combined["token_scores"].chunk(chunks=2, dim=0)
            lower_sequence_scores, higher_sequence_scores = combined["sequence_scores"].chunk(chunks=2, dim=0)
        else:
            lower = self._encode(safer_input_ids, safer_attention_mask)
            higher = self._encode(unsafer_input_ids, unsafer_attention_mask)
            lower_token_scores = lower["token_scores"]
            higher_token_scores = higher["token_scores"]
            lower_sequence_scores = lower["sequence_scores"]
            higher_sequence_scores = higher["sequence_scores"]

        return {
            "lower_token_scores": lower_token_scores,
            "higher_token_scores": higher_token_scores,
            "lower_sequence_scores": lower_sequence_scores,
            "higher_sequence_scores": higher_sequence_scores,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_cost_loss(
    lower_sequence_scores: torch.Tensor,    # (B,) — safer responses, expected lower cost
    higher_sequence_scores: torch.Tensor,   # (B,) — unsafer responses, expected higher cost
    safer_safety_sign: torch.Tensor, # (B,) — +1 safe / -1 unsafe
    unsafer_safety_sign: torch.Tensor, # (B,) — +1 safe / -1 unsafe
    regularization: float = 0.0,
    loss_type: str = "sequence-wise",
    lower_token_scores: torch.Tensor | None = None,
    higher_token_scores: torch.Tensor | None = None,
    safer_input_ids: torch.Tensor | None = None,
    unsafer_input_ids: torch.Tensor | None = None,
    safer_attention_mask: torch.Tensor | None = None,
    unsafer_attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PKU-SafeRLHF 3-term cost loss.

    cost_sign = -safety_sign:
        safe  (safety_sign=+1) → cost_sign=-1 → we want cost<0 → logsigmoid(-1*cost) = logsigmoid(|cost|) ✓
        unsafe(safety_sign=-1) → cost_sign=+1 → we want cost>0 → logsigmoid(+1*cost) = logsigmoid(|cost|) ✓
    """
    # Always compute loss in float32 to prevent FP16 overflow in logsigmoid
    lower_sequence_scores_f = lower_sequence_scores.float()
    higher_sequence_scores_f = higher_sequence_scores.float()

    # Clamp scores to prevent extreme values that cause logsigmoid to return -inf
    lower_sequence_scores_f = lower_sequence_scores_f.clamp(-50.0, 50.0)
    higher_sequence_scores_f = higher_sequence_scores_f.clamp(-50.0, 50.0)

    # cost_sign = -safety_sign
    lower_cost_sign = -safer_safety_sign.float()    # (B,)
    higher_cost_sign = -unsafer_safety_sign.float()  # (B,)

    if loss_type == "token-wise":
        if any(v is None for v in (
            lower_token_scores,
            higher_token_scores,
            safer_input_ids,
            unsafer_input_ids,
            safer_attention_mask,
            unsafer_attention_mask,
        )):
            raise ValueError("token-wise loss requires token scores, input ids, and attention masks.")

        lower_token_scores_f = lower_token_scores.float().clamp(-50.0, 50.0)
        higher_token_scores_f = higher_token_scores.float().clamp(-50.0, 50.0)
        losses = []
        batch_size = lower_sequence_scores_f.size(0)

        for i in range(batch_size):
            lower_end_index = safer_attention_mask[i].nonzero()[-1].squeeze().item()
            higher_end_index = unsafer_attention_mask[i].nonzero()[-1].squeeze().item()
            end_index = max(higher_end_index, lower_end_index)

            diverge_positions = (safer_input_ids[i] != unsafer_input_ids[i]).nonzero(as_tuple=False)
            if diverge_positions.numel() == 0:
                sample_loss = (
                    -F.logsigmoid(higher_sequence_scores_f[i] - lower_sequence_scores_f[i])
                    - F.logsigmoid(lower_cost_sign[i] * lower_sequence_scores_f[i])
                    - F.logsigmoid(higher_cost_sign[i] * higher_sequence_scores_f[i])
                )
                if regularization > 0.0:
                    sample_loss = sample_loss + regularization * torch.stack(
                        [lower_sequence_scores_f[i], higher_sequence_scores_f[i]]
                    ).square().mean()
                losses.append(sample_loss)
                continue

            diverge_index = diverge_positions[0].squeeze().item()
            lower_truncated_scores = lower_token_scores_f[i, diverge_index : end_index + 1]
            higher_truncated_scores = higher_token_scores_f[i, diverge_index : end_index + 1]

            sample_loss = (
                -F.logsigmoid(higher_truncated_scores - lower_truncated_scores).mean()
                - F.logsigmoid(lower_cost_sign[i] * lower_truncated_scores).mean()
                - F.logsigmoid(higher_cost_sign[i] * higher_truncated_scores).mean()
            )

            if regularization > 0.0:
                sample_loss = sample_loss + regularization * torch.stack(
                    [lower_truncated_scores, higher_truncated_scores]
                ).square().mean()
            losses.append(sample_loss)

        loss = torch.stack(losses).mean()
    elif loss_type == "sequence-wise":
        loss = (
            -F.logsigmoid(higher_sequence_scores_f - lower_sequence_scores_f)   # Term 1: pairwise ordering
            - F.logsigmoid(lower_cost_sign * lower_sequence_scores_f)           # Term 2: safe → negative cost
            - F.logsigmoid(higher_cost_sign * higher_sequence_scores_f)         # Term 3: unsafe → positive cost
        ).mean()

        if regularization > 0.0:
            loss = loss + regularization * torch.stack(
                [lower_sequence_scores_f, higher_sequence_scores_f]
            ).square().mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Metrics (computed in float32)
    accuracy = (higher_sequence_scores_f > lower_sequence_scores_f).float().mean()
    accuracy_sign = torch.stack([
        lower_cost_sign * lower_sequence_scores_f > 0.0,
        higher_cost_sign * higher_sequence_scores_f > 0.0,
    ]).float().mean()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_sign": accuracy_sign,
        "lower_sequence_scores": lower_sequence_scores,
        "higher_sequence_scores": higher_sequence_scores,
    }


def maybe_normalize_pair_scores(
    pair_scores: dict[str, torch.Tensor],
    sequence_normalizer: RunningScoreNormalizer | None,
    token_normalizer: RunningScoreNormalizer | None,
    safer_attention_mask: torch.Tensor,
    unsafer_attention_mask: torch.Tensor,
    *,
    update_stats: bool,
) -> dict[str, torch.Tensor]:
    """Apply running-score normalization when enabled."""
    normalized_scores = dict(pair_scores)

    if sequence_normalizer is not None:
        stacked_sequence_scores = torch.cat(
            [pair_scores["lower_sequence_scores"], pair_scores["higher_sequence_scores"]],
            dim=0,
        )
        if update_stats:
            sequence_normalizer.update(stacked_sequence_scores)
        normalized_scores["lower_sequence_scores"] = sequence_normalizer.normalize(
            pair_scores["lower_sequence_scores"]
        )
        normalized_scores["higher_sequence_scores"] = sequence_normalizer.normalize(
            pair_scores["higher_sequence_scores"]
        )

    if token_normalizer is not None:
        stacked_token_scores = torch.cat(
            [pair_scores["lower_token_scores"], pair_scores["higher_token_scores"]],
            dim=0,
        )
        stacked_attention_mask = torch.cat([safer_attention_mask, unsafer_attention_mask], dim=0)
        if update_stats:
            token_normalizer.update(stacked_token_scores, stacked_attention_mask)
        normalized_scores["lower_token_scores"] = token_normalizer.normalize(
            pair_scores["lower_token_scores"]
        )
        normalized_scores["higher_token_scores"] = token_normalizer.normalize(
            pair_scores["higher_token_scores"]
        )

    return normalized_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CostModel,
    dataloader: DataLoader,
    device: torch.device,
    regularization: float = 0.0,
    loss_type: str = "sequence-wise",
    concat_forward: bool = False,
    sequence_normalizer: RunningScoreNormalizer | None = None,
    token_normalizer: RunningScoreNormalizer | None = None,
    dataset_subset: Subset | None = None,
    epoch: int | None = None,
    pred_output_path: Path | None = None,
) -> dict[str, float]:
    """Evaluate on a dataloader.

    When dataset_subset + pred_output_path are provided, appends per-sample
    predictions to pred_output_path as JSONL (one JSON object per line).
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_acc_sign = 0.0
    total_count = 0
    all_costs = []

    # Pre-extract PairSample objects in eval order for prediction logging.
    all_samples: list[PairSample] | None = None
    if pred_output_path is not None and dataset_subset is not None:
        all_samples = [dataset_subset.dataset[idx] for idx in dataset_subset.indices]
    sample_cursor = 0
    batch_predictions: list[dict] = []

    for batch in dataloader:
        safer_ids = batch["safer_input_ids"].to(device, non_blocking=True)
        safer_mask = batch["safer_attention_mask"].to(device, non_blocking=True)
        safer_sign = batch["safer_safety_sign"].to(device, non_blocking=True)
        unsafer_ids = batch["unsafer_input_ids"].to(device, non_blocking=True)
        unsafer_mask = batch["unsafer_attention_mask"].to(device, non_blocking=True)
        unsafer_sign = batch["unsafer_safety_sign"].to(device, non_blocking=True)

        pair_scores = model(
            safer_ids,
            safer_mask,
            unsafer_ids,
            unsafer_mask,
            concat_forward=concat_forward,
        )
        normalized_pair_scores = maybe_normalize_pair_scores(
            pair_scores,
            sequence_normalizer,
            token_normalizer,
            safer_mask,
            unsafer_mask,
            update_stats=False,
        )

        _score_dev = normalized_pair_scores["lower_sequence_scores"].device
        result = compute_cost_loss(
            normalized_pair_scores["lower_sequence_scores"],
            normalized_pair_scores["higher_sequence_scores"],
            safer_sign.to(_score_dev),
            unsafer_sign.to(_score_dev),
            regularization,
            loss_type=loss_type,
            lower_token_scores=normalized_pair_scores["lower_token_scores"],
            higher_token_scores=normalized_pair_scores["higher_token_scores"],
            safer_input_ids=safer_ids.to(_score_dev),
            unsafer_input_ids=unsafer_ids.to(_score_dev),
            safer_attention_mask=safer_mask.to(_score_dev),
            unsafer_attention_mask=unsafer_mask.to(_score_dev),
        )

        bs = safer_ids.size(0)
        total_loss += result["loss"].item() * bs
        total_acc += result["accuracy"].item() * bs
        total_acc_sign += result["accuracy_sign"].item() * bs
        total_count += bs

        all_costs.extend(pair_scores["lower_sequence_scores"].tolist())
        all_costs.extend(pair_scores["higher_sequence_scores"].tolist())

        # Build per-sample prediction records
        if all_samples is not None:
            safer_scores_cpu = pair_scores["lower_sequence_scores"].cpu().tolist()
            unsafer_scores_cpu = pair_scores["higher_sequence_scores"].cpu().tolist()
            safer_signs_cpu = safer_sign.cpu().tolist()
            unsafer_signs_cpu = unsafer_sign.cpu().tolist()
            for i in range(bs):
                s = all_samples[sample_cursor + i]
                ss = safer_scores_cpu[i]
                us = unsafer_scores_cpu[i]
                s_sign = int(safer_signs_cpu[i])
                u_sign = int(unsafer_signs_cpu[i])
                batch_predictions.append({
                    "epoch": epoch,
                    "sample_idx": int(dataset_subset.indices[sample_cursor + i]),
                    "prompt": s.prompt,
                    "safer_response": s.safer_response,
                    "unsafer_response": s.unsafer_response,
                    "safer_score": round(ss, 4),
                    "unsafer_score": round(us, 4),
                    "pairwise_correct": bool(us > ss),
                    "safer_sign": s_sign,
                    "unsafer_sign": u_sign,
                    "safer_sign_correct": bool((s_sign > 0 and ss < 0) or (s_sign < 0 and ss > 0)),
                    "unsafer_sign_correct": bool((u_sign > 0 and us < 0) or (u_sign < 0 and us > 0)),
                })
        sample_cursor += bs

    model.train()

    # Write predictions for this epoch
    if pred_output_path is not None and batch_predictions:
        with pred_output_path.open("a", encoding="utf-8") as fout:
            for rec in batch_predictions:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if total_count == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "accuracy_sign": float("nan")}

    costs_tensor = torch.tensor(all_costs)
    info = {
        "loss": total_loss / total_count,
        "accuracy": total_acc / total_count,
        "accuracy_sign": total_acc_sign / total_count,
        "cost_mean": costs_tensor.mean().item(),
        "cost_std": costs_tensor.std().item(),
        "num_samples": total_count,
    }
    if sequence_normalizer is not None and sequence_normalizer.initialized:
        normalized_costs = sequence_normalizer.normalize(costs_tensor)
        info["normalized_cost_mean"] = normalized_costs.mean().item()
        info["normalized_cost_std"] = normalized_costs.std().item()
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, save_model, tokenizer, args, save_backbone: bool) -> None:
    """Write a full, self-contained, reloadable checkpoint: backbone + score_head +
    tokenizer + arguments.json (so eval/inference can recover model id, pooling, max_length)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    torch.save(save_model.score_head.state_dict(), out_dir / "score_head.pt")
    if save_backbone or getattr(args, "lora_r", 0) > 0:
        try:
            save_model.backbone.save_pretrained(out_dir, safe_serialization=True)
        except RuntimeError as e:
            print(f"  safetensors save failed ({e}); retrying with safe_serialization=False")
            save_model.backbone.save_pretrained(out_dir, safe_serialization=False)
    with (out_dir / "arguments.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cost model with PKU-SafeRLHF 3-term loss")
    parser.add_argument("--model-name-or-path", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=0.001,
                        help="L2 regularization weight on scores")
    parser.add_argument("--eval-split-ratio", type=float, default=0.1,
                        help="Fraction of data for evaluation")
    parser.add_argument("--eval-dataset-path", type=Path, default=None,
                        help="Explicit eval split produced by scripts/cost/split_cost_dataset.py. "
                             "When given, --eval-split-ratio is ignored and no in-trainer split "
                             "occurs. Preferred path; the internal split is retained only for "
                             "backward compatibility with historical runs.")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", type=str, choices=["mean", "cls", "last-token"], default="mean",
                        help="Pooling method for the sequence")
    parser.add_argument("--loss-type", type=str, choices=["sequence-wise", "token-wise"], default="sequence-wise",
                        help="PKU cost loss variant to optimize")
    parser.add_argument("--concat-forward", action="store_true",
                        help="Run safer/unsafer pairs in a single concatenated forward pass")
    parser.add_argument("--normalize-score-during-training", action="store_true",
                        help="Apply running z-score normalization to sequence/token scores during training")
    parser.add_argument("--normalizer-momentum", type=float, default=0.9,
                        help="EMA momentum for running score normalization")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true",
                        help="Use autocast FP16 (best for encoder models like DeBERTa)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 (autocast or native, best for newer decoder models like Gemma/Qwen)")
    parser.add_argument("--load-in-half", action="store_true",
                        help="Load backbone in native half precision (FP16 or BF16)")
    parser.add_argument("--device-map", action="store_true",
                        help="Use device_map='auto' to shard large models across all GPUs (skips DataParallel)")
    parser.add_argument("--adafactor", action="store_true",
                        help="Use Adafactor optimizer (memory-efficient; recommended for large decoder models)")
    parser.add_argument("--save-eval-predictions", action="store_true",
                        help="Save per-sample eval predictions to eval_predictions.jsonl in output-dir")
    parser.add_argument("--save-backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Persist the fine-tuned backbone via save_pretrained (default: on). REQUIRED for "
                             "full fine-tunes — without it the trained model is lost and only score_head.pt remains. "
                             "Pass --no-save-backbone only if you deliberately want to discard backbone weights.")
    parser.add_argument("--save-best", type=str, default="",
                        help="Comma-separated metric(s) to keep the BEST per-epoch checkpoint for, e.g. "
                             "'pairwise,loss'. Each is saved to <output_dir>/best-<metric>/ whenever it improves. "
                             "Metrics: pairwise (eval_accuracy, higher=better), loss (eval_loss, lower), "
                             "sign (eval_accuracy_sign, higher). When set, the final last-epoch backbone is NOT "
                             "written to the run-dir root (keep best, not latest).")
    parser.add_argument("--lora-r", type=int, default=0,
                        help="LoRA rank. 0 (default) means full fine-tuning.")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def configure_cost_lora(backbone, r: int, alpha: int, dropout: float):
    """Wrap the backbone in a LoRA adapter. r <= 0 means full fine-tuning."""
    if r <= 0:
        return backbone

    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    return get_peft_model(backbone, config)


def cost_trainable_parameter_names(model) -> list[str]:
    """Names of parameters that will receive gradients."""
    return [name for name, param in model.named_parameters() if param.requires_grad]


def load_tokenizer(model_name_or_path: str, model_revision: str | None = None):
    """Load the tokenizer from the same pinned revision as the backbone."""
    revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, fix_mistral_regex=True, **revision_kwargs
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name_or_path, **revision_kwargs)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo_root))
    from scripts.lib.run_manifest import (
        build_run_manifest,
        resolve_model_revision,
        write_run_manifest,
    )

    _manifest_inputs = [args.dataset_path]
    if getattr(args, "eval_dataset_path", None) is not None:
        _manifest_inputs.append(args.eval_dataset_path)
    _model_revision = resolve_model_revision(args.model_name_or_path)
    write_run_manifest(
        build_run_manifest(
            model_name=args.model_name_or_path,
            model_revision=_model_revision,
            output_dir=args.output_dir,
            params=vars(args),
            input_files=_manifest_inputs,
            code_files=[
                Path(__file__).resolve(),
                _repo_root / "scripts" / "trainer.py",
                _repo_root / "scripts" / "lib" / "run_manifest.py",
            ],
            extra={"stage": "cost_model_train"},
        ),
        args.output_dir,
    )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load dataset ---
    full_dataset = PairwiseCostDataset(args.dataset_path)
    if len(full_dataset) == 0:
        raise ValueError("No valid training samples found in dataset.")

    print(f"Loaded samples: {len(full_dataset)}")
    print(f"Skipped rows: {full_dataset.skipped}")

    # --- Train/eval split ---
    if args.eval_dataset_path is not None:
        eval_dataset = PairwiseCostDataset(args.eval_dataset_path)
        if len(eval_dataset) == 0:
            raise ValueError(f"No valid eval samples found in {args.eval_dataset_path}")
        train_indices = list(range(len(full_dataset)))
        eval_indices = list(range(len(eval_dataset)))
        print(f"Using external eval split: {args.eval_dataset_path}")
    else:
        eval_dataset = full_dataset
        n_total = len(full_dataset)
        indices = list(range(n_total))
        random.Random(args.seed).shuffle(indices)

        n_eval = max(1, int(n_total * args.eval_split_ratio))
        eval_indices = indices[:n_eval]
        train_indices = indices[n_eval:]
        print("WARNING: using the in-trainer pair-level split. This leaks prompts "
              "across train/eval. Prefer --eval-dataset-path.")

    print(f"Train samples: {len(train_indices)}")
    print(f"Eval samples: {len(eval_indices)}")

    # --- Tokenizer ---
    tokenizer = load_tokenizer(args.model_name_or_path, _model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    # --- Model ---
    load_in_half = getattr(args, 'load_in_half', False)
    use_bf16 = getattr(args, 'bf16', False)
    use_device_map = getattr(args, 'device_map', False)
    model = CostModel(
        args.model_name_or_path,
        load_in_half=load_in_half,
        pooling=args.pooling,
        use_bf16=use_bf16,
        use_device_map=use_device_map,
        model_revision=_model_revision,
    )
    if args.lora_r > 0:
        model.backbone = configure_cost_lora(
            model.backbone, args.lora_r, args.lora_alpha, args.lora_dropout)
        n_trainable = len(cost_trainable_parameter_names(model.backbone))
        print(f"LoRA enabled: r={args.lora_r} alpha={args.lora_alpha} "
              f"dropout={args.lora_dropout}, {n_trainable} trainable backbone tensors")
    else:
        print("Full fine-tuning (no LoRA)")
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    if use_device_map:
        # Backbone already sharded across GPUs; only move the score head to its target device.
        print(f"Using device_map='auto' — backbone sharded across {torch.cuda.device_count()} GPUs")
        model.score_head = model.score_head.to(model._last_device)
    else:
        model.to(device)

    # --- Max length ---
    model_max_length = getattr(model.backbone.config, "max_position_embeddings", None)
    effective_max_length = args.max_length
    if isinstance(model_max_length, int) and model_max_length > 0:
        effective_max_length = min(effective_max_length, model_max_length)
    if effective_max_length < args.max_length:
        print(f"Requested max_length={args.max_length} exceeds model limit; using {effective_max_length}.")

    # Wrap in DataParallel only when NOT using device_map (device_map already distributes the model)
    if not use_device_map and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # When using load_in_half, disable autocast to avoid double-casting issues
    use_autocast = (args.fp16 or args.bf16) and device.type == "cuda" and not load_in_half
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    # GradScaler: FP16 only; disabled for BF16, load_in_half, or Adafactor (which handles clipping internally)
    is_half_model = any(p.dtype in (torch.float16, torch.bfloat16) for p in model.parameters())
    use_scaler = (
        args.fp16
        and device.type == "cuda"
        and not load_in_half
        and not is_half_model
        and not getattr(args, 'adafactor', False)
    )

    # Print param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # --- DataLoaders ---
    collator = PairCollator(tokenizer=tokenizer, max_length=effective_max_length)

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        Subset(eval_dataset, eval_indices),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # --- Optimizer & Scheduler ---
    use_adafactor = getattr(args, 'adafactor', False)
    if use_adafactor:
        # Adafactor stores factored second moments only (~1 GB for 8B model vs ~64 GB for Adam).
        # scale_parameter=False + relative_step=False lets us specify an explicit lr.
        optimizer = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )
        print("Optimizer: Adafactor (memory-efficient)")
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print("Optimizer: AdamW")

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    sequence_normalizer = (
        RunningScoreNormalizer(momentum=args.normalizer_momentum)
        if args.normalize_score_during_training else None
    )
    token_normalizer = (
        RunningScoreNormalizer(momentum=args.normalizer_momentum)
        if args.normalize_score_during_training else None
    )

    print(f"\n{'=' * 60}")
    print(f"TRAINING CONFIG")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Effective max_length: {effective_max_length}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Regularization: {args.regularization}")
    print(f"Loss type: {args.loss_type}")
    print(f"Concat forward: {args.concat_forward}")
    print(f"Score normalization: {args.normalize_score_during_training}")
    print(f"Epochs: {args.epochs}")
    print(f"Updates/epoch: {updates_per_epoch}")
    print(f"Total updates: {total_updates}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Autocast: {use_autocast} (dtype: {autocast_dtype if use_autocast else 'N/A'})")
    print(f"Load in half: {load_in_half} (BF16: {args.bf16})")
    print(f"{'=' * 60}\n")

    # --- Training Loop ---
    start_time = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []

    # --- Best-checkpoint selection (--save-best) ---
    save_best_metrics = [m.strip() for m in args.save_best.split(",") if m.strip()]
    _valid_best = {"pairwise", "loss", "sign"}
    _bad = set(save_best_metrics) - _valid_best
    if _bad:
        raise SystemExit(f"--save-best: unknown metric(s) {_bad}; choose from {sorted(_valid_best)}")
    best_metric_values: dict[str, float] = {}
    best_metric_epoch: dict[str, int] = {}
    if save_best_metrics:
        print(f"--save-best active for: {save_best_metrics} "
              f"(checkpoints -> {args.output_dir}/best-<metric>/; final latest backbone NOT saved to root)")

    ema_loss = None
    ema_acc = None
    ema_acc_sign = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_acc_sign = 0.0
        epoch_steps = 0

        step_loss = 0.0
        step_acc = 0.0
        step_acc_sign = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, batch in enumerate(pbar, start=1):
            safer_ids = batch["safer_input_ids"].to(device, non_blocking=True)
            safer_mask = batch["safer_attention_mask"].to(device, non_blocking=True)
            safer_sign = batch["safer_safety_sign"].to(device, non_blocking=True)
            unsafer_ids = batch["unsafer_input_ids"].to(device, non_blocking=True)
            unsafer_mask = batch["unsafer_attention_mask"].to(device, non_blocking=True)
            unsafer_sign = batch["unsafer_safety_sign"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                pair_scores = model(
                    safer_ids,
                    safer_mask,
                    unsafer_ids,
                    unsafer_mask,
                    concat_forward=args.concat_forward,
                )
                normalized_pair_scores = maybe_normalize_pair_scores(
                    pair_scores,
                    sequence_normalizer,
                    token_normalizer,
                    safer_mask,
                    unsafer_mask,
                    update_stats=True,
                )

                # With device_map, scores may land on a different GPU than the sign tensors.
                _score_dev = normalized_pair_scores["lower_sequence_scores"].device
                result = compute_cost_loss(
                    normalized_pair_scores["lower_sequence_scores"],
                    normalized_pair_scores["higher_sequence_scores"],
                    safer_sign.to(_score_dev),
                    unsafer_sign.to(_score_dev),
                    regularization=args.regularization,
                    loss_type=args.loss_type,
                    lower_token_scores=normalized_pair_scores["lower_token_scores"],
                    higher_token_scores=normalized_pair_scores["higher_token_scores"],
                    safer_input_ids=safer_ids.to(_score_dev),
                    unsafer_input_ids=unsafer_ids.to(_score_dev),
                    safer_attention_mask=safer_mask.to(_score_dev),
                    unsafer_attention_mask=unsafer_mask.to(_score_dev),
                )
                loss = result["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            with torch.no_grad():
                step_loss += result["loss"].item() / args.gradient_accumulation_steps
                step_acc += result["accuracy"].item() / args.gradient_accumulation_steps
                step_acc_sign += result["accuracy_sign"].item() / args.gradient_accumulation_steps

            if step_idx % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                with torch.no_grad():
                    current_lr = scheduler.get_last_lr()[0]

                    if ema_loss is None:
                        ema_loss = step_loss
                        ema_acc = step_acc
                        ema_acc_sign = step_acc_sign
                    else:
                        alpha = 0.1
                        ema_loss = alpha * step_loss + (1 - alpha) * ema_loss
                        ema_acc = alpha * step_acc + (1 - alpha) * ema_acc
                        ema_acc_sign = alpha * step_acc_sign + (1 - alpha) * ema_acc_sign

                epoch_loss += step_loss
                epoch_acc += step_acc
                epoch_acc_sign += step_acc_sign
                epoch_steps += 1

                if global_step % args.log_steps == 0 or global_step == 1:
                    pbar.set_postfix(
                        loss=f"{step_loss:.4f}",
                        acc=f"{step_acc:.4f}",
                        acc_s=f"{step_acc_sign:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

                train_history.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": step_loss,
                    "accuracy": step_acc,
                    "accuracy_sign": step_acc_sign,
                    "loss_ema": ema_loss,
                    "accuracy_ema": ema_acc,
                    "accuracy_sign_ema": ema_acc_sign,
                    "learning_rate": current_lr,
                })

                step_loss = 0.0
                step_acc = 0.0
                step_acc_sign = 0.0

        # Flush remaining gradients
        if len(train_loader) % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        # --- Epoch eval ---
        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_acc = epoch_acc / max(1, epoch_steps)
        avg_acc_sign = epoch_acc_sign / max(1, epoch_steps)

        save_preds = getattr(args, 'save_eval_predictions', False)
        eval_metrics = evaluate(
            model,
            eval_loader,
            device,
            regularization=args.regularization,
            loss_type=args.loss_type,
            concat_forward=args.concat_forward,
            sequence_normalizer=sequence_normalizer,
            token_normalizer=token_normalizer,
            dataset_subset=Subset(eval_dataset, eval_indices) if save_preds else None,
            epoch=epoch,
            pred_output_path=args.output_dir / "eval_predictions.jsonl" if save_preds else None,
        )

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  Train: loss={avg_loss:.4f}, acc={avg_acc:.4f}, acc_sign={avg_acc_sign:.4f}")
        print(f"  Eval:  loss={eval_metrics['loss']:.4f}, acc={eval_metrics['accuracy']:.4f}, "
              f"acc_sign={eval_metrics['accuracy_sign']:.4f}")
        print(f"  Eval cost: mean={eval_metrics['cost_mean']:.4f}, std={eval_metrics['cost_std']:.4f}")
        if "normalized_cost_mean" in eval_metrics:
            print(f"  Eval normalized cost: mean={eval_metrics['normalized_cost_mean']:.4f}, "
                  f"std={eval_metrics['normalized_cost_std']:.4f}")
        print()

        eval_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "train_accuracy_sign": avg_acc_sign,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_accuracy_sign": eval_metrics["accuracy_sign"],
            "eval_cost_mean": eval_metrics["cost_mean"],
            "eval_cost_std": eval_metrics["cost_std"],
            "eval_normalized_cost_mean": eval_metrics.get("normalized_cost_mean"),
            "eval_normalized_cost_std": eval_metrics.get("normalized_cost_std"),
        })

        # --- Best-checkpoint saving (keep best per metric, not just latest) ---
        if save_best_metrics:
            save_model_ck = model.module if isinstance(model, nn.DataParallel) else model
            metric_vals = {
                "pairwise": (eval_metrics["accuracy"], "max"),
                "loss": (eval_metrics["loss"], "min"),
                "sign": (eval_metrics["accuracy_sign"], "max"),
            }
            for m in save_best_metrics:
                val, direction = metric_vals[m]
                prev = best_metric_values.get(m)
                improved = prev is None or (val > prev if direction == "max" else val < prev)
                if improved:
                    best_metric_values[m] = val
                    best_metric_epoch[m] = epoch
                    ck_dir = args.output_dir / f"best-{m}"
                    print(f"  [save-best] new best {m}={val:.4f} @ epoch {epoch} -> {ck_dir}")
                    save_checkpoint(ck_dir, save_model_ck, tokenizer, args, save_backbone=args.save_backbone)

    elapsed = time.time() - start_time

    # --- Save ---
    # This is a FULL fine-tune (trainable params == total params), so the backbone
    # weights ARE the trained model. The previous version skipped saving the backbone
    # to "save disk" — that silently discarded the entire fine-tune (the May-18 bug:
    # only score_head.pt survived, which is useless on top of the pretrained backbone).
    # We now persist the backbone via save_pretrained in its native dtype (FP16 under
    # --load-in-half, ~6-7GB for Ministral-3B). eval/inference reload it from output_dir.
    print("Saving score head and tokenizer config...")
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    tokenizer.save_pretrained(args.output_dir)
    torch.save(save_model.score_head.state_dict(), args.output_dir / "score_head.pt")
    if save_best_metrics:
        # Keep best, not latest: the root holds logs/summary; deployable checkpoints are the best-<metric> dirs.
        print("[save-best] final last-epoch backbone NOT saved to run-dir root. Best checkpoints:")
        for m in save_best_metrics:
            print(f"   - {args.output_dir / ('best-' + m)}  (best {m} @ epoch {best_metric_epoch.get(m)}, "
                  f"value={best_metric_values.get(m):.4f})")
    elif args.save_backbone or args.lora_r > 0:
        bb_dtype = next(save_model.backbone.parameters()).dtype
        artifact_kind = "LoRA adapter" if args.lora_r > 0 else "fine-tuned backbone"
        print(f"Saving {artifact_kind} (dtype={bb_dtype}) to {args.output_dir} ...")
        try:
            save_model.backbone.save_pretrained(args.output_dir, safe_serialization=True)
        except RuntimeError as e:
            # safetensors refuses shared/tied tensors; fall back to pickle .bin format.
            print(f"  safetensors save failed ({e}); retrying with safe_serialization=False")
            save_model.backbone.save_pretrained(args.output_dir, safe_serialization=False)
        print(f"{artifact_kind.capitalize()} saved.")
    else:
        print("WARNING: --no-save-backbone set — fine-tuned backbone will NOT be persisted; "
              "this run's trained model cannot be reloaded.")
    if sequence_normalizer is not None or token_normalizer is not None:
        torch.save(
            {
                "sequence_normalizer": None if sequence_normalizer is None else sequence_normalizer.export_state(),
                "token_normalizer": None if token_normalizer is None else token_normalizer.export_state(),
            },
            args.output_dir / "score_normalizer.pt",
        )

    # Training summary
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "requested_max_length": args.max_length,
        "effective_max_length": effective_max_length,
        "num_train_samples": len(train_indices),
        "num_eval_samples": len(eval_indices),
        "skipped_samples": full_dataset.skipped,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "regularization": args.regularization,
        "total_update_steps": global_step,
        "autocast": use_autocast,
        "bf16": args.bf16,
        "load_in_half": load_in_half,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "train_seconds": elapsed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "loss_type": f"pku_3term_{args.loss_type}",
        "pooling": args.pooling,
        "concat_forward": args.concat_forward,
        "normalize_score_during_training": args.normalize_score_during_training,
        "normalizer_momentum": args.normalizer_momentum,
        "last_train_log": train_history[-1] if train_history else None,
        "last_eval_log": eval_history[-1] if eval_history else None,
        "best_eval_accuracy": max((e["eval_accuracy"] for e in eval_history), default=None),
        "best_eval_accuracy_sign": max((e["eval_accuracy_sign"] for e in eval_history), default=None),
        "best_eval_loss": min((e["eval_loss"] for e in eval_history), default=None),
    }

    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_history, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "eval_log.json").open("w", encoding="utf-8") as f:
        json.dump(eval_history, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "arguments.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a helpfulness Reward Model with the pure Bradley-Terry loss.

Adapted from scripts/train_cost_model_v2.py (the cost-model trainer). The
backbone, pooling, device-map / load-in-half handling, and optimizer setup are
reused verbatim; the cost-specific safety-sign loss terms are removed.

Reward loss (sequence-wise Bradley-Terry):
  L = -log σ(R(chosen) - R(rejected))   [ + regularization * mean(score^2) ]

Documented additions over the cost trainer (see docs/adr/0001-...):
  - reward pair dataset reading (input, chosen, rejected)
  - --eval-dataset-path : use a separate eval file (skip internal split) so we
    can evaluate on a by-prompt holdout
  - per-epoch backbone checkpointing (epochN/ subdirs) to keep the best epoch

Input JSONL schema (per line): input, chosen, rejected [, prompt_id, pair].
"""
from __future__ import annotations

import argparse
import json
import math
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


LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = max(0, patience)
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.non_improvements = 0

    def is_improvement(self, loss: float) -> bool:
        return loss < self.best_loss - self.min_delta

    def update(self, loss: float) -> bool:
        if self.is_improvement(loss):
            self.best_loss = loss
            self.non_improvements = 0
        else:
            self.non_improvements += 1
        return self.patience > 0 and self.non_improvements >= self.patience


def select_best_epoch(eval_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(eval_history, key=lambda entry: entry["loss"]) if eval_history else None


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def build_checkpoint_metadata(
    base_model_name_or_path: str,
    pooling: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "base_model_name_or_path": base_model_name_or_path,
        "pooling": pooling,
        "lora": {
            "enabled": lora_r > 0,
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
        },
    }


def configure_lora(
    backbone: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    config_factory=None,
    peft_factory=None,
) -> nn.Module:
    if r <= 0:
        return backbone
    if config_factory is None or peft_factory is None:
        from peft import LoraConfig, get_peft_model
        config_factory = LoraConfig
        peft_factory = get_peft_model
    config = config_factory(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGETS),
        task_type="FEATURE_EXTRACTION",
    )
    return peft_factory(backbone, config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RewardPair:
    prompt: str
    chosen: str      # more helpful -> higher reward expected
    rejected: str    # less helpful -> lower reward expected


class RewardPairDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        self.samples: list[RewardPair] = []
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
                chosen = str(row.get("chosen", "")).strip()
                rejected = str(row.get("rejected", "")).strip()
                if not prompt or not chosen or not rejected or chosen == rejected:
                    self.skipped += 1
                    continue
                self.samples.append(RewardPair(prompt=prompt, chosen=chosen, rejected=rejected))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RewardPair:
        return self.samples[idx]


class PairCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _format_text(prompt: str, response: str) -> str:
        return f"User: {prompt}\nAssistant: {response}"

    def __call__(self, batch: list[RewardPair]) -> dict[str, torch.Tensor]:
        chosen_texts = [self._format_text(s.prompt, s.chosen) for s in batch]
        rejected_texts = [self._format_text(s.prompt, s.rejected) for s in batch]
        enc = self.tokenizer(
            chosen_texts + rejected_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        chosen_ids, rejected_ids = enc["input_ids"].chunk(2, dim=0)
        chosen_mask, rejected_mask = enc["attention_mask"].chunk(2, dim=0)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


# ---------------------------------------------------------------------------
# Model (pooling + multimodal/decoder loading reused from the cost trainer)
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    def __init__(self, model_name_or_path: str, load_in_half: bool = False,
                 pooling: str = "last-token", use_bf16: bool = False,
                 use_device_map: bool = False, lora_r: int = 0,
                 lora_alpha: int = 32, lora_dropout: float = 0.05,
                 model_revision: str | None = None) -> None:
        super().__init__()
        self.pooling = pooling
        self._use_device_map = use_device_map
        self.base_model_name_or_path = model_name_or_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        dtype = (torch.bfloat16 if use_bf16 else torch.float16) if load_in_half else torch.float32
        load_kwargs: dict = {"dtype": dtype, "use_safetensors": True}
        if use_device_map:
            load_kwargs["device_map"] = "auto"
        if model_revision is not None:
            load_kwargs["revision"] = model_revision

        try:
            raw = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        except Exception:
            raw = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

        self.backbone = (
            raw.language_model if hasattr(raw, "language_model")
            else raw.model if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens")
            else raw
        )
        self.backbone = configure_lora(
            self.backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
        )
        cfg = self.backbone.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(raw.config, "hidden_size", None)
            or getattr(getattr(raw.config, "text_config", None), "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config.")

        if use_device_map:
            params = list(self.backbone.parameters())
            self._first_device = params[0].device
            self._last_device = params[-1].device
            self.score_head = nn.Linear(hidden_size, 1).float().to(self._last_device)
        else:
            self.score_head = nn.Linear(hidden_size, 1).float()
        nn.init.zeros_(self.score_head.bias)

        import inspect
        self._supports_use_cache = "use_cache" in inspect.signature(self.backbone.forward).parameters

    def _score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._use_device_map:
            input_ids = input_ids.to(self._first_device)
            attention_mask = attention_mask.to(self._first_device)
        call_kwargs: dict = {"use_cache": False} if self._supports_use_cache else {}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **call_kwargs)
        last_hidden_state = outputs.last_hidden_state  # (B, L, E)
        hidden_dev = last_hidden_state.device
        attn = attention_mask.to(hidden_dev)

        if self.pooling == "last-token":
            end_index = attn.long().sum(dim=1).sub(1).clamp(min=0)
            pooled = last_hidden_state[torch.arange(last_hidden_state.size(0), device=hidden_dev), end_index]
        elif self.pooling == "cls":
            pooled = last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            mask = attn.unsqueeze(-1).expand(last_hidden_state.size()).float()
            pooled = (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.score_head(pooled.float()).squeeze(-1)  # (B,)

    def forward(self, chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask, concat_forward: bool = False):
        if concat_forward:
            combined = self._score(
                torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
                torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),
            )
            chosen_scores, rejected_scores = combined.chunk(2, dim=0)
        else:
            chosen_scores = self._score(chosen_input_ids, chosen_attention_mask)
            rejected_scores = self._score(rejected_input_ids, rejected_attention_mask)
        return {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_reward_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor,
                        regularization: float = 0.0) -> dict[str, torch.Tensor]:
    cs = chosen_scores.float().clamp(-50.0, 50.0)
    rs = rejected_scores.float().clamp(-50.0, 50.0)
    loss = -F.logsigmoid(cs - rs).mean()
    if regularization > 0.0:
        loss = loss + regularization * torch.stack([cs, rs]).square().mean()
    accuracy = (cs > rs).float().mean()
    return {"loss": loss, "accuracy": accuracy,
            "chosen_scores": chosen_scores, "rejected_scores": rejected_scores}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, regularization=0.0, concat_forward=False) -> dict[str, float]:
    model.eval()
    total_loss = total_acc = total_count = 0.0
    all_scores = []
    for batch in dataloader:
        c_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
        c_mask = batch["chosen_attention_mask"].to(device, non_blocking=True)
        r_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
        r_mask = batch["rejected_attention_mask"].to(device, non_blocking=True)
        scores = model(c_ids, c_mask, r_ids, r_mask, concat_forward=concat_forward)
        result = compute_reward_loss(scores["chosen_scores"], scores["rejected_scores"], regularization)
        bs = c_ids.size(0)
        total_loss += result["loss"].item() * bs
        total_acc += result["accuracy"].item() * bs
        total_count += bs
        all_scores.extend(scores["chosen_scores"].tolist())
        all_scores.extend(scores["rejected_scores"].tolist())
    model.train()
    if total_count == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "num_samples": 0}
    t = torch.tensor(all_scores)
    return {"loss": total_loss / total_count, "accuracy": total_acc / total_count,
            "reward_mean": t.mean().item(), "reward_std": t.std().item(),
            "num_samples": int(total_count)}


def save_checkpoint(model, tokenizer, out_dir: Path, save_backbone: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    tokenizer.save_pretrained(out_dir)
    torch.save(save_model.score_head.state_dict(), out_dir / "score_head.pt")
    metadata = build_checkpoint_metadata(
        save_model.base_model_name_or_path,
        save_model.pooling,
        save_model.lora_r,
        save_model.lora_alpha,
        save_model.lora_dropout,
    )
    (out_dir / "reward_model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if save_backbone or save_model.lora_r > 0:
        try:
            save_model.backbone.save_pretrained(out_dir, safe_serialization=True)
        except RuntimeError as e:
            print(f"  safetensors save failed ({e}); retrying with safe_serialization=False")
            save_model.backbone.save_pretrained(out_dir, safe_serialization=False)


def load_tokenizer(model_name_or_path: str, model_revision: str | None = None):
    """Load the tokenizer from the same pinned revision as the backbone."""
    revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, fix_mistral_regex=True, **revision_kwargs
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name_or_path, **revision_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train helpfulness reward model (Bradley-Terry)")
    p.add_argument("--model-name-or-path", type=str, default="mistralai/Ministral-3-3B-Instruct-2512")
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--eval-dataset-path", type=Path, default=None,
                   help="Separate eval file (e.g. by-prompt holdout). If set, skips the internal split.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--regularization", type=float, default=0.001)
    p.add_argument("--eval-split-ratio", type=float, default=0.1,
                   help="Used only when --eval-dataset-path is not given.")
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooling", type=str, choices=["mean", "cls", "last-token"], default="last-token")
    p.add_argument("--concat-forward", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--load-in-half", action="store_true")
    p.add_argument("--device-map", action="store_true")
    p.add_argument("--adafactor", action="store_true")
    p.add_argument("--save-each-epoch", action=argparse.BooleanOptionalAction, default=True,
                   help="Save a backbone checkpoint after every epoch (epochN/ subdirs).")
    p.add_argument("--save-backbone", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-best-only", action="store_true",
                   help="Persist only best/ when validation loss improves.")
    p.add_argument("--lora-r", type=int, default=0,
                   help="Enable LoRA with this rank; zero preserves full fine-tuning.")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop after this many epochs without eval-loss improvement; zero disables.")
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo_root))
    from scripts.lib.run_manifest import (
        build_run_manifest,
        resolve_model_revision,
        write_run_manifest,
    )

    _manifest_inputs = [args.dataset_path]
    if args.eval_dataset_path is not None:
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
                _repo_root / "scripts" / "lib" / "run_manifest.py",
            ],
            extra={"stage": "reward_model_train"},
        ),
        args.output_dir,
    )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")

    # --- Data + split ---
    full_dataset = RewardPairDataset(args.dataset_path)
    if len(full_dataset) == 0:
        raise ValueError("No valid training pairs found.")
    print(f"Loaded train pairs: {len(full_dataset)} (skipped {full_dataset.skipped})")

    if args.eval_dataset_path is not None:
        eval_dataset = RewardPairDataset(args.eval_dataset_path)
        train_subset = full_dataset
        eval_subset = eval_dataset
        print(f"Eval pairs (separate file): {len(eval_dataset)} (skipped {eval_dataset.skipped})")
    else:
        n = len(full_dataset)
        idx = list(range(n))
        random.Random(args.seed).shuffle(idx)
        n_eval = max(1, int(n * args.eval_split_ratio))
        eval_subset = Subset(full_dataset, idx[:n_eval])
        train_subset = Subset(full_dataset, idx[n_eval:])
        print(f"Internal split -> train {len(train_subset)} | eval {len(eval_subset)}")

    # --- Tokenizer ---
    tokenizer = load_tokenizer(args.model_name_or_path, _model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # --- Model ---
    model = RewardModel(args.model_name_or_path, load_in_half=args.load_in_half,
                        pooling=args.pooling, use_bf16=args.bf16, use_device_map=args.device_map,
                        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, model_revision=_model_revision)
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    if args.device_map:
        print(f"device_map='auto' across {torch.cuda.device_count()} GPUs")
        model.score_head = model.score_head.to(model._last_device)
    else:
        model.to(device)

    model_max = getattr(model.backbone.config, "max_position_embeddings", None)
    eff_max_length = min(args.max_length, model_max) if isinstance(model_max, int) and model_max > 0 else args.max_length
    if eff_max_length < args.max_length:
        print(f"max_length capped to model limit: {eff_max_length}")

    if not args.device_map and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | trainable: {trainable_params:,} "
          f"({100 * trainable_params / max(1, total_params):.3f}%)")

    # --- Loaders ---
    collator = PairCollator(tokenizer=tokenizer, max_length=eff_max_length)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=device.type == "cuda")
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size * 2, shuffle=False,
                             collate_fn=collator, num_workers=2, pin_memory=device.type == "cuda")

    # --- Optimizer / scheduler ---
    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.adafactor:
        optimizer = Adafactor(optimizer_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False, warmup_init=False, weight_decay=args.weight_decay)
        print("Optimizer: Adafactor")
    else:
        optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        print("Optimizer: AdamW")

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)

    print("=" * 60)
    print(f"Reward training | model={args.model_name_or_path}")
    print(f"max_length={eff_max_length} batch={args.batch_size}x{args.gradient_accumulation_steps} "
          f"lr={args.learning_rate} epochs={args.epochs} reg={args.regularization}")
    print(f"train={len(train_subset)} eval={len(eval_subset)} updates/epoch={updates_per_epoch}")
    print("=" * 60)

    # --- Train loop ---
    start = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    early_stopper = EarlyStopper(args.early_stopping_patience, args.early_stopping_min_delta)

    for epoch in range(1, args.epochs + 1):
        model.train()
        step_loss = step_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step_idx, batch in enumerate(pbar, start=1):
            c_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            c_mask = batch["chosen_attention_mask"].to(device, non_blocking=True)
            r_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            r_mask = batch["rejected_attention_mask"].to(device, non_blocking=True)

            scores = model(c_ids, c_mask, r_ids, r_mask, concat_forward=args.concat_forward)
            result = compute_reward_loss(scores["chosen_scores"], scores["rejected_scores"],
                                         regularization=args.regularization)
            (result["loss"] / args.gradient_accumulation_steps).backward()
            step_loss += result["loss"].item() / args.gradient_accumulation_steps
            step_acc += result["accuracy"].item() / args.gradient_accumulation_steps

            if step_idx % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % args.log_steps == 0 or global_step == 1:
                    pbar.set_postfix(loss=f"{step_loss:.4f}", acc=f"{step_acc:.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")
                train_history.append({"step": global_step, "epoch": epoch,
                                      "loss": step_loss, "accuracy": step_acc,
                                      "learning_rate": scheduler.get_last_lr()[0]})
                step_loss = step_acc = 0.0

        # flush partial accumulation
        if len(train_loader) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        eval_metrics = evaluate(model, eval_loader, device,
                                regularization=args.regularization, concat_forward=args.concat_forward)
        print(f"\n--- Epoch {epoch} --- eval: loss={eval_metrics['loss']:.4f} "
              f"acc={eval_metrics['accuracy']:.4f} reward_mean={eval_metrics.get('reward_mean', float('nan')):.4f}\n")
        eval_metrics["epoch"] = epoch
        eval_history.append(eval_metrics)

        if args.save_each_epoch:
            ep_dir = args.output_dir / f"epoch{epoch}"
            print(f"Saving epoch {epoch} checkpoint -> {ep_dir}")
            save_checkpoint(model, tokenizer, ep_dir, args.save_backbone)

        if args.save_best_only and early_stopper.is_improvement(eval_metrics["loss"]):
            best_dir = args.output_dir / "best"
            print(f"New best validation loss; saving -> {best_dir}")
            save_checkpoint(model, tokenizer, best_dir, args.save_backbone)

        if early_stopper.update(eval_metrics["loss"]):
            print(f"Early stopping after epoch {epoch}: no eval-loss improvement for "
                  f"{args.early_stopping_patience} epoch(s)")
            break

    elapsed = time.time() - start

    # Select by loss: accuracy is discrete and hid severe overfitting in prior runs.
    best = select_best_epoch(eval_history)
    if not args.save_each_epoch and not args.save_best_only:
        save_checkpoint(model, tokenizer, args.output_dir, args.save_backbone)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": str(args.dataset_path),
        "eval_dataset_path": str(args.eval_dataset_path) if args.eval_dataset_path else None,
        "num_train_pairs": len(train_subset),
        "num_eval_pairs": len(eval_subset),
        "effective_max_length": eff_max_length,
        "epochs_requested": args.epochs,
        "epochs_completed": len(eval_history),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "pooling": args.pooling,
        "loss_type": "bradley_terry_sequence_wise",
        "train_seconds": elapsed,
        "best_epoch": best["epoch"] if best else None,
        "best_eval_accuracy": best["accuracy"] if best else None,
        "best_eval_loss": best["loss"] if best else None,
        "selection_metric": "eval_loss",
        "lora": build_checkpoint_metadata(
            args.model_name_or_path, args.pooling, args.lora_r,
            args.lora_alpha, args.lora_dropout,
        )["lora"],
        "trainable_params": trainable_params,
        "eval_history": eval_history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.output_dir / "training_log.json").write_text(json.dumps(train_history, ensure_ascii=False, indent=2))
    (args.output_dir / "eval_log.json").write_text(json.dumps(eval_history, ensure_ascii=False, indent=2))
    (args.output_dir / "arguments.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

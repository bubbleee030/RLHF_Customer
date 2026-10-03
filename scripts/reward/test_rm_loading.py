#!/usr/bin/env python3
"""Compare our reward-model loading and input-format choices on one pair.

This diagnostic deliberately loads the trained external ``score_head.pt``
instead of constructing ``AutoModelForSequenceClassification``.  The latter's
Ministral score layer may use ``bias=False`` and silently ignore our saved
``score.bias``.

Examples:
  # Original RM: base model + LoRA adapter + external score head
  python3 scripts/reward/test_rm_loading.py \
    --checkpoint reward_output/run_reward_cs_within_20260803/best \
    --load-mode lora --template-mode training --device cuda:1

  # Merged backbone + the same external score-head convention
  python3 scripts/reward/test_rm_loading.py \
    --checkpoint /workspace/merged_reward_model \
    --load-mode full --template-mode chat --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class LoadSpec:
    checkpoint: Path
    load_mode: str
    base_model_name_or_path: str


@dataclass
class LoadedRewardModel:
    backbone: Any
    score_head: nn.Linear
    tokenizer: Any
    spec: LoadSpec
    device: str
    requested_dtype: torch.dtype


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rendered_text: str
    token_count: int
    pooled_token_index: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load our RM as LoRA or merged-full, independently select the "
            "training or chat-template input format, and print one scalar."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--load-mode", choices=("lora", "full"), required=True)
    parser.add_argument(
        "--template-mode", choices=("training", "chat"), default="training",
    )
    parser.add_argument(
        "--base-model",
        help=(
            "LoRA base-model override. By default it is read from "
            "reward_model_config.json."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--prompt", default="你好，請介紹一下你自己。")
    parser.add_argument("--response", default="我是 Ministral，很高興為您服務！")
    return parser


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _adapter_exists(checkpoint: Path) -> bool:
    return any(
        (checkpoint / filename).exists()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def checkpoint_load_spec(
    checkpoint: Path,
    load_mode: str,
    base_model_override: str | None = None,
) -> LoadSpec:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "score_head.pt").is_file():
        raise FileNotFoundError(
            f"Missing external reward head: {checkpoint / 'score_head.pt'}"
        )

    has_adapter = _adapter_exists(checkpoint)
    if load_mode == "full":
        if has_adapter:
            raise ValueError(
                "--load-mode full received a LoRA adapter directory. "
                "Use --load-mode lora, or point --checkpoint at the merged model."
            )
        return LoadSpec(checkpoint, load_mode, str(checkpoint))

    if load_mode != "lora":
        raise ValueError(f"Unsupported load mode: {load_mode}")
    if not has_adapter:
        raise FileNotFoundError(
            f"--load-mode lora requires adapter_model.safetensors or "
            f"adapter_model.bin in {checkpoint}"
        )

    base_model = base_model_override
    metadata_path = checkpoint / "reward_model_config.json"
    if not base_model and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        base_model = metadata.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "Cannot resolve the LoRA base model. Add "
            "base_model_name_or_path to reward_model_config.json or pass "
            "--base-model explicitly."
        )
    return LoadSpec(checkpoint, load_mode, str(base_model))


def _extract_language_model(raw: Any) -> Any:
    if hasattr(raw, "language_model"):
        return raw.language_model
    if hasattr(raw, "model") and hasattr(raw.model, "language_model"):
        return raw.model.language_model
    if hasattr(raw, "model") and hasattr(raw.model, "embed_tokens"):
        return raw.model
    return raw


def _hidden_size(backbone: Any) -> int:
    config = backbone.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Cannot infer hidden_size from the loaded backbone config.")
    return int(hidden_size)


def load_score_head(checkpoint: Path, hidden_size: int) -> nn.Linear:
    head_path = Path(checkpoint) / "score_head.pt"
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    score_head = nn.Linear(hidden_size, 1, bias=True).float()
    # strict=True is intentional: missing or unexpected weight/bias must fail.
    score_head.load_state_dict(state, strict=True)
    return score_head.eval()


def load_reward_model(
    checkpoint: Path,
    load_mode: str,
    base_model_override: str | None,
    device: str,
    dtype: torch.dtype,
    *,
    auto_model_cls: Any | None = None,
    auto_tokenizer_cls: Any | None = None,
    peft_model_cls: Any | None = None,
) -> LoadedRewardModel:
    spec = checkpoint_load_spec(checkpoint, load_mode, base_model_override)

    if auto_model_cls is None or auto_tokenizer_cls is None:
        from transformers import AutoModel, AutoTokenizer

        auto_model_cls = auto_model_cls or AutoModel
        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer

    tokenizer = auto_tokenizer_cls.from_pretrained(str(spec.checkpoint))
    if spec.load_mode == "lora":
        if peft_model_cls is None:
            from peft import PeftModel

            peft_model_cls = PeftModel
        raw = auto_model_cls.from_pretrained(
            spec.base_model_name_or_path,
            dtype=dtype,
        )
        base_backbone = _extract_language_model(raw)
        backbone = peft_model_cls.from_pretrained(
            base_backbone,
            str(spec.checkpoint),
        ).merge_and_unload()
    else:
        raw = auto_model_cls.from_pretrained(str(spec.checkpoint), dtype=dtype)
        backbone = _extract_language_model(raw)

    backbone = backbone.to(device).eval()
    score_head = load_score_head(spec.checkpoint, _hidden_size(backbone))
    return LoadedRewardModel(
        backbone=backbone,
        score_head=score_head,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        requested_dtype=dtype,
    )


def render_input(
    tokenizer: Any,
    prompt: str,
    response: str,
    template_mode: str,
) -> str:
    if template_mode == "training":
        return f"User: {prompt}\nAssistant: {response}"
    if template_mode == "chat":
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    raise ValueError(f"Unsupported template mode: {template_mode}")


def last_nonpadding_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if last_hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden state [B,L,H] and attention mask [B,L].")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden-state and attention-mask batch/sequence shapes differ.")
    if not torch.all(attention_mask.bool().any(dim=1)):
        raise ValueError("Every input must contain at least one non-padding token.")

    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device,
    ).unsqueeze(0).expand_as(attention_mask)
    indexes = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    batch = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    pooled = last_hidden_state[batch, indexes.to(last_hidden_state.device)]
    return pooled, indexes


@torch.inference_mode()
def score_pair(
    loaded: LoadedRewardModel,
    prompt: str,
    response: str,
    template_mode: str,
    max_length: int,
) -> ScoreResult:
    rendered = render_input(loaded.tokenizer, prompt, response, template_mode)
    encoded = loaded.tokenizer(
        rendered,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "to"):
        encoded = encoded.to(loaded.device)
    else:
        encoded = {key: value.to(loaded.device) for key, value in encoded.items()}

    outputs = loaded.backbone(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False,
    )
    pooled, indexes = last_nonpadding_pool(
        outputs.last_hidden_state,
        encoded["attention_mask"],
    )
    # Match scripts/serve/gate_rank.py: FP32 pooled state and head on CPU.
    score = loaded.score_head(pooled[0].float().cpu()).item()
    return ScoreResult(
        score=score,
        rendered_text=rendered,
        token_count=int(encoded["attention_mask"][0].sum().item()),
        pooled_token_index=int(indexes[0].item()),
    )


def _first_parameter_facts(backbone: Any) -> tuple[str, str]:
    parameter = next(backbone.parameters())
    return str(parameter.device), str(parameter.dtype)


def main() -> int:
    args = build_parser().parse_args()
    if args.max_length <= 0:
        raise SystemExit("--max-length must be greater than zero")

    dtype = resolve_dtype(args.dtype)
    loaded = load_reward_model(
        checkpoint=args.checkpoint,
        load_mode=args.load_mode,
        base_model_override=args.base_model,
        device=args.device,
        dtype=dtype,
    )
    result = score_pair(
        loaded,
        prompt=args.prompt,
        response=args.response,
        template_mode=args.template_mode,
        max_length=args.max_length,
    )
    actual_device, actual_dtype = _first_parameter_facts(loaded.backbone)

    print("=== Reward Model Loading Test ===")
    print(f"checkpoint       : {loaded.spec.checkpoint}")
    print(f"load mode       : {loaded.spec.load_mode}")
    if loaded.spec.load_mode == "lora":
        print(f"base model      : {loaded.spec.base_model_name_or_path}")
    print(f"template mode   : {args.template_mode}")
    print(f"backbone class  : {type(loaded.backbone).__name__}")
    print(f"backbone device : {actual_device}")
    print(f"backbone dtype  : {actual_dtype} (requested {dtype})")
    print(f"head weight     : {tuple(loaded.score_head.weight.shape)}")
    print(f"head bias       : {tuple(loaded.score_head.bias.shape)} "
          f"value={loaded.score_head.bias.item():+.8f}")
    print(f"token count     : {result.token_count}")
    print(f"pooled index    : {result.pooled_token_index} (last non-padding token)")
    print("--- Rendered input ---")
    print(result.rendered_text)
    print("--- Result ---")
    print(f"Pred Score: {result.score:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

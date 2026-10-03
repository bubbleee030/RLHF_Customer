#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import time
from pathlib import Path

from scripts.policy_eval.policy_config import serialize_customer_input
from scripts.policy_eval.preflight import LowSpaceError, MIN_FREE_BYTES
from scripts.policy_eval.resume_io import (
    append_checkpoint,
    job_key,
    load_checkpoint,
    stable_sha256,
)


VARIANTS = (
    "base_raw",
    "base_policy_zh",
    "base_policy_bilingual",
    "ppo_raw",
)


def sample_seeds(prompt_sha: str, n: int = 3, root_seed: int = 42) -> list[int]:
    seeds: list[int] = []
    for index in range(n):
        digest = hashlib.sha256(
            f"{root_seed}\0{prompt_sha}\0{index}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
        while seed in seeds:
            seed = (seed + 1) % (2**31 - 1)
        seeds.append(seed)
    return seeds


def input_text(variant: str, prompt: str, compiled: dict[str, str]) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant in {"base_raw", "ppo_raw"}:
        return serialize_customer_input(prompt, None)
    if variant == "base_policy_zh":
        return serialize_customer_input(prompt, compiled["zh"])
    return serialize_customer_input(prompt, compiled["bilingual"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("manifest is empty or contains duplicate prompt_id")
    return rows


def _generate_one(actor, serialized: str, seed: int, config: dict) -> dict:
    import torch

    from scripts.ppo_lag.models_ppo import strip_think_response

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = actor.tokenizer(
        serialized,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(actor.device)
    started = time.monotonic()
    with torch.inference_mode():
        output = actor.model.generate(
            **encoded,
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_new_tokens=config["max_new_tokens"],
            pad_token_id=actor.tokenizer.pad_token_id or actor.tokenizer.eos_token_id,
        )
    prompt_tokens = encoded["input_ids"].shape[1]
    generated_ids = output[0, prompt_tokens:]
    raw = actor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    raw = raw.replace("</s>", "").replace("<pad>", "").strip()
    visible = strip_think_response(raw)
    return {
        "raw_response": raw,
        "visible_answer": visible,
        "prompt_tokens": int(prompt_tokens),
        "generation_tokens": int(generated_ids.numel()),
        "elapsed_seconds": time.monotonic() - started,
    }


def run_worker(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown variant: {args.variant}")
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    manifest = _load_manifest(manifest_path)
    manifest_hash = _file_sha256(manifest_path)
    compiled = {
        "zh": Path(args.compiled_zh).read_text(encoding="utf-8").strip(),
        "bilingual": Path(args.compiled_bilingual).read_text(encoding="utf-8").strip(),
    }
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "n": args.n,
        "root_seed": args.root_seed,
        "actor_model": str(Path(args.actor_model).resolve()),
        "adapter_dir": str(Path(args.adapter_dir).resolve()) if args.variant == "ppo_raw" else None,
        "serialization": "customer_inst_policy_eval_v1",
    }
    config_hash = stable_sha256(generation_config)
    completed = load_checkpoint(output_path)

    from scripts.ppo_lag.models_ppo import LoRAActor

    actor = LoRAActor(
        device=args.device,
        model_id=args.actor_model,
        prompt_format="customer_inst",
    )
    if args.variant == "ppo_raw":
        actor.model.load_adapter(args.adapter_dir, adapter_name="ppo")
        actor.model.set_adapter("ppo")
        adapter_context = contextlib.nullcontext
    else:
        adapter_context = actor.model.disable_adapter

    expected = len(manifest) * args.n
    for prompt_row in manifest:
        for sample_index, seed in enumerate(
            sample_seeds(prompt_row["prompt_sha256"], args.n, args.root_seed)
        ):
            key = job_key(
                manifest_hash,
                prompt_row["prompt_sha256"],
                args.variant,
                seed,
                config_hash,
            )
            if key in completed:
                continue
            free_bytes = shutil.disk_usage(output_path.parent).free
            if free_bytes < args.min_free_bytes:
                raise LowSpaceError(free_bytes, str(output_path.parent))
            serialized = input_text(args.variant, prompt_row["prompt"], compiled)
            with adapter_context():
                generated = _generate_one(actor, serialized, seed, generation_config)
            row = {
                "job_key": key,
                "status": "complete",
                "manifest_sha256": manifest_hash,
                "generation_config_sha256": config_hash,
                "prompt_id": prompt_row["prompt_id"],
                "prompt_sha256": prompt_row["prompt_sha256"],
                "prompt": prompt_row["prompt"],
                "tracks": prompt_row["tracks"],
                "ppo_seen": prompt_row["ppo_seen"],
                "variant": args.variant,
                "sample_index": sample_index,
                "seed": seed,
                **generated,
            }
            append_checkpoint(output_path, row, completed)
            print(
                json.dumps(
                    {"variant": args.variant, "complete": len(completed), "expected": expected},
                    separators=(",", ":"),
                ),
                flush=True,
            )
    if len(completed) != expected:
        raise RuntimeError(f"completion mismatch: {len(completed)} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compiled-zh", required=True)
    parser.add_argument("--compiled-bilingual", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--root-seed", type=int, default=42)
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)
    run_worker(parser.parse_args())


if __name__ == "__main__":
    main()

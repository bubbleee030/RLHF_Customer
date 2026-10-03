#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestInputs:
    sealed: Path
    eval_all: Path
    redteam: Path
    seeds: Path
    ppo_prompts: Path


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    prompt_sha256: str
    tracks: tuple[str, ...]
    ppo_seen: bool
    sources: tuple[dict, ...]


def _normalized(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    return prompt.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_unique_jsonl_prompts(path: str | Path, key: str = "input") -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if key not in row:
                raise ValueError(f"{path}:{line_no} missing {key}")
            prompt = _normalized(row[key])
            if prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    return prompts


def load_redteam_prompts(path: str | Path) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) != 18:
        raise ValueError("redteam source must contain exactly 18 top-level scenarios")
    result: list[tuple[str, str]] = []
    for scenario_id, row in data.items():
        if not isinstance(row, dict) or "prompt" not in row:
            raise ValueError(f"redteam scenario {scenario_id!r} has no prompt")
        result.append((str(scenario_id), _normalized(row["prompt"])))
    return result


def load_clean_policy_seeds(
    path: str | Path, ppo_prompts: set[str]
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            section = row.get("section")
            if not (
                section == "A1"
                or section == "A3"
                or (isinstance(section, str) and section.startswith("A2-"))
            ):
                continue
            prompt = _normalized(row.get("prompt"))
            if prompt in ppo_prompts or prompt in seen:
                continue
            rows.append({**row, "prompt": prompt, "source_line": line_no})
            seen.add(prompt)
    return rows


def build_prompt_manifest(inputs: ManifestInputs) -> list[PromptRecord]:
    ppo_data = json.loads(Path(inputs.ppo_prompts).read_text(encoding="utf-8"))
    if not isinstance(ppo_data, list):
        raise ValueError("PPO prompts must be a JSON list")
    ppo_prompts = {_normalized(prompt) for prompt in ppo_data}

    merged: dict[str, dict] = {}

    def add(prompt: str, track: str, source: dict) -> None:
        normalized = _normalized(prompt)
        entry = merged.setdefault(
            normalized,
            {"prompt": normalized, "tracks": set(), "sources": []},
        )
        entry["tracks"].add(track)
        entry["sources"].append(source)

    for index, prompt in enumerate(load_unique_jsonl_prompts(inputs.sealed)):
        add(prompt, "sealed_customer", {"source": "sealed_customer", "index": index})
    for index, prompt in enumerate(load_unique_jsonl_prompts(inputs.eval_all)):
        add(prompt, "eval_all", {"source": "eval_all", "index": index})
        if prompt not in ppo_prompts:
            add(prompt, "eval_clean", {"source": "eval_all", "index": index, "view": "clean"})
    for scenario_id, prompt in load_redteam_prompts(inputs.redteam):
        add(prompt, "redteam", {"source": "redteam", "scenario_id": scenario_id})
    for row in load_clean_policy_seeds(inputs.seeds, ppo_prompts):
        add(
            row["prompt"],
            "clean_policy_seeds",
            {
                "source": "seed_prompts",
                "section": row["section"],
                "severity": row.get("severity"),
                "source_line": row["source_line"],
            },
        )

    records: list[PromptRecord] = []
    for prompt, entry in merged.items():
        digest = _sha256(prompt)
        records.append(
            PromptRecord(
                prompt_id=f"p-{digest[:16]}",
                prompt=entry["prompt"],
                prompt_sha256=digest,
                tracks=tuple(sorted(entry["tracks"])),
                ppo_seen=prompt in ppo_prompts,
                sources=tuple(entry["sources"]),
            )
        )
    return sorted(records, key=lambda row: row.prompt_sha256)


def count_track(rows: list[PromptRecord], track: str) -> int:
    return sum(track in row.tracks for row in rows)


def _write_jsonl_atomic(path: Path, rows: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--eval-all", required=True)
    parser.add_argument("--redteam", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--ppo-prompts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_prompt_manifest(
        ManifestInputs(
            sealed=Path(args.sealed),
            eval_all=Path(args.eval_all),
            redteam=Path(args.redteam),
            seeds=Path(args.seeds),
            ppo_prompts=Path(args.ppo_prompts),
        )
    )
    _write_jsonl_atomic(Path(args.output), rows)
    print(
        f"unique={len(rows)} sealed={count_track(rows, 'sealed_customer')} "
        f"eval_all={count_track(rows, 'eval_all')} eval_clean={count_track(rows, 'eval_clean')} "
        f"redteam={count_track(rows, 'redteam')} "
        f"clean_policy_seeds={count_track(rows, 'clean_policy_seeds')} "
        f"eval_all_ppo_seen={sum('eval_all' in row.tracks and row.ppo_seen for row in rows)}"
    )


if __name__ == "__main__":
    main()

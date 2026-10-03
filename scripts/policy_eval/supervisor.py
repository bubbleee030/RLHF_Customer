#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.policy_eval.generate_responses import VARIANTS
from scripts.policy_eval.preflight import (
    MIN_FREE_BYTES,
    PreflightConfig,
    classify_failure,
    run_preflight,
)
from scripts.policy_eval.resume_io import atomic_write_json, load_checkpoint


STAGES = ("preflight", "generation", "scoring", "judging", "aggregation", "verification")


def container_pythonpath() -> str:
    return "/deps:/eval_code:/workspace:/workspace/scripts/ppo_lag"


def docker_gpu_request(gpus: str) -> str:
    return f'"device={gpus}"' if "," in gpus else f"device={gpus}"


def container_cache_mount(hf_cache_dir: str) -> str:
    return f"{hf_cache_dir}:/root/.cache/huggingface:ro"


@dataclass
class SupervisorState:
    status: str = "running"
    current_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    retries: dict[str, int] = field(default_factory=dict)
    block_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def next_stage(self) -> str | None:
        return next((stage for stage in STAGES if stage not in self.completed_stages), None)

    def block(self, reason: str) -> None:
        self.status = "blocked"
        self.block_reason = reason
        self.updated_at = time.time()

    def complete_stage(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.current_stage = None
        self.updated_at = time.time()

    def mark_complete(self) -> None:
        self.status = "complete"
        self.block_reason = None
        self.current_stage = None
        self.updated_at = time.time()


def retry_delay(attempt: int) -> int:
    return min(300, 2 ** max(0, int(attempt) - 1))


def classify_worker_failure(stderr_tail: str, attempt: int) -> str:
    text = stderr_tail.lower()
    permanent_markers = (
        "low space",
        "no space left",
        "nchc_api_key is required",
        "http 401",
        "http 403",
        "unexpected docker image",
        "hostname",
        "artifact",
        "completion mismatch",
        "input count mismatch",
        "expected four v100",
    )
    if any(marker in text for marker in permanent_markers):
        return "permanent"
    if "out of memory" in text or "cuda oom" in text:
        return "transient" if attempt < 2 else "permanent"
    return "transient"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle(code_dir: Path) -> list[str]:
    checksum_file = code_dir / "SHA256SUMS"
    if not checksum_file.is_file():
        raise RuntimeError("bundle SHA256SUMS is missing")
    files: list[str] = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError("invalid bundle checksum entry")
        path = code_dir / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
            raise RuntimeError(f"bundle hash mismatch: {relative}")
        files.append(str(path))
    return files


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workspace = Path(args.workspace).resolve()
        self.code_dir = Path(args.code_dir).resolve()
        self.deps_dir = Path(args.deps_dir).resolve()
        self.hf_cache_dir = Path(args.hf_cache_dir).resolve()
        self.run_dir = Path(args.run_dir).resolve()
        self.status_path = self.run_dir / "STATUS.json"
        self.logs_dir = self.run_dir / "logs"
        self.responses_dir = self.run_dir / "responses"
        self.image = args.image
        self.run_id = self.run_dir.name
        self.n_prompts = 1 if args.mode == "smoke" else 163
        self.n_samples = 1 if args.mode == "smoke" else 3
        self.expected_responses = self.n_prompts * self.n_samples * len(VARIANTS)
        self.expected_groups = self.n_prompts * self.n_samples
        if self.status_path.exists():
            raw = json.loads(self.status_path.read_text(encoding="utf-8"))
            fields = SupervisorState.__dataclass_fields__
            self.state = SupervisorState(**{key: value for key, value in raw.items() if key in fields})
        else:
            self.state = SupervisorState()

    def save(self) -> None:
        self.state.updated_at = time.time()
        payload = asdict(self.state)
        payload.update(
            {
                "run_id": self.run_id,
                "mode": self.args.mode,
                "expected": {
                    "prompts": self.n_prompts,
                    "responses": self.expected_responses,
                    "scores": self.expected_responses,
                    "judge_runs": self.expected_groups * 2,
                },
            }
        )
        atomic_write_json(self.status_path, payload)

    def _check_space(self) -> None:
        for path in (Path("/home/ubuntu"), Path("/var/lib/docker")):
            free = shutil.disk_usage(path).free
            if free < self.args.min_free_bytes:
                raise RuntimeError(f"low space at {path}: {free} bytes")

    def _docker_python(self, gpus: str, module: str, module_args: list[str]) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            docker_gpu_request(gpus),
            "--shm-size=10g",
            "--label",
            f"policy_eval_run={self.run_id}",
            "-v",
            f"{self.workspace}:/workspace:ro",
            "-v",
            f"{self.code_dir}:/eval_code:ro",
            "-v",
            f"{self.deps_dir}:/deps:ro",
            "-v",
            container_cache_mount(str(self.hf_cache_dir)),
            "-v",
            f"{self.run_dir}:/run:rw",
            "-w",
            "/workspace",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            f"PYTHONPATH={container_pythonpath()}",
            "--entrypoint",
            "/opt/conda/bin/python3",
            self.image,
            "-m",
            module,
            *module_args,
        ]

    def _manifest_paths(self) -> tuple[Path, str]:
        full_host = self.code_dir / "artifacts/prompt_manifest.jsonl"
        if self.args.mode == "full":
            return full_host, "/eval_code/artifacts/prompt_manifest.jsonl"
        smoke_host = self.run_dir / "prompt_manifest.smoke.jsonl"
        if not smoke_host.exists():
            first = full_host.read_text(encoding="utf-8").splitlines()[0]
            tmp = smoke_host.with_suffix(smoke_host.suffix + ".tmp")
            tmp.write_text(first + "\n", encoding="utf-8")
            os.replace(tmp, smoke_host)
        return smoke_host, "/run/prompt_manifest.smoke.jsonl"

    def preflight(self) -> None:
        if "preflight" in self.state.completed_stages:
            return
        if self.run_dir.exists():
            raise RuntimeError(f"new run directory already exists: {self.run_dir}")
        source_files = _verify_bundle(self.code_dir)
        config = PreflightConfig(
            expected_hostname=self.args.expected_hostname,
            image_name=self.image,
            expected_image_id=self.args.expected_image_id,
            artifact_paths={
                "actor": str(self.workspace / "model/costomer_model"),
                "ppo_adapter": str(self.workspace / "ppo_output/run_customer_8b_20260803_final/final/actor_adapter"),
                "rm": str(self.workspace / "reward_output/run_reward_cs_within_20260803/best"),
                "cm": str(self.workspace / "cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss"),
                "peft_deps": str(self.deps_dir),
                "hf_ministral3b": str(
                    self.hf_cache_dir
                    / "hub/models--mistralai--Ministral-3-3B-Instruct-2512"
                ),
            },
            run_dir=str(self.run_dir),
            source_files=source_files,
            key_file=self.args.key_file,
            require_key=True,
            min_free_bytes=self.args.min_free_bytes,
        )
        result = run_preflight(config)
        self.run_dir.mkdir(parents=False, exist_ok=False)
        self.logs_dir.mkdir()
        self.responses_dir.mkdir()
        atomic_write_json(self.run_dir / "PREFLIGHT.json", result)
        self.state.complete_stage("preflight")
        self.save()

    def _response_container_paths(self) -> list[str]:
        return [f"/run/responses/{variant}.jsonl" for variant in VARIANTS]

    def generation(self) -> None:
        if "generation" in self.state.completed_stages:
            return
        self._check_space()
        _, container_manifest = self._manifest_paths()
        pending = set(VARIANTS)
        gpu_by_variant = {variant: str(index) for index, variant in enumerate(VARIANTS)}
        while pending:
            processes: dict[str, tuple[subprocess.Popen, object, Path, int]] = {}
            for variant in sorted(pending):
                retry_key = f"generation:{variant}"
                attempt = self.state.retries.get(retry_key, 0) + 1
                self.state.retries[retry_key] = attempt
                self.save()
                log_path = self.logs_dir / f"generation_{variant}_attempt{attempt}.log"
                log_handle = log_path.open("ab", buffering=0)
                command = self._docker_python(
                    gpu_by_variant[variant],
                    "scripts.policy_eval.generate_responses",
                    [
                        "--variant", variant,
                        "--manifest", container_manifest,
                        "--compiled-zh", "/eval_code/artifacts/system_prompt_zh.txt",
                        "--compiled-bilingual", "/eval_code/artifacts/system_prompt_bilingual.txt",
                        "--actor-model", "/workspace/model/costomer_model",
                        "--adapter-dir", "/workspace/ppo_output/run_customer_8b_20260803_final/final/actor_adapter",
                        "--output", f"/run/responses/{variant}.jsonl",
                        "--n", str(self.n_samples),
                        "--root-seed", "42",
                        "--max-new-tokens", "224",
                        "--min-free-bytes", str(self.args.min_free_bytes),
                    ],
                )
                processes[variant] = (
                    subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT),
                    log_handle,
                    log_path,
                    attempt,
                )
            retrying: set[str] = set()
            for variant, (process, log_handle, log_path, attempt) in processes.items():
                return_code = process.wait()
                log_handle.close()
                if return_code == 0:
                    pending.remove(variant)
                    continue
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
                if classify_worker_failure(tail, attempt) == "permanent":
                    raise RuntimeError(f"generation {variant} permanently failed: {tail[-2000:]}")
                retrying.add(variant)
            if retrying:
                delay = max(retry_delay(self.state.retries[f"generation:{variant}"]) for variant in retrying)
                time.sleep(delay)
        for variant in VARIANTS:
            count = len(load_checkpoint(self.responses_dir / f"{variant}.jsonl"))
            expected = self.n_prompts * self.n_samples
            if count != expected:
                raise RuntimeError(f"completion mismatch for {variant}: {count} != {expected}")
        self.state.complete_stage("generation")
        self.save()

    def _run_until_success(
        self,
        stage: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        while True:
            self._check_space()
            attempt = self.state.retries.get(stage, 0) + 1
            self.state.retries[stage] = attempt
            self.save()
            log_path = self.logs_dir / f"{stage}_attempt{attempt}.log"
            with log_path.open("ab", buffering=0) as log_handle:
                return_code = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                ).returncode
            if return_code == 0:
                return
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            if classify_worker_failure(tail, attempt) == "permanent":
                raise RuntimeError(f"{stage} permanently failed: {tail[-2000:]}")
            time.sleep(retry_delay(attempt))

    def scoring(self) -> None:
        if "scoring" in self.state.completed_stages:
            return
        command = self._docker_python(
            "0,1",
            "scripts.policy_eval.score_responses",
            [
                "--responses", *self._response_container_paths(),
                "--rm-dir", "/workspace/reward_output/run_reward_cs_within_20260803/best",
                "--cm-dir", "/workspace/cost_output/run_ministral_3b_instruct_20260625_034415_len4096_best/best-loss",
                "--output", "/run/scores.jsonl",
                "--expected", str(self.expected_responses),
                "--min-free-bytes", str(self.args.min_free_bytes),
            ],
        )
        self._run_until_success("scoring", command)
        self.state.complete_stage("scoring")
        self.save()

    def _judge_environment(self) -> dict[str, str]:
        key_file = Path(self.args.key_file)
        env = os.environ.copy()
        found = False
        for line in key_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if name == "NCHC_API_KEY" and separator and value:
                env[name] = value
                found = True
        if not found:
            raise RuntimeError("NCHC_API_KEY is required")
        env["PYTHONPATH"] = f"{self.code_dir}:{self.workspace}"
        return env

    def judging(self) -> None:
        if "judging" in self.state.completed_stages:
            return
        command = [
            sys.executable,
            "-m",
            "scripts.policy_eval.judge_policy",
            "--responses",
            *[str(self.responses_dir / f"{variant}.jsonl") for variant in VARIANTS],
            "--policies",
            str(self.code_dir / "configs/policy_eval/example_policy.jsonl"),
            "--output",
            str(self.run_dir / "judges.jsonl"),
            "--expected-groups",
            str(self.expected_groups),
            "--min-free-bytes",
            str(self.args.min_free_bytes),
        ]
        self._run_until_success("judging", command, env=self._judge_environment())
        self.state.complete_stage("judging")
        self.save()

    def aggregation(self) -> None:
        if "aggregation" in self.state.completed_stages:
            return
        host_manifest, _ = self._manifest_paths()
        command = [
            sys.executable,
            "-m",
            "scripts.policy_eval.aggregate_report",
            "--manifest",
            str(host_manifest),
            "--responses",
            *[str(self.responses_dir / f"{variant}.jsonl") for variant in VARIANTS],
            "--scores",
            str(self.run_dir / "scores.jsonl"),
            "--judges",
            str(self.run_dir / "judges.jsonl"),
            "--output-dir",
            str(self.run_dir / "report"),
            "--expected-prompts",
            str(self.n_prompts),
            "--expected-responses",
            str(self.expected_responses),
            "--expected-scores",
            str(self.expected_responses),
            "--expected-judges",
            str(self.expected_groups * 2),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.code_dir}:{self.workspace}"
        self._run_until_success("aggregation", command, env=env)
        self.state.complete_stage("aggregation")
        self.save()

    def verification(self) -> None:
        if "verification" in self.state.completed_stages:
            return
        self._check_space()
        responses = sum(
            len(load_checkpoint(self.responses_dir / f"{variant}.jsonl")) for variant in VARIANTS
        )
        scores = len(load_checkpoint(self.run_dir / "scores.jsonl"))
        judges = len(load_checkpoint(self.run_dir / "judges.jsonl"))
        expected = (self.expected_responses, self.expected_responses, self.expected_groups * 2)
        if (responses, scores, judges) != expected:
            raise RuntimeError(
                f"completion mismatch responses/scores/judges={(responses, scores, judges)} expected={expected}"
            )
        checksum_path = self.run_dir / "report/SHA256SUMS"
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.partition("  ")
            if _sha256(checksum_path.parent / name) != digest:
                raise RuntimeError(f"report checksum mismatch: {name}")
        orphaned = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", f"label=policy_eval_run={self.run_id}"],
            text=True,
        ).strip()
        if orphaned:
            raise RuntimeError(f"orphaned evaluation containers: {orphaned}")
        atomic_write_json(
            self.run_dir / "VERIFICATION.json",
            {
                "status": "passed",
                "responses": responses,
                "scores": scores,
                "judge_runs": judges,
                "missing_jobs": 0,
                "free_bytes": shutil.disk_usage("/home/ubuntu").free,
            },
        )
        self.state.complete_stage("verification")
        self.state.mark_complete()
        self.save()

    def run(self) -> None:
        try:
            self.preflight()
            for stage in STAGES[1:]:
                self.state.current_stage = stage
                self.save()
                getattr(self, stage)()
        except BaseException as error:
            if self.run_dir.exists():
                self.state.block(str(error))
                self.save()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--deps-dir", default="/home/ubuntu/policy_eval_deps_peft_20260818")
    parser.add_argument("--hf-cache-dir", default="/home/ubuntu/.cache/huggingface")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--image", default="cost-model-trainer:v2")
    parser.add_argument("--expected-hostname", default="vm1777862919292-5901600-iaas")
    parser.add_argument(
        "--expected-image-id",
        default="sha256:932b5dbbdca70d8a584c474102f3a36d29ecb7c95cd820093d9eddc55d706801",
    )
    parser.add_argument("--key-file", default="/home/ubuntu/.config/reward_model/nchc_api.env")
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)
    Supervisor(parser.parse_args()).run()


if __name__ == "__main__":
    main()

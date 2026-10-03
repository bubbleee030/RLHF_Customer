    #!/usr/bin/env python3
"""Visualize training metrics from trainer_state.json, tensorboard events, or training_log.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _find_trainer_states(output_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        candidates.append(root_state)

    for path in output_dir.rglob("trainer_state.json"):
        if path not in candidates:
            candidates.append(path)

    return sorted(candidates)


def _find_event_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("events.out.tfevents.*"))


def _find_training_logs(output_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    root_log = output_dir / "training_log.json"
    if root_log.exists():
        candidates.append(root_log)

    for path in output_dir.rglob("training_log.json"):
        if path not in candidates:
            candidates.append(path)

    return sorted(candidates)


def _load_logs(trainer_state_path: Path) -> list[dict[str, Any]]:
    with trainer_state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logs = data.get("log_history", [])
    if not isinstance(logs, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in logs:
        if isinstance(item, dict):
            cleaned.append(item)

    return cleaned


def _extract_series(logs: list[dict[str, Any]]) -> dict[str, list[float]]:
    steps: list[float] = []
    losses: list[float] = []
    learning_rates: list[float] = []
    epochs: list[float] = []
    accuracies: list[float] = []
    accuracy_signs: list[float] = []

    for item in logs:
        step = _safe_float(item.get("step"))
        if step is None:
            continue

        loss = _safe_float(item.get("loss"))
        lr = _safe_float(item.get("learning_rate"))
        epoch = _safe_float(item.get("epoch"))
        accuracy = _safe_float(item.get("accuracy"))
        accuracy_sign = _safe_float(item.get("accuracy_sign"))

        if loss is None and lr is None and accuracy is None and accuracy_sign is None:
            continue

        steps.append(step)
        losses.append(loss if loss is not None else float("nan"))
        learning_rates.append(lr if lr is not None else float("nan"))
        epochs.append(epoch if epoch is not None else float("nan"))
        accuracies.append(accuracy if accuracy is not None else float("nan"))
        accuracy_signs.append(accuracy_sign if accuracy_sign is not None else float("nan"))

    return {
        "steps": steps,
        "loss": losses,
        "learning_rate": learning_rates,
        "epoch": epochs,
        "accuracy": accuracies,
        "accuracy_sign": accuracy_signs,
    }


def _load_training_log_series(training_log_path: Path) -> dict[str, list[float]]:
    with training_log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid training_log.json format: {training_log_path}")

    logs: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            logs.append(item)

    return _extract_series(logs)


def _load_tensorboard_series(event_file: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "tensorboard is required to read event files. Install in uv env first."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    scalar_tags = set(accumulator.Tags().get("scalars", []))

    def _pick_tag(options: list[str]) -> str | None:
        for tag in options:
            if tag in scalar_tags:
                return tag
        return None

    def _to_map(tag: str | None) -> dict[int, float]:
        if tag is None:
            return {}
        values: dict[int, float] = {}
        for item in accumulator.Scalars(tag):
            values[int(item.step)] = float(item.value)
        return values

    step_map = _to_map(_pick_tag(["train/step", "step"]))
    loss_map = _to_map(_pick_tag(["train/loss", "loss"]))
    lr_map = _to_map(_pick_tag(["train/lr", "train/learning_rate", "learning_rate"]))
    epoch_map = _to_map(_pick_tag(["train/epoch", "epoch"]))
    accuracy_map = _to_map(_pick_tag(["train/accuracy", "accuracy"]))
    accuracy_sign_map = _to_map(_pick_tag(["train/accuracy_sign", "accuracy_sign"]))

    metric_maps = [loss_map, lr_map, epoch_map, accuracy_map, accuracy_sign_map]
    event_steps = sorted({k for metric_map in metric_maps for k in metric_map})
    if not event_steps and step_map:
        event_steps = sorted(step_map)

    steps: list[float] = []
    losses: list[float] = []
    learning_rates: list[float] = []
    epochs: list[float] = []
    accuracies: list[float] = []
    accuracy_signs: list[float] = []

    for event_step in event_steps:
        x_step = step_map.get(event_step, float(event_step))
        steps.append(float(x_step))
        losses.append(loss_map.get(event_step, float("nan")))
        learning_rates.append(lr_map.get(event_step, float("nan")))
        epochs.append(epoch_map.get(event_step, float("nan")))
        accuracies.append(accuracy_map.get(event_step, float("nan")))
        accuracy_signs.append(accuracy_sign_map.get(event_step, float("nan")))

    return {
        "steps": steps,
        "loss": losses,
        "learning_rate": learning_rates,
        "epoch": epochs,
        "accuracy": accuracies,
        "accuracy_sign": accuracy_signs,
    }


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values.copy()

    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = [v for v in values[start : i + 1] if not math.isnan(v)]
        if not chunk:
            result.append(float("nan"))
        else:
            result.append(sum(chunk) / len(chunk))
    return result


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []

    alpha = min(max(alpha, 0.0), 1.0)
    result: list[float] = []
    prev: float | None = None

    for v in values:
        if math.isnan(v):
            result.append(prev if prev is not None else float("nan"))
            continue

        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1.0 - alpha) * prev

        result.append(prev)

    return result


def _plot_series(
    series: dict[str, list[float]],
    output_dir: Path,
    prefix: str,
    smooth_window: int,
    ema_alpha: float,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install in uv env first."
        ) from exc

    saved_paths: list[Path] = []
    steps = series["steps"]

    if not steps:
        return saved_paths

    plt.style.use("seaborn-v0_8-whitegrid")

    if any(not math.isnan(v) for v in series["loss"]):
        raw_loss = series["loss"]
        loss_ma = _rolling_mean(raw_loss, smooth_window)
        loss_ema = _ema(raw_loss, ema_alpha)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, raw_loss, linewidth=1.2, alpha=0.35, color="#005f73", label="raw")
        ax.plot(steps, loss_ma, linewidth=2.0, color="#0a9396", label=f"MA({smooth_window})")
        ax.plot(steps, loss_ema, linewidth=2.0, color="#bb3e03", label=f"EMA(alpha={ema_alpha})")
        ax.set_title("Training Loss vs Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        loss_path = output_dir / f"{prefix}loss_curve.png"
        fig.tight_layout()
        fig.savefig(loss_path, dpi=160)
        plt.close(fig)
        saved_paths.append(loss_path)

    if any(not math.isnan(v) for v in series["learning_rate"]):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, series["learning_rate"], linewidth=2.0, color="#9b2226")
        ax.set_title("Learning Rate vs Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        lr_path = output_dir / f"{prefix}learning_rate_curve.png"
        fig.tight_layout()
        fig.savefig(lr_path, dpi=160)
        plt.close(fig)
        saved_paths.append(lr_path)

    if any(not math.isnan(v) for v in series["accuracy"]):
        raw_acc = series["accuracy"]
        acc_ma = _rolling_mean(raw_acc, smooth_window)
        acc_ema = _ema(raw_acc, ema_alpha)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, raw_acc, linewidth=1.2, alpha=0.35, color="#0a9396", label="raw")
        ax.plot(steps, acc_ma, linewidth=2.0, color="#005f73", label=f"MA({smooth_window})")
        ax.plot(steps, acc_ema, linewidth=2.0, color="#ca6702", label=f"EMA(alpha={ema_alpha})")
        ax.set_title("Training Accuracy vs Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best")
        acc_path = output_dir / f"{prefix}accuracy_curve.png"
        fig.tight_layout()
        fig.savefig(acc_path, dpi=160)
        plt.close(fig)
        saved_paths.append(acc_path)

    if any(not math.isnan(v) for v in series["accuracy_sign"]):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, series["accuracy_sign"], linewidth=2.0, color="#ca6702")
        ax.set_title("Training Accuracy Sign vs Global Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Accuracy Sign")
        acc_sign_path = output_dir / f"{prefix}accuracy_sign_curve.png"
        fig.tight_layout()
        fig.savefig(acc_sign_path, dpi=160)
        plt.close(fig)
        saved_paths.append(acc_sign_path)

    return saved_paths


def _nan_safe(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def _summarize(series: dict[str, list[float]], source_path: Path, source_type: str) -> dict[str, Any]:
    losses = _nan_safe(series["loss"])
    lrs = _nan_safe(series["learning_rate"])
    epochs = _nan_safe(series["epoch"])
    accuracies = _nan_safe(series["accuracy"])
    accuracy_signs = _nan_safe(series["accuracy_sign"])
    steps = series["steps"]

    return {
        "source_type": source_type,
        "source": str(source_path),
        "num_logged_points": len(steps),
        "max_step": max(steps) if steps else None,
        "last_step": steps[-1] if steps else None,
        "last_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "last_learning_rate": lrs[-1] if lrs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "last_accuracy": accuracies[-1] if accuracies else None,
        "best_accuracy": max(accuracies) if accuracies else None,
        "last_accuracy_sign": accuracy_signs[-1] if accuracy_signs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize training outputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save figures/JSON (default: output-dir/plots)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Rolling window size for smoothing raw curves.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA alpha for smoothing raw curves.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    save_dir = args.save_dir.expanduser().resolve() if args.save_dir else output_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer_states = _find_trainer_states(output_dir)
    event_files = _find_event_files(output_dir)
    training_logs = _find_training_logs(output_dir)

    if not trainer_states and not event_files and not training_logs:
        raise FileNotFoundError(
            f"No trainer_state.json, TensorBoard event files, or training_log.json found under: {output_dir}"
        )

    all_summaries: list[dict[str, Any]] = []
    all_images: list[Path] = []

    if trainer_states:
        source_count = len(trainer_states)
        for idx, trainer_state in enumerate(trainer_states, start=1):
            logs = _load_logs(trainer_state)
            series = _extract_series(logs)

            prefix = "" if source_count == 1 else f"run_{idx}_"
            image_paths = _plot_series(series, save_dir, prefix, args.smooth_window, args.ema_alpha)
            summary = _summarize(series, trainer_state, "trainer_state")
            summary["images"] = [str(p) for p in image_paths]

            summary_path = save_dir / f"{prefix}summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            all_summaries.append(summary)
            all_images.extend(image_paths)
    else:
        if event_files:
            source_count = len(event_files)
            for idx, event_file in enumerate(event_files, start=1):
                series = _load_tensorboard_series(event_file)

                prefix = "" if source_count == 1 else f"run_{idx}_"
                image_paths = _plot_series(series, save_dir, prefix, args.smooth_window, args.ema_alpha)
                summary = _summarize(series, event_file, "tensorboard_event")
                summary["images"] = [str(p) for p in image_paths]

                summary_path = save_dir / f"{prefix}summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                all_summaries.append(summary)
                all_images.extend(image_paths)
        else:
            source_count = len(training_logs)
            for idx, training_log in enumerate(training_logs, start=1):
                series = _load_training_log_series(training_log)

                prefix = "" if source_count == 1 else f"run_{idx}_"
                image_paths = _plot_series(series, save_dir, prefix, args.smooth_window, args.ema_alpha)
                summary = _summarize(series, training_log, "training_log")
                summary["images"] = [str(p) for p in image_paths]

                summary_path = save_dir / f"{prefix}summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                all_summaries.append(summary)
                all_images.extend(image_paths)

    aggregate_path = save_dir / "all_runs_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("Visualization complete.")
    print(f"Detected trainer states: {len(trainer_states)}")
    print(f"Detected event files: {len(event_files)}")
    print(f"Detected training logs: {len(training_logs)}")
    print(f"Summary file: {aggregate_path}")
    if all_images:
        print("Saved figures:")
        for image in all_images:
            print(f"- {image}")
    else:
        print("No drawable points found in logs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

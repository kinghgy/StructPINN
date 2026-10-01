from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

from .config import load_config
from .solver import predict_checkpoint, solve
from .utils import ensure_directory, save_json


def flatten_numeric(data: Any, prefix: str = "") -> dict[str, float]:
    """Return finite scalar values from a nested result using dotted keys."""
    flattened: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(value, child_prefix))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        value = float(data)
        if math.isfinite(value):
            flattened[prefix] = value
    return flattened


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics present in every run; ``std`` is the sample standard deviation."""
    if not summaries:
        raise ValueError("At least one run summary is required.")
    flattened = [flatten_numeric(summary) for summary in summaries]
    common_keys = set(flattened[0]).intersection(*(set(item) for item in flattened[1:]))
    statistics: dict[str, dict[str, float | int]] = {}
    for key in sorted(common_keys):
        values = [item[key] for item in flattened]
        statistics[key] = {
            "count": len(values),
            "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {
        "run_count": len(summaries),
        "standard_deviation_definition": "sample (n-1 denominator)",
        "statistics": statistics,
    }


def _load_completed_run(
    run_directory: Path,
    expected_seed: int,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metrics_path = run_directory / "metrics.json"
    config_path = run_directory / "config.resolved.yaml"
    model_path = run_directory / "model.pt"
    if not (metrics_path.is_file() and config_path.is_file() and model_path.is_file()):
        return None
    with config_path.open("r", encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream) or {}
    if int(resolved.get("training", {}).get("seed", -1)) != expected_seed:
        return None
    if expected_config is not None:
        serializable = {
            key: value for key, value in expected_config.items() if not key.startswith("_")
        }
        if resolved != serializable:
            return None
    with metrics_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _save_run_table(
    seeds: list[int], summaries: list[dict[str, Any]], output_root: Path
) -> None:
    rows = []
    for seed, summary in zip(seeds, summaries, strict=True):
        rows.append({"seed": seed, **flatten_numeric(summary)})
    fieldnames = ["seed", *sorted(set().union(*(set(row) for row in rows)) - {"seed"})]
    with (output_root / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed_sweep(
    config_path: str | Path,
    seeds: list[int],
    output_root: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Train one independent model per seed and save cross-run statistics."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")

    base_config = load_config(config_path)
    root = ensure_directory(output_root)
    summaries: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        run_directory = root / f"seed_{seed}"
        config = deepcopy(base_config)
        config["training"]["seed"] = seed
        config["output"]["directory"] = str(run_directory)
        existing = (
            _load_completed_run(run_directory, seed, config) if resume else None
        )
        if existing is not None:
            print(f"[{index}/{len(seeds)}] Seed {seed}: using completed run at {run_directory}.")
            summaries.append(existing)
            continue

        print(f"[{index}/{len(seeds)}] Seed {seed}: starting training.")
        summaries.append(solve(config))

    aggregate = aggregate_summaries(summaries)
    aggregate["seeds"] = seeds
    aggregate["source_config"] = str(Path(config_path).resolve())
    save_json(aggregate, root / "aggregate_metrics.json")
    _save_run_table(seeds, summaries, root)
    print(f"Sweep complete. Aggregate results: {(root / 'aggregate_metrics.json').resolve()}")
    return aggregate


def run_finetune_seed_sweep(
    config_path: str | Path,
    checkpoint_root: str | Path,
    seeds: list[int],
    output_root: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Fine-tune one source checkpoint per seed and aggregate the new metrics."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")

    base_config = load_config(config_path)
    source_root = Path(checkpoint_root)
    root = ensure_directory(output_root)
    summaries: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        checkpoint = source_root / f"seed_{seed}" / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing source checkpoint for seed {seed}: {checkpoint}")
        run_directory = root / f"seed_{seed}"
        config = deepcopy(base_config)
        config["training"]["seed"] = seed
        config["output"]["directory"] = str(run_directory)
        existing = (
            _load_completed_run(run_directory, seed, config) if resume else None
        )
        expected_source = str(checkpoint.resolve())
        if existing is not None and existing.get("initialized_from") == expected_source:
            print(f"[{index}/{len(seeds)}] Seed {seed}: using completed fine-tuned run.")
            summaries.append(existing)
            continue

        print(f"[{index}/{len(seeds)}] Seed {seed}: fine-tuning {checkpoint}.")
        summaries.append(solve(config, initialization_checkpoint=checkpoint))

    aggregate = aggregate_summaries(summaries)
    aggregate["seeds"] = seeds
    aggregate["source_config"] = str(Path(config_path).resolve())
    aggregate["checkpoint_root"] = str(source_root.resolve())
    save_json(aggregate, root / "aggregate_metrics.json")
    _save_run_table(seeds, summaries, root)
    print(f"Fine-tune sweep complete: {(root / 'aggregate_metrics.json').resolve()}")
    return aggregate


def evaluate_checkpoint_sweep(
    checkpoint_root: str | Path,
    seeds: list[int],
    output_root: str | Path,
    device: str = "auto",
    plate_field_source: str = "mixed",
) -> dict[str, Any]:
    """Evaluate matching seed checkpoints using a common plate field projection."""
    source_root = Path(checkpoint_root)
    root = ensure_directory(output_root)
    summaries: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        checkpoint = source_root / f"seed_{seed}" / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing source checkpoint for seed {seed}: {checkpoint}")
        run_directory = root / f"seed_{seed}"
        print(f"[{index}/{len(seeds)}] Evaluating seed {seed} with {plate_field_source} fields.")
        predict_checkpoint(checkpoint, run_directory, device, plate_field_source)
        with (run_directory / "metrics.json").open("r", encoding="utf-8") as stream:
            summaries.append(json.load(stream))
    aggregate = aggregate_summaries(summaries)
    aggregate["seeds"] = seeds
    aggregate["checkpoint_root"] = str(source_root.resolve())
    aggregate["plate_field_source"] = plate_field_source
    save_json(aggregate, root / "aggregate_metrics.json")
    _save_run_table(seeds, summaries, root)
    return aggregate

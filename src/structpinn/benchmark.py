from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter
from typing import Any

import torch

from .config import load_config
from .solver import _build_problem
from .trainer import train_problem
from .utils import ensure_directory, resolve_device, resolve_dtype, save_json, set_seed


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def benchmark_devices(
    config_path: str | Path,
    devices: list[str],
    repetitions: int,
    adam_epochs: int,
    output_directory: str | Path,
    dtype_name: str | None = None,
    lbfgs_max_iter: int = 0,
) -> dict[str, Any]:
    """Benchmark the actual training loop with identical seeds and configuration."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if adam_epochs < 1:
        raise ValueError("adam_epochs must be positive.")
    if lbfgs_max_iter < 0:
        raise ValueError("lbfgs_max_iter must be non-negative.")
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be a non-empty list without duplicates.")

    root = ensure_directory(output_directory)
    base = load_config(config_path)
    base_seed = int(base["training"]["seed"])
    rows: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        seed = base_seed + repetition
        for device_name in devices:
            device = resolve_device(device_name)
            config = deepcopy(base)
            training = config["training"]
            training["seed"] = seed
            training["device"] = device_name
            training["adam_epochs"] = adam_epochs
            training["lbfgs_max_iter"] = lbfgs_max_iter
            training["log_every"] = adam_epochs
            training["early_stopping"]["enabled"] = False
            if dtype_name is not None:
                training["dtype"] = dtype_name
            dtype = resolve_dtype(training["dtype"])

            set_seed(seed)
            problem = _build_problem(config).to(device=device, dtype=dtype)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            started = perf_counter()
            result = train_problem(problem, training, device, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_seconds = perf_counter() - started
            canonical_device = (
                f"cuda:{torch.cuda.current_device()}" if device.type == "cuda" else str(device)
            )
            row: dict[str, Any] = {
                "repetition": repetition + 1,
                "seed": seed,
                "device": canonical_device,
                "dtype": training["dtype"],
                "adam_epochs": adam_epochs,
                "wall_seconds": wall_seconds,
                "epochs_per_second": adam_epochs / wall_seconds,
                "final_logged_loss": result.history[-1]["total"],
            }
            if device.type == "cuda":
                row["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                row["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
            rows.append(row)
            print(
                f"Benchmark {device} repetition {repetition + 1}/{repetitions}: "
                f"{wall_seconds:.3f} s, {row['epochs_per_second']:.3f} epochs/s"
            )
            del problem
            if device.type == "cuda":
                torch.cuda.empty_cache()

    successful_devices = sorted({str(row["device"]) for row in rows})
    per_device: dict[str, Any] = {}
    for device in successful_devices:
        device_rows = [row for row in rows if row["device"] == device]
        per_device[device] = {
            "wall_seconds": _summary([float(row["wall_seconds"]) for row in device_rows]),
            "epochs_per_second": _summary(
                [float(row["epochs_per_second"]) for row in device_rows]
            ),
        }
        memory = [float(row["peak_allocated_mib"]) for row in device_rows if "peak_allocated_mib" in row]
        if memory:
            per_device[device]["peak_allocated_mib"] = _summary(memory)

    cpu_time = per_device.get("cpu", {}).get("wall_seconds", {}).get("mean")
    cuda_key = next((name for name in per_device if name.startswith("cuda")), None)
    cuda_time = per_device.get(cuda_key, {}).get("wall_seconds", {}).get("mean")
    speedup = float(cpu_time / cuda_time) if cpu_time and cuda_time else None
    report = {
        "source_config": str(Path(config_path).resolve()),
        "repetitions": repetitions,
        "adam_epochs": adam_epochs,
        "lbfgs_max_iter": lbfgs_max_iter,
        "dtype": dtype_name or base["training"]["dtype"],
        "collocation_points": int(base["training"]["collocation_points"]),
        "per_device": per_device,
        "cpu_over_cuda_speedup": speedup,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "runs": rows,
    }
    save_json(report, root / "benchmark.json")
    fields = sorted(set().union(*(set(row) for row in rows)))
    with (root / "benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return report

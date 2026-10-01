from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _beam_coordinate(x: torch.Tensor, length: float) -> torch.Tensor:
    return x / length


def beam_load_torch(x: torch.Tensor, length: float, config: dict[str, Any]) -> torch.Tensor:
    load_type = config.get("type", "uniform")
    coordinate = _beam_coordinate(x, length)
    if load_type == "uniform":
        return torch.full_like(x, float(config.get("value", 0.0)))
    if load_type == "polynomial":
        result = torch.zeros_like(x)
        for power, coefficient in enumerate(config.get("coefficients", [0.0])):
            result = result + float(coefficient) * coordinate**power
        return result
    if load_type == "sine":
        amplitude = float(config.get("amplitude", 1.0))
        frequency = float(config.get("frequency", 1.0))
        phase = float(config.get("phase", 0.0))
        return amplitude * torch.sin(frequency * torch.pi * coordinate + phase)
    if load_type == "piecewise_constant":
        result = torch.zeros_like(x)
        for segment in config.get("segments", []):
            start = float(segment["start"])
            end = float(segment["end"])
            value = float(segment["value"])
            mask = (x >= start) & (x <= end)
            result = torch.where(mask, torch.full_like(result, value), result)
        return result
    raise ValueError(f"Unsupported beam distributed load: {load_type}")


def beam_load_numpy(x: np.ndarray, length: float, config: dict[str, Any]) -> np.ndarray:
    load_type = config.get("type", "uniform")
    coordinate = x / length
    if load_type == "uniform":
        return np.full_like(x, float(config.get("value", 0.0)), dtype=float)
    if load_type == "polynomial":
        result = np.zeros_like(x, dtype=float)
        for power, coefficient in enumerate(config.get("coefficients", [0.0])):
            result += float(coefficient) * coordinate**power
        return result
    if load_type == "sine":
        return float(config.get("amplitude", 1.0)) * np.sin(
            float(config.get("frequency", 1.0)) * np.pi * coordinate
            + float(config.get("phase", 0.0))
        )
    if load_type == "piecewise_constant":
        result = np.zeros_like(x, dtype=float)
        for segment in config.get("segments", []):
            mask = (x >= float(segment["start"])) & (x <= float(segment["end"]))
            result[mask] = float(segment["value"])
        return result
    raise ValueError(f"Unsupported beam distributed load: {load_type}")


def plate_load_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    a: float,
    b: float,
    config: dict[str, Any],
) -> torch.Tensor:
    load_type = config.get("type", "uniform")
    if load_type == "uniform":
        return torch.full_like(x, float(config.get("value", 0.0)))
    if load_type == "sine":
        amplitude = float(config.get("amplitude", 1.0))
        mode_x, mode_y = config.get("modes", [1, 1])
        return amplitude * torch.sin(float(mode_x) * torch.pi * x / a) * torch.sin(
            float(mode_y) * torch.pi * y / b
        )
    if load_type == "gaussian":
        amplitude = float(config.get("amplitude", 1.0))
        center_x, center_y = config.get("center", [a / 2.0, b / 2.0])
        sigma_x, sigma_y = config.get("sigma", [a / 10.0, b / 10.0])
        exponent = -0.5 * (
            ((x - float(center_x)) / float(sigma_x)) ** 2
            + ((y - float(center_y)) / float(sigma_y)) ** 2
        )
        return amplitude * torch.exp(exponent)
    raise ValueError(f"Unsupported plate load: {load_type}")


def plate_load_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    config: dict[str, Any],
) -> np.ndarray:
    """NumPy counterpart of :func:`plate_load_torch` for reference solvers."""

    load_type = config.get("type", "uniform")
    if load_type == "uniform":
        return np.full_like(x, float(config.get("value", 0.0)), dtype=float)
    if load_type == "sine":
        amplitude = float(config.get("amplitude", 1.0))
        mode_x, mode_y = config.get("modes", [1, 1])
        return amplitude * np.sin(float(mode_x) * np.pi * x / a) * np.sin(
            float(mode_y) * np.pi * y / b
        )
    if load_type == "gaussian":
        amplitude = float(config.get("amplitude", 1.0))
        center_x, center_y = config.get("center", [a / 2.0, b / 2.0])
        sigma_x, sigma_y = config.get("sigma", [a / 10.0, b / 10.0])
        if float(sigma_x) <= 0.0 or float(sigma_y) <= 0.0:
            raise ValueError("Gaussian plate-load sigma values must be positive.")
        exponent = -0.5 * (
            ((x - float(center_x)) / float(sigma_x)) ** 2
            + ((y - float(center_y)) / float(sigma_y)) ** 2
        )
        return amplitude * np.exp(exponent)
    raise ValueError(f"Unsupported plate load: {load_type}")


def distributed_load_scale(config: dict[str, Any]) -> float:
    values: list[float] = []
    for key in ("value", "amplitude"):
        if key in config:
            values.append(abs(float(config[key])))
    values.extend(abs(float(value)) for value in config.get("coefficients", []))
    values.extend(abs(float(segment["value"])) for segment in config.get("segments", []))
    return max(values, default=1.0)

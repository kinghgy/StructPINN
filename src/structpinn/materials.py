from __future__ import annotations

from typing import Any

import numpy as np
import torch


def beam_ei_torch(s: torch.Tensor, material: dict[str, Any]) -> torch.Tensor:
    if "EI" in material:
        return torch.full_like(s, float(material["EI"]))
    profile = material["profile"]
    profile_type = profile.get("type", "linear")
    if profile_type == "linear":
        start = float(profile["start"])
        end = float(profile["end"])
        return start + (end - start) * s
    if profile_type == "piecewise_constant":
        result = torch.zeros_like(s)
        for segment in profile.get("segments", []):
            start = float(segment["start"])
            end = float(segment["end"])
            value = float(segment["value"])
            mask = (s >= start) & (s <= end)
            result = torch.where(mask, torch.full_like(result, value), result)
        if torch.any(result <= 0):
            raise ValueError("EI profile leaves uncovered or non-positive regions.")
        return result
    raise ValueError(f"Unsupported EI profile: {profile_type}")


def beam_ei_numpy(s: np.ndarray, material: dict[str, Any]) -> np.ndarray:
    if "EI" in material:
        return np.full_like(s, float(material["EI"]), dtype=float)
    profile = material["profile"]
    profile_type = profile.get("type", "linear")
    if profile_type == "linear":
        return float(profile["start"]) + (
            float(profile["end"]) - float(profile["start"])
        ) * s
    if profile_type == "piecewise_constant":
        result = np.zeros_like(s, dtype=float)
        for segment in profile.get("segments", []):
            mask = (s >= float(segment["start"])) & (s <= float(segment["end"]))
            result[mask] = float(segment["value"])
        if np.any(result <= 0):
            raise ValueError("EI profile leaves uncovered or non-positive regions.")
        return result
    raise ValueError(f"Unsupported EI profile: {profile_type}")


def representative_beam_ei(material: dict[str, Any]) -> float:
    samples = np.linspace(0.0, 1.0, 101)
    return float(np.median(beam_ei_numpy(samples, material)))

from __future__ import annotations

import math
from typing import Any

import torch

from .problems.base import PINNProblem


def sample_unit_domain(
    dimension: int,
    count: int,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if method == "uniform":
        return torch.rand((count, dimension), device=device, dtype=dtype)
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True)
        return engine.draw(count).to(device=device, dtype=dtype)
    if method == "grid":
        if dimension == 1:
            return torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)[:, None]
        side = math.ceil(math.sqrt(count))
        line = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
        x, y = torch.meshgrid(line, line, indexing="xy")
        return torch.stack([x.flatten(), y.flatten()], dim=1)[:count]
    raise ValueError(f"Unknown sampler: {method}")


def _exclude_beam_singularities(
    points: torch.Tensor,
    problem: PINNProblem,
    exclusion_radius: float,
) -> torch.Tensor:
    point_loads = getattr(problem, "point_loads", [])
    length = getattr(problem, "length", 1.0)
    if not point_loads:
        return points
    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for point_load in point_loads:
        location = float(point_load["position"]) / float(length)
        mask = mask & (torch.abs(points[:, 0] - location) > exclusion_radius)
    return points[mask]


def sample_interior(
    problem: PINNProblem,
    count: int,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    exclusion_radius: float = 1.0e-4,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        candidates = sample_unit_domain(
            problem.dimension,
            max(remaining * 2, 32),
            method,
            device,
            dtype,
        )
        candidates = _exclude_beam_singularities(candidates, problem, exclusion_radius)
        take = min(remaining, candidates.shape[0])
        chunks.append(candidates[:take])
        remaining -= take
    return torch.cat(chunks, dim=0).detach().requires_grad_(True)


def residual_adaptive_sample(
    problem: PINNProblem,
    count: int,
    settings: dict[str, Any],
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    exclusion_radius: float = 1.0e-4,
) -> torch.Tensor:
    candidate_count = max(int(settings.get("candidate_pool", 4096)), count)
    candidates = sample_interior(
        problem,
        candidate_count,
        method,
        device,
        dtype,
        exclusion_radius,
    )
    chunk_size = int(settings.get("chunk_size", 1024))
    scores: list[torch.Tensor] = []
    for start in range(0, candidate_count, chunk_size):
        chunk = candidates[start : start + chunk_size].detach().requires_grad_(True)
        score = problem.residual_score(chunk)
        scores.append(score.detach())
    combined = torch.cat(scores)
    power = float(settings.get("power", 1.0))
    offset = float(settings.get("offset", 0.1))
    normalized = combined.clamp_min(0.0).pow(power)
    probabilities = normalized / (normalized.mean() + 1.0e-12) + offset
    probabilities = probabilities / probabilities.sum()
    indices = torch.multinomial(probabilities, count, replacement=False)
    return candidates.detach()[indices].clone().requires_grad_(True)

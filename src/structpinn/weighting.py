from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class GradientLossBalancer:
    """EMA-smoothed inverse-gradient weighting for multi-objective PINN losses."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.enabled = bool(settings.get("enabled", True))
        self.ema = float(settings.get("ema", 0.9))
        self.min_weight = float(settings.get("min_weight", 0.05))
        self.max_weight = float(settings.get("max_weight", 20.0))
        self.weights: dict[str, float] = {}

    @staticmethod
    def _gradient_norm(loss: torch.Tensor, parameters: list[nn.Parameter]) -> float:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = 0.0
        for gradient in gradients:
            if gradient is not None:
                squared += float(torch.sum(gradient.detach() ** 2).cpu())
        return math.sqrt(squared)

    def update(self, losses: dict[str, torch.Tensor], model: nn.Module) -> None:
        if not self.enabled:
            return
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        norms: dict[str, float] = {}
        for name, loss in losses.items():
            if not loss.requires_grad or float(loss.detach().cpu()) < 1.0e-18:
                continue
            norms[name] = self._gradient_norm(loss, parameters)
        positive = [norm for norm in norms.values() if norm > 1.0e-16]
        if not positive:
            return
        target = sum(positive) / len(positive)
        proposed = {
            name: target / (norm + 1.0e-12)
            for name, norm in norms.items()
            if norm > 1.0e-16
        }
        normalization = len(proposed) / max(sum(proposed.values()), 1.0e-12)
        for name, value in proposed.items():
            value = min(self.max_weight, max(self.min_weight, value * normalization))
            old = self.weights.get(name, 1.0)
            self.weights[name] = self.ema * old + (1.0 - self.ema) * value

    def combine(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights.get(name, 1.0) * loss for name, loss in losses.items())

    def snapshot(self) -> dict[str, float]:
        return dict(self.weights)

    def state_dict(self) -> dict[str, Any]:
        return {"weights": self.snapshot()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.weights = {str(name): float(value) for name, value in state["weights"].items()}

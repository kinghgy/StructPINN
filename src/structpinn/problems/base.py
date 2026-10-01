from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class PINNProblem(nn.Module, ABC):
    dimension: int

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def residuals(self, interior: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def boundary_residuals(
        self, boundary_points: Any | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def make_boundary_points(self, count: int) -> Any | None:
        raise NotImplementedError

    def loss_terms(
        self,
        interior: torch.Tensor,
        boundary_points: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        residuals = self.residuals(interior)
        boundaries = self.boundary_residuals(boundary_points)
        terms = {f"physics/{name}": torch.mean(value**2) for name, value in residuals.items()}
        terms.update(
            {f"boundary/{name}": torch.mean(value**2) for name, value in boundaries.items()}
        )
        return terms

    def residual_score(self, interior: torch.Tensor) -> torch.Tensor:
        residuals = self.residuals(interior)
        return torch.sum(torch.cat([value**2 for value in residuals.values()], dim=1), dim=1)

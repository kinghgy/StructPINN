from __future__ import annotations

from typing import Any

import torch

from ..loads import beam_load_torch, distributed_load_scale
from ..materials import beam_ei_torch, representative_beam_ei
from ..networks import FieldNetwork
from ..utils import derivative
from .base import PINNProblem


class BeamProblem(PINNProblem):
    dimension = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        problem = config["problem"]
        model = config["model"]
        self.length = float(problem["geometry"]["length"])
        self.material = problem["material"]
        self.load = problem.get("load", {}).get("distributed", {"type": "uniform", "value": 0.0})
        self.point_loads = problem.get("load", {}).get("point_loads", [])
        self.boundary = problem["boundary"]
        self.enforcement = problem.get("boundary_enforcement", "soft")

        for point_load in self.point_loads:
            position = float(point_load["position"])
            if not 0.0 < position < self.length:
                raise ValueError(
                    "point_loads are for interior forces only. Use a free-end shear boundary "
                    "condition for an end force."
                )

        input_dim = 1 + len(self.point_loads)
        self.network = FieldNetwork(
            input_dim=input_dim,
            field_dims={"u": 1, "theta": 1, "kappa": 1, "shear": 1},
            hidden_layers=list(model["hidden_layers"]),
            activation=model["activation"],
            architecture=model["architecture"],
        )
        self.scales = self._build_scales()

    def _build_scales(self) -> dict[str, float]:
        q_scale = distributed_load_scale(self.load)
        point_scale = sum(abs(float(load["value"])) for load in self.point_loads)
        boundary_shear = max(
            abs(float(condition.get("shear", 0.0)))
            for condition in self.boundary.values()
        )
        boundary_moment = max(
            abs(float(condition.get("moment", 0.0)))
            for condition in self.boundary.values()
        )
        shear = max(q_scale * self.length + point_scale, boundary_shear, 1.0e-8)
        moment = max(shear * self.length, boundary_moment, 1.0e-8)
        shear = max(shear, moment / self.length)
        curvature = max(moment / representative_beam_ei(self.material), 1.0e-8)
        rotation = max(curvature * self.length, 1.0e-8)
        displacement = max(rotation * self.length, 1.0e-8)
        return {
            "q": max(q_scale, shear / self.length, 1.0e-8),
            "shear": shear,
            "moment": moment,
            "kappa": curvature,
            "theta": rotation,
            "u": displacement,
        }

    def _features(self, s: torch.Tensor) -> torch.Tensor:
        features = [2.0 * s - 1.0]
        for point_load in self.point_loads:
            location = float(point_load["position"]) / self.length
            features.append(torch.abs(s - location))
        return torch.cat(features, dim=1)

    @staticmethod
    def _boundary_value(boundary: dict[str, Any], field: str, default: float = 0.0) -> float:
        return float(boundary.get(field, default))

    def _jump(self, s: torch.Tensor) -> torch.Tensor:
        jump = torch.zeros_like(s)
        for point_load in self.point_loads:
            location = float(point_load["position"]) / self.length
            value = float(point_load["value"])
            jump = jump + torch.where(s > location, torch.full_like(s, -value), torch.zeros_like(s))
        return jump

    def fields(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.network(self._features(s))
        u_raw = raw["u"]
        theta_raw = raw["theta"]
        kappa_raw = raw["kappa"]
        shear_raw = raw["shear"]
        ei = beam_ei_torch(s, self.material)
        jump = self._jump(s)

        left = self.boundary["left"]
        right = self.boundary["right"]
        pair = (left["kind"], right["kind"])

        if self.enforcement == "hard" and pair == ("pinned", "pinned"):
            u_left = self._boundary_value(left, "displacement")
            u_right = self._boundary_value(right, "displacement")
            m_left = self._boundary_value(left, "moment")
            m_right = self._boundary_value(right, "moment")
            u = (1.0 - s) * u_left + s * u_right + s * (1.0 - s) * u_raw
            moment = (1.0 - s) * m_left + s * m_right + s * (1.0 - s) * kappa_raw
            kappa = moment / ei
            theta = theta_raw
            shear = shear_raw + jump
        elif self.enforcement == "hard" and pair == ("fixed", "free"):
            u_left = self._boundary_value(left, "displacement")
            theta_left = self._boundary_value(left, "rotation")
            m_right = self._boundary_value(right, "moment")
            q_right = self._boundary_value(right, "shear")
            jump_at_right = -sum(float(load["value"]) for load in self.point_loads)
            u = u_left + self.length * theta_left * s + s**2 * u_raw
            theta = theta_left + s * theta_raw
            kappa = (m_right + (1.0 - s) * kappa_raw) / ei
            shear = q_right - jump_at_right + (1.0 - s) * shear_raw + jump
        elif self.enforcement == "hard" and pair == ("free", "fixed"):
            u_right = self._boundary_value(right, "displacement")
            theta_right = self._boundary_value(right, "rotation")
            m_left = self._boundary_value(left, "moment")
            q_left = self._boundary_value(left, "shear")
            u = u_right + self.length * theta_right * (s - 1.0) + (1.0 - s) ** 2 * u_raw
            theta = theta_right + (1.0 - s) * theta_raw
            kappa = (m_left + s * kappa_raw) / ei
            shear = q_left + s * shear_raw + jump
        elif self.enforcement == "hard" and pair == ("fixed", "fixed"):
            u0 = self._boundary_value(left, "displacement")
            u1 = self._boundary_value(right, "displacement")
            t0 = self._boundary_value(left, "rotation")
            t1 = self._boundary_value(right, "rotation")
            h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
            h10 = s**3 - 2.0 * s**2 + s
            h01 = -2.0 * s**3 + 3.0 * s**2
            h11 = s**3 - s**2
            u = h00 * u0 + h10 * self.length * t0 + h01 * u1 + h11 * self.length * t1
            u = u + s**2 * (1.0 - s) ** 2 * u_raw
            theta = (1.0 - s) * t0 + s * t1 + s * (1.0 - s) * theta_raw
            kappa = kappa_raw
            shear = shear_raw + jump
        else:
            u = u_raw
            theta = theta_raw
            kappa = kappa_raw
            shear = shear_raw + jump

        return {
            "u": u,
            "theta": theta,
            "kappa": kappa,
            "moment": ei * kappa,
            "shear": shear,
        }

    def residuals(self, interior: torch.Tensor) -> dict[str, torch.Tensor]:
        if not interior.requires_grad:
            interior = interior.requires_grad_(True)
        fields = self.fields(interior)
        x = interior * self.length
        distributed = beam_load_torch(x, self.length, self.load)
        u_x = derivative(fields["u"], interior) / self.length
        theta_x = derivative(fields["theta"], interior) / self.length
        moment_x = derivative(fields["moment"], interior) / self.length
        shear_x = derivative(fields["shear"], interior) / self.length
        return {
            "force_equilibrium": (shear_x + distributed) / self.scales["q"],
            "moment_equilibrium": (moment_x - fields["shear"]) / self.scales["shear"],
            "curvature_compatibility": (theta_x + fields["kappa"]) / self.scales["kappa"],
            "rotation_compatibility": (u_x - fields["theta"]) / self.scales["theta"],
        }

    def _endpoint_residuals(
        self,
        side: str,
        s: torch.Tensor,
        fields: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        boundary = self.boundary[side]
        kind = boundary["kind"]
        residuals: dict[str, torch.Tensor] = {}
        if kind == "fixed":
            residuals[f"{side}_displacement"] = (
                fields["u"] - self._boundary_value(boundary, "displacement")
            ) / self.scales["u"]
            residuals[f"{side}_rotation"] = (
                fields["theta"] - self._boundary_value(boundary, "rotation")
            ) / self.scales["theta"]
        elif kind == "pinned":
            residuals[f"{side}_displacement"] = (
                fields["u"] - self._boundary_value(boundary, "displacement")
            ) / self.scales["u"]
            residuals[f"{side}_moment"] = (
                fields["moment"] - self._boundary_value(boundary, "moment")
            ) / self.scales["moment"]
        elif kind == "free":
            residuals[f"{side}_moment"] = (
                fields["moment"] - self._boundary_value(boundary, "moment")
            ) / self.scales["moment"]
            residuals[f"{side}_shear"] = (
                fields["shear"] - self._boundary_value(boundary, "shear")
            ) / self.scales["shear"]
        else:
            aliases = {
                "displacement": ("u", "u"),
                "rotation": ("theta", "theta"),
                "moment": ("moment", "moment"),
                "shear": ("shear", "shear"),
            }
            for name, (field_name, scale_name) in aliases.items():
                if name in boundary:
                    residuals[f"{side}_{name}"] = (
                        fields[field_name] - float(boundary[name])
                    ) / self.scales[scale_name]
        return residuals

    def boundary_residuals(self, boundary_points: Any | None = None) -> dict[str, torch.Tensor]:
        parameter = next(self.parameters())
        left_s = torch.zeros((1, 1), device=parameter.device, dtype=parameter.dtype)
        right_s = torch.ones((1, 1), device=parameter.device, dtype=parameter.dtype)
        residuals = self._endpoint_residuals("left", left_s, self.fields(left_s))
        residuals.update(self._endpoint_residuals("right", right_s, self.fields(right_s)))
        return residuals

    def make_boundary_points(self, count: int) -> None:
        return None

    def predict(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = self.fields(s)
        return {"x": s * self.length, **fields}

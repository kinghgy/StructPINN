from __future__ import annotations

from typing import Any

import torch

from ..loads import distributed_load_scale, plate_load_torch
from ..networks import FieldNetwork
from ..utils import derivative
from .base import PINNProblem


class PlateProblem(PINNProblem):
    dimension = 2

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        problem = config["problem"]
        model = config["model"]
        self.a = float(problem["geometry"]["a"])
        self.b = float(problem["geometry"]["b"])
        self.D = float(problem["material"]["D"])
        self.nu = float(problem["material"]["nu"])
        self.load = problem.get("load", {"type": "uniform", "value": 0.0})
        self.boundary = problem["boundary"]
        self.enforcement = problem.get("boundary_enforcement", "soft")
        self.constitutive_enforcement = problem.get("constitutive_enforcement", "soft")
        self.boundary_compatibility = bool(problem.get("boundary_compatibility", False))
        self.boundary_compatibility_grouping = problem.get(
            "boundary_compatibility_grouping", "orientation"
        )
        self.residual_scaling = problem.get("residual_scaling", "isotropic")
        self.training_formulation = problem.get("training_formulation", "mixed")
        self.field_source = config.get("output", {}).get("plate_field_source", "mixed")
        field_dims = {"w": 1, "moment": 3}
        if self.constitutive_enforcement == "soft":
            field_dims["curvature"] = 3
        self.network = FieldNetwork(
            input_dim=2,
            field_dims=field_dims,
            hidden_layers=list(model["hidden_layers"]),
            activation=model["activation"],
            architecture=model["architecture"],
        )
        self.scales = self._build_scales()

    def _build_scales(self) -> dict[str, float]:
        q_scale = max(distributed_load_scale(self.load), 1.0e-8)
        characteristic_length = max(self.a, self.b)
        moment = max(q_scale * characteristic_length**2, 1.0e-8)
        curvature = max(moment / self.D, 1.0e-8)
        displacement = max(curvature * characteristic_length**2, 1.0e-8)
        scales = {
            "q": q_scale,
            "moment": moment,
            "curvature": curvature,
            "w": displacement,
            "slope": max(displacement / characteristic_length, 1.0e-8),
            "shear": max(moment / characteristic_length, 1.0e-8),
        }
        if self.residual_scaling == "anisotropic":
            # With sqrt(a*b) as the displacement length scale, the typical
            # curvatures scale as w/a^2, w/b^2 and 2w/(a*b). Their ratios are
            # what matters; retain the previous geometric-mean magnitude.
            scales["curvature_x"] = curvature * self.b / self.a
            scales["curvature_y"] = curvature * self.a / self.b
            scales["curvature_xy"] = 2.0 * curvature
        else:
            scales["curvature_x"] = curvature
            scales["curvature_y"] = curvature
            scales["curvature_xy"] = curvature
        return scales

    def _all_edges_are(self, kind: str) -> bool:
        return all(self.boundary[edge]["kind"] == kind for edge in self.boundary)

    def fields(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        sx = coordinates[:, 0:1]
        sy = coordinates[:, 1:2]
        raw = self.network(torch.cat([2.0 * sx - 1.0, 2.0 * sy - 1.0], dim=1))
        w_raw = raw["w"]
        moment = raw["moment"]

        if self.enforcement == "hard" and self._all_edges_are("simply_supported"):
            envelope = sx * (1.0 - sx) * sy * (1.0 - sy)
            w = envelope * w_raw
        elif self.enforcement == "hard" and self._all_edges_are("clamped"):
            envelope = (sx * (1.0 - sx) * sy * (1.0 - sy)) ** 2
            w = envelope * w_raw
        else:
            w = w_raw

        if (
            self.constitutive_enforcement == "hard"
            and self.enforcement == "hard"
            and self._all_edges_are("simply_supported")
        ):
            mx = sx * (1.0 - sx) * moment[:, 0:1]
            my = sy * (1.0 - sy) * moment[:, 1:2]
            mxy = moment[:, 2:3]
        else:
            mx = moment[:, 0:1]
            my = moment[:, 1:2]
            mxy = moment[:, 2:3]

        if self.constitutive_enforcement == "hard":
            denominator = self.D * (1.0 - self.nu**2)
            kx = (mx - self.nu * my) / denominator
            ky = (my - self.nu * mx) / denominator
            kxy = 2.0 * mxy / (self.D * (1.0 - self.nu))
        else:
            curvature = raw["curvature"]
            kx = curvature[:, 0:1]
            ky = curvature[:, 1:2]
            kxy = curvature[:, 2:3]

        return {
            "w": w,
            "kx": kx,
            "ky": ky,
            "kxy": kxy,
            "Mx": mx,
            "My": my,
            "Mxy": mxy,
        }

    def _field_derivatives(
        self, coordinates: torch.Tensor, fields: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        grad_w = derivative(fields["w"], coordinates)
        wx = grad_w[:, 0:1] / self.a
        wy = grad_w[:, 1:2] / self.b
        grad_wx = derivative(wx, coordinates)
        grad_wy = derivative(wy, coordinates)
        wxx = grad_wx[:, 0:1] / self.a
        wxy = grad_wx[:, 1:2] / self.b
        wyy = grad_wy[:, 1:2] / self.b
        grad_mx = derivative(fields["Mx"], coordinates)
        grad_my = derivative(fields["My"], coordinates)
        grad_mxy = derivative(fields["Mxy"], coordinates)
        mx_x = grad_mx[:, 0:1] / self.a
        my_y = grad_my[:, 1:2] / self.b
        mxy_x = grad_mxy[:, 0:1] / self.a
        mxy_y = grad_mxy[:, 1:2] / self.b
        grad_mx_x = derivative(mx_x, coordinates)
        grad_my_y = derivative(my_y, coordinates)
        grad_mxy_x = derivative(mxy_x, coordinates)
        return {
            "wx": wx,
            "wy": wy,
            "wxx": wxx,
            "wyy": wyy,
            "wxy": wxy,
            "Mx_x": mx_x,
            "My_y": my_y,
            "Mxy_x": mxy_x,
            "Mxy_y": mxy_y,
            "Mx_xx": grad_mx_x[:, 0:1] / self.a,
            "My_yy": grad_my_y[:, 1:2] / self.b,
            "Mxy_xy": grad_mxy_x[:, 1:2] / self.b,
        }

    def displacement_derived_fields(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project the displacement branch onto geometry and constitutive relations."""
        with torch.enable_grad():
            if not coordinates.requires_grad:
                coordinates = coordinates.detach().requires_grad_(True)
            mixed = self.fields(coordinates)
            grad_w = derivative(mixed["w"], coordinates)
            wx = grad_w[:, 0:1] / self.a
            wy = grad_w[:, 1:2] / self.b
            grad_wx = derivative(wx, coordinates)
            grad_wy = derivative(wy, coordinates)
            kx = -grad_wx[:, 0:1] / self.a
            ky = -grad_wy[:, 1:2] / self.b
            kxy = -2.0 * grad_wx[:, 1:2] / self.b
            mx = self.D * (kx + self.nu * ky)
            my = self.D * (ky + self.nu * kx)
            mxy = 0.5 * self.D * (1.0 - self.nu) * kxy
            return {
                "w": mixed["w"],
                "kx": kx,
                "ky": ky,
                "kxy": kxy,
                "Mx": mx,
                "My": my,
                "Mxy": mxy,
            }

    def projected_residuals(self, interior: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fourth-order equilibrium residual of the displacement-derived projection."""
        if not interior.requires_grad:
            interior = interior.detach().requires_grad_(True)
        fields = self.displacement_derived_fields(interior)
        derivatives = self._field_derivatives(interior, fields)
        x = interior[:, 0:1] * self.a
        y = interior[:, 1:2] * self.b
        load = plate_load_torch(x, y, self.a, self.b, self.load)
        return {
            "equilibrium": (
                derivatives["Mx_xx"]
                + 2.0 * derivatives["Mxy_xy"]
                + derivatives["My_yy"]
                + load
            )
            / self.scales["q"]
        }

    def residuals(self, interior: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training_formulation == "displacement":
            return self.projected_residuals(interior)
        if not interior.requires_grad:
            interior = interior.requires_grad_(True)
        fields = self.fields(interior)
        derivatives = self._field_derivatives(interior, fields)
        x = interior[:, 0:1] * self.a
        y = interior[:, 1:2] * self.b
        load = plate_load_torch(x, y, self.a, self.b, self.load)
        residuals = {
            "geometry_x": (fields["kx"] + derivatives["wxx"])
            / self.scales["curvature_x"],
            "geometry_y": (fields["ky"] + derivatives["wyy"])
            / self.scales["curvature_y"],
            "geometry_xy": (fields["kxy"] + 2.0 * derivatives["wxy"])
            / self.scales["curvature_xy"],
            "equilibrium": (
                derivatives["Mx_xx"]
                + 2.0 * derivatives["Mxy_xy"]
                + derivatives["My_yy"]
                + load
            )
            / self.scales["q"],
        }
        if self.constitutive_enforcement == "soft":
            residuals.update(
                {
                    "constitutive_x": (
                        fields["Mx"] - self.D * (fields["kx"] + self.nu * fields["ky"])
                    )
                    / self.scales["moment"],
                    "constitutive_y": (
                        fields["My"] - self.D * (fields["ky"] + self.nu * fields["kx"])
                    )
                    / self.scales["moment"],
                    "constitutive_xy": (
                        fields["Mxy"]
                        - 0.5 * self.D * (1.0 - self.nu) * fields["kxy"]
                    )
                    / self.scales["moment"],
                }
            )
        return residuals

    def make_boundary_points(self, count: int) -> dict[str, torch.Tensor]:
        parameter = next(self.parameters())
        line = torch.linspace(0.0, 1.0, count, device=parameter.device, dtype=parameter.dtype)
        zeros = torch.zeros_like(line)
        ones = torch.ones_like(line)
        return {
            "left": torch.stack([zeros, line], dim=1),
            "right": torch.stack([ones, line], dim=1),
            "bottom": torch.stack([line, zeros], dim=1),
            "top": torch.stack([line, ones], dim=1),
        }

    def boundary_residuals(
        self, boundary_points: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if boundary_points is None:
            boundary_points = self.make_boundary_points(64)
        residuals: dict[str, torch.Tensor] = {}
        compatibility_residuals: dict[str, torch.Tensor] = {}
        for edge, coordinates in boundary_points.items():
            if not coordinates.requires_grad:
                coordinates = coordinates.detach().clone().requires_grad_(True)
            fields = (
                self.displacement_derived_fields(coordinates)
                if self.training_formulation == "displacement"
                else self.fields(coordinates)
            )
            derivatives = self._field_derivatives(coordinates, fields)
            kind = self.boundary[edge]["kind"]
            normal_x = edge in {"left", "right"}
            if kind == "simply_supported":
                residuals[f"{edge}_w"] = fields["w"] / self.scales["w"]
                normal_moment = fields["Mx"] if normal_x else fields["My"]
                residuals[f"{edge}_moment"] = normal_moment / self.scales["moment"]
                if self.boundary_compatibility:
                    compatibility_residuals[edge] = torch.cat(
                        [
                            (fields["kx"] + derivatives["wxx"])
                            / self.scales["curvature_x"],
                            (fields["ky"] + derivatives["wyy"])
                            / self.scales["curvature_y"],
                            (fields["kxy"] + 2.0 * derivatives["wxy"])
                            / self.scales["curvature_xy"],
                        ],
                        dim=1,
                    )
            elif kind == "clamped":
                residuals[f"{edge}_w"] = fields["w"] / self.scales["w"]
                normal_slope = derivatives["wx"] if normal_x else derivatives["wy"]
                residuals[f"{edge}_slope"] = normal_slope / self.scales["slope"]
            else:
                normal_moment = fields["Mx"] if normal_x else fields["My"]
                transverse_shear = (
                    derivatives["Mx_x"] + 2.0 * derivatives["Mxy_y"]
                    if normal_x
                    else derivatives["My_y"] + 2.0 * derivatives["Mxy_x"]
                )
                residuals[f"{edge}_moment"] = normal_moment / self.scales["moment"]
                residuals[f"{edge}_shear"] = transverse_shear / self.scales["shear"]
        if self.boundary_compatibility_grouping == "edge":
            for edge, value in compatibility_residuals.items():
                residuals[f"{edge}_compatibility"] = value
        elif self.boundary_compatibility_grouping == "orientation":
            for orientation, edges in {
                "x_edges": ("left", "right"),
                "y_edges": ("bottom", "top"),
            }.items():
                values = [compatibility_residuals[edge] for edge in edges if edge in compatibility_residuals]
                if values:
                    residuals[f"compatibility_{orientation}"] = torch.cat(values, dim=0)
        elif compatibility_residuals:
            residuals["compatibility"] = torch.cat(
                list(compatibility_residuals.values()), dim=0
            )
        return residuals

    def predict(self, coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = (
            self.displacement_derived_fields(coordinates)
            if self.field_source == "displacement"
            else self.fields(coordinates)
        )
        return {
            "x": coordinates[:, 0:1] * self.a,
            "y": coordinates[:, 1:2] * self.b,
            **fields,
        }

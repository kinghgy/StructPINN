from pathlib import Path
from types import MethodType

import pytest
import torch

from structpinn.config import load_config
from structpinn.problems import BeamProblem, PlateProblem

ROOT = Path(__file__).resolve().parents[1]


def test_beam_hard_pinned_constraints() -> None:
    config = load_config(ROOT / "configs" / "beam_uniform.yaml")
    problem = BeamProblem(config).to(dtype=torch.float64)
    endpoints = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    fields = problem.fields(endpoints)
    assert torch.max(torch.abs(fields["u"])).item() < 1.0e-12
    assert torch.max(torch.abs(fields["moment"])).item() < 1.0e-12


def test_plate_hard_clamped_constraints() -> None:
    config = load_config(ROOT / "configs" / "plate_clamped.yaml")
    problem = PlateProblem(config).to(dtype=torch.float64)
    coordinates = torch.tensor([[0.0, 0.2], [1.0, 0.7]], dtype=torch.float64, requires_grad=True)
    fields = problem.fields(coordinates)
    gradient = torch.autograd.grad(
        fields["w"],
        coordinates,
        grad_outputs=torch.ones_like(fields["w"]),
        create_graph=True,
    )[0]
    assert torch.max(torch.abs(fields["w"])).item() < 1.0e-12
    assert torch.max(torch.abs(gradient[:, 0])).item() < 1.0e-12


def test_plate_hard_constitutive_and_supported_moment_constraints() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(dtype=torch.float64)
    coordinates = torch.tensor(
        [[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0]], dtype=torch.float64
    )
    fields = problem.fields(coordinates)
    assert torch.max(torch.abs(fields["Mx"][:2])).item() < 1.0e-12
    assert torch.max(torch.abs(fields["My"][2:])).item() < 1.0e-12
    assert torch.max(
        torch.abs(fields["Mx"] - problem.D * (fields["kx"] + problem.nu * fields["ky"]))
    ).item() < 1.0e-12
    assert torch.max(
        torch.abs(fields["My"] - problem.D * (fields["ky"] + problem.nu * fields["kx"]))
    ).item() < 1.0e-12
    assert torch.max(
        torch.abs(
            fields["Mxy"]
            - 0.5 * problem.D * (1.0 - problem.nu) * fields["kxy"]
        )
    ).item() < 1.0e-12


def test_plate_boundary_compatibility_adds_nontrivial_edge_losses() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["problem"]["boundary_compatibility"] = True
    config["problem"]["boundary_compatibility_grouping"] = "orientation"
    config["problem"]["residual_scaling"] = "anisotropic"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(dtype=torch.float64)
    residuals = problem.boundary_residuals(problem.make_boundary_points(9))
    for orientation in ("x_edges", "y_edges"):
        compatibility = residuals[f"compatibility_{orientation}"]
        assert compatibility.shape == (18, 3)
        assert torch.isfinite(compatibility).all()
        assert torch.max(torch.abs(compatibility)).item() > 0.0


def test_plate_anisotropic_curvature_scales_follow_aspect_ratio() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["problem"]["residual_scaling"] = "anisotropic"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config)
    expected_ratio = (problem.b / problem.a) ** 2
    actual_ratio = problem.scales["curvature_x"] / problem.scales["curvature_y"]
    assert actual_ratio == pytest.approx(expected_ratio)


def test_plate_displacement_projection_enforces_geometry_and_constitutive_relations() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(dtype=torch.float64)
    coordinates = torch.rand((5, 2), dtype=torch.float64, requires_grad=True)
    fields = problem.displacement_derived_fields(coordinates)
    derivatives = problem._field_derivatives(coordinates, fields)
    assert torch.max(torch.abs(fields["kx"] + derivatives["wxx"])).item() < 1.0e-12
    assert torch.max(torch.abs(fields["ky"] + derivatives["wyy"])).item() < 1.0e-12
    assert torch.max(torch.abs(fields["kxy"] + 2.0 * derivatives["wxy"])).item() < 1.0e-12
    assert torch.max(
        torch.abs(fields["Mx"] - problem.D * (fields["kx"] + problem.nu * fields["ky"]))
    ).item() < 1.0e-12


def test_plate_free_edge_uses_kirchhoff_effective_shear() -> None:
    config = load_config(ROOT / "configs" / "plate_clamped.yaml")
    for edge in ("left", "right", "bottom", "top"):
        config["problem"]["boundary"][edge] = {"kind": "free"}
    config["problem"]["boundary_enforcement"] = "soft"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(dtype=torch.float64)

    def analytic_fields(
        self: PlateProblem, coordinates: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        sx = coordinates[:, 0:1]
        sy = coordinates[:, 1:2]
        smooth_zero = 0.0 * (sx**4 + sy**4)
        return {
            "w": smooth_zero,
            "kx": smooth_zero,
            "ky": smooth_zero,
            "kxy": smooth_zero,
            "Mx": smooth_zero,
            "My": smooth_zero,
            # Mxy = x*y gives dMxy/dy=x and dMxy/dx=y.
            "Mxy": (sx * self.a) * (sy * self.b),
        }

    problem.fields = MethodType(analytic_fields, problem)  # type: ignore[method-assign]
    residuals = problem.boundary_residuals(problem.make_boundary_points(5))
    assert torch.allclose(
        residuals["right_shear"],
        torch.full_like(residuals["right_shear"], 2.0 * problem.a / problem.scales["shear"]),
    )
    assert torch.allclose(
        residuals["top_shear"],
        torch.full_like(residuals["top_shear"], 2.0 * problem.b / problem.scales["shear"]),
    )


def test_beam_hard_free_end_resultants_and_scales() -> None:
    config = load_config(ROOT / "configs" / "beam_end_load.yaml")
    problem = BeamProblem(config).to(dtype=torch.float64)
    fields = problem.fields(torch.ones((1, 1), dtype=torch.float64))
    assert torch.allclose(fields["moment"], torch.zeros_like(fields["moment"]))
    assert torch.allclose(fields["shear"], torch.full_like(fields["shear"], -1.0))
    assert problem.scales["q"] == 1.0
    assert problem.scales["shear"] == 1.0


def test_beam_point_load_jump_is_embedded_exactly() -> None:
    config = load_config(ROOT / "configs" / "quick_smoke_test.yaml")
    config["problem"]["load"]["distributed"] = {"type": "uniform", "value": 0.0}
    config["problem"]["load"]["point_loads"] = [{"position": 0.37, "value": -0.8}]
    problem = BeamProblem(config).to(dtype=torch.float64)
    for parameter in problem.parameters():
        parameter.data.zero_()
    sides = torch.tensor([[0.36], [0.38]], dtype=torch.float64)
    shear = problem.fields(sides)["shear"].flatten()
    assert torch.isclose(shear[1] - shear[0], torch.tensor(0.8, dtype=torch.float64))

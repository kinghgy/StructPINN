from pathlib import Path

import torch

import structpinn.solver as solver_module
from structpinn.config import load_config, validate_config
from structpinn.problems import BeamProblem, PlateProblem
from structpinn.sampling import residual_adaptive_sample, sample_interior
from structpinn.solver import _beam_outputs
from structpinn.trainer import _reference_anchor_losses, _reference_anchor_targets
from structpinn.weighting import GradientLossBalancer

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def test_beam_adaptive_sampling_and_gradient_weighting() -> None:
    config = load_config(ROOT / "configs" / "quick_smoke_test.yaml")
    config["problem"]["load"]["point_loads"] = [
        {"position": 0.37, "value": -0.8},
    ]
    problem = BeamProblem(config).to(device=DEVICE, dtype=DTYPE)
    interior = sample_interior(problem, 16, "sobol", DEVICE, DTYPE)
    boundary = problem.make_boundary_points(8)
    losses = problem.loss_terms(interior, boundary)

    balancer = GradientLossBalancer(
        {
            "enabled": True,
            "ema": 0.0,
            "min_weight": 0.05,
            "max_weight": 20.0,
        }
    )
    balancer.update(losses, problem)
    total = balancer.combine(losses)
    total.backward()

    assert torch.isfinite(total)
    assert balancer.snapshot()
    assert all(0.05 <= weight <= 20.0 for weight in balancer.snapshot().values())
    assert any(parameter.grad is not None for parameter in problem.parameters())

    resampled = residual_adaptive_sample(
        problem,
        count=8,
        settings={"candidate_pool": 32, "chunk_size": 8, "power": 1.0, "offset": 0.1},
        method="sobol",
        device=DEVICE,
        dtype=DTYPE,
        exclusion_radius=1.0e-3,
    )
    assert resampled.shape == (8, 1)
    assert torch.all(torch.abs(resampled[:, 0] - 0.37) > 1.0e-3)


def test_plate_full_residual_graph_is_differentiable() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["model"]["architecture"] = "shared"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(device=DEVICE, dtype=DTYPE)
    interior = sample_interior(problem, 8, "sobol", DEVICE, DTYPE)
    boundary = problem.make_boundary_points(4)

    losses = problem.loss_terms(interior, boundary)
    total = sum(losses.values())
    total.backward()

    assert torch.isfinite(total)
    assert "physics/equilibrium" in losses
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(parameter.grad is not None for parameter in problem.parameters())


def test_split_plate_can_freeze_displacement_branch() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(device=DEVICE, dtype=DTYPE)
    problem.network.freeze_fields(["w"])
    assert problem.network.networks is not None
    assert all(
        not parameter.requires_grad for parameter in problem.network.networks["w"].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in problem.network.networks["moment"].parameters()
    )


def test_plate_displacement_formulation_has_fourth_order_trainable_loss() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["problem"]["training_formulation"] = "displacement"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(device=DEVICE, dtype=DTYPE)
    problem.network.freeze_fields(["moment"])
    interior = sample_interior(problem, 4, "sobol", DEVICE, DTYPE)
    boundary = problem.make_boundary_points(4)
    losses = problem.loss_terms(interior, boundary)
    assert set(losses) >= {"physics/equilibrium", "boundary/left_moment"}
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert problem.network.networks is not None
    assert any(parameter.grad is not None for parameter in problem.network.networks["w"].parameters())
    assert all(parameter.grad is None for parameter in problem.network.networks["moment"].parameters())


def test_plate_reference_anchor_is_zero_before_parameter_update() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["problem"]["training_formulation"] = "displacement"
    config["model"]["hidden_layers"] = [8, 8]
    problem = PlateProblem(config).to(device=DEVICE, dtype=DTYPE)
    interior = sample_interior(problem, 4, "sobol", DEVICE, DTYPE)
    settings = {"fields": ["w", "kx", "ky", "kxy"], "weight": 100.0}
    targets = _reference_anchor_targets(problem, interior, settings)
    losses = _reference_anchor_losses(problem, interior, targets, settings)
    assert set(losses) == {"anchor/w", "anchor/kx", "anchor/ky", "anchor/kxy"}
    assert all(float(loss.detach()) == 0.0 for loss in losses.values())

    with torch.no_grad():
        first_parameter = next(problem.network.parameters())
        first_parameter.add_(1.0e-3)
    changed = _reference_anchor_losses(problem, interior, targets, settings)
    assert any(float(loss.detach()) > 0.0 for loss in changed.values())


def test_reference_anchor_rejects_adaptive_sampling() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["problem"]["training_formulation"] = "displacement"
    config["training"]["reference_anchor"] = {
        "enabled": True,
        "fields": ["w"],
        "weight": 1.0,
    }
    config["training"]["adaptive_sampling"]["enabled"] = True
    try:
        validate_config(config)
    except ValueError as error:
        assert "adaptive_sampling.enabled=false" in str(error)
    else:
        raise AssertionError("Expected reference-anchor validation to fail.")


def test_residual_diagnostics_use_an_independent_seed(monkeypatch) -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    config["model"]["hidden_layers"] = [8, 8]
    config["training"]["diagnostic_seed"] = 31415
    problem = PlateProblem(config).to(device=DEVICE, dtype=DTYPE)

    def random_points(problem, count, method, device, dtype, *args):
        del count, method, args
        return torch.rand((8, problem.dimension), device=device, dtype=dtype)

    monkeypatch.setattr(solver_module, "sample_interior", random_points)
    first = solver_module._projected_plate_diagnostics(
        problem, config["training"], DEVICE, DTYPE
    )
    torch.manual_seed(999)
    _ = torch.rand(100)
    second = solver_module._projected_plate_diagnostics(
        problem, config["training"], DEVICE, DTYPE
    )
    assert first == second


def test_beam_output_excludes_singular_shear_values_and_reports_jumps(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "beam_point_load.yaml")
    config["model"]["hidden_layers"] = [8, 8]
    config["validation"]["beam_fem_elements"] = 20
    config["output"]["prediction_points"] = 101
    config["output"]["save_plots"] = False
    problem = BeamProblem(config).to(device=DEVICE, dtype=DTYPE)

    metrics = _beam_outputs(problem, config, tmp_path, DEVICE, DTYPE)

    assert metrics["shear"]["excluded_singular_points"] == 2
    assert len(metrics["point_load_jumps"]) == 2
    for result, point_load in zip(
        metrics["point_load_jumps"], problem.point_loads, strict=True
    ):
        assert abs(result["predicted_jump"] + float(point_load["value"])) < 1.0e-5

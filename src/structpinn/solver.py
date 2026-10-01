from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config, save_resolved_config, validate_config
from .problems import BeamProblem, PlateProblem
from .reporting import plot_beam, plot_loss, plot_plate, save_history, save_table
from .sampling import sample_interior
from .trainer import train_problem
from .utils import ensure_directory, resolve_device, resolve_dtype, save_json, set_seed
from .validation import (
    error_metrics,
    plate_navier_reference,
    plate_ritz_reference,
    solve_beam_fem,
)


def _build_problem(config: dict[str, Any]) -> BeamProblem | PlateProblem:
    if config["problem"]["type"] == "beam":
        return BeamProblem(config)
    return PlateProblem(config)


def _transfer_signature(config: dict[str, Any]) -> dict[str, Any]:
    problem = config["problem"]
    signature: dict[str, Any] = {
        "problem_type": problem["type"],
        "model": deepcopy(config["model"]),
    }
    if problem["type"] == "beam":
        signature["point_load_count"] = len(
            problem.get("load", {}).get("point_loads", [])
        )
    else:
        signature["constitutive_enforcement"] = problem.get(
            "constitutive_enforcement", "soft"
        )
    return signature


def validate_transfer_compatibility(
    source_config: dict[str, Any], target_config: dict[str, Any]
) -> None:
    """Reject warm starts whose parameter tensors do not represent the same network."""

    source = _transfer_signature(source_config)
    target = _transfer_signature(target_config)
    if source != target:
        raise ValueError(
            "Initialization checkpoint is network-incompatible with the target config. "
            f"Source signature={source}; target signature={target}. Geometry, material, "
            "load values, and boundary kinds may change, but problem type, model layout, "
            "beam point-load count, and plate constitutive layout must remain compatible."
        )


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = deepcopy(config["training"])
    for key in ("adam_epochs", "device", "log_every", "diagnostic_seed", "checkpointing"):
        training.pop(key, None)
    return {
        "problem": deepcopy(config["problem"]),
        "model": deepcopy(config["model"]),
        "training": training,
    }


def _numpy_prediction(
    problem: BeamProblem | PlateProblem,
    output: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().reshape(-1)
        for name, value in output.items()
    }


def _physics_diagnostics(
    problem: BeamProblem | PlateProblem,
    training: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    set_seed(int(training.get("diagnostic_seed", 1729)))
    points = sample_interior(
        problem,
        min(max(int(training["collocation_points"]), 512), 4096),
        "sobol",
        device,
        dtype,
        float(training.get("point_exclusion_radius", 1.0e-4)),
    )
    residuals = problem.residuals(points)
    diagnostics: dict[str, dict[str, float]] = {}
    for name, value in residuals.items():
        array = value.detach().cpu().numpy()
        diagnostics[name] = {
            "rms": float(np.sqrt(np.mean(array**2))),
            "max_abs": float(np.max(np.abs(array))),
        }
    boundaries = problem.boundary_residuals(
        problem.make_boundary_points(int(training.get("boundary_points", 64)))
    )
    for name, value in boundaries.items():
        array = value.detach().cpu().numpy()
        diagnostics[f"boundary/{name}"] = {
            "rms": float(np.sqrt(np.mean(array**2))),
            "max_abs": float(np.max(np.abs(array))),
        }
    return diagnostics


def _projected_plate_diagnostics(
    problem: PlateProblem,
    training: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    # Use a dedicated, common grid for projected fields. This makes the
    # diagnostic independent of training history and prior evaluation calls.
    set_seed(int(training.get("diagnostic_seed", 1729)) + 1)
    count = min(max(int(training["collocation_points"]), 256), 1024)
    points = sample_interior(problem, count, "sobol", device, dtype)
    residuals = problem.projected_residuals(points)
    diagnostics: dict[str, dict[str, float]] = {}
    for name, value in residuals.items():
        array = value.detach().cpu().numpy()
        diagnostics[name] = {
            "rms": float(np.sqrt(np.mean(array**2))),
            "max_abs": float(np.max(np.abs(array))),
        }
    return diagnostics


def _beam_outputs(
    problem: BeamProblem,
    config: dict[str, Any],
    output_directory: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    count = int(config["output"]["prediction_points"])
    s = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)[:, None]
    with torch.no_grad():
        prediction = _numpy_prediction(problem, problem.predict(s))
    save_table(prediction, output_directory / "predictions.csv")
    reference = None
    metrics: dict[str, Any] = {}
    if config["validation"].get("enabled", True):
        reference = solve_beam_fem(
            config["problem"],
            prediction["x"],
            int(config["validation"].get("beam_fem_elements", 300)),
        )
        save_table(reference, output_directory / "reference_fem.csv")
        for field in ("u", "theta", "moment", "shear"):
            if field == "shear" and problem.point_loads:
                regular = np.ones_like(prediction["x"], dtype=bool)
                tolerance = max(problem.length, 1.0) * 1.0e-10
                for point_load in problem.point_loads:
                    location = float(point_load["position"])
                    regular &= np.abs(prediction["x"] - location) > tolerance
                metrics[field] = error_metrics(
                    prediction[field][regular], reference[field][regular]
                )
                metrics[field]["excluded_singular_points"] = int(np.count_nonzero(~regular))
            else:
                metrics[field] = error_metrics(prediction[field], reference[field])

        if problem.point_loads:
            epsilon_s = max(1.0e-7, float(torch.finfo(dtype).eps) ** 0.5)
            jump_metrics: list[dict[str, float]] = []
            for point_load in problem.point_loads:
                location = float(point_load["position"])
                normalized_location = location / problem.length
                sides = torch.tensor(
                    [
                        [normalized_location - epsilon_s],
                        [normalized_location + epsilon_s],
                    ],
                    device=device,
                    dtype=dtype,
                )
                with torch.no_grad():
                    shear = problem.predict(sides)["shear"].detach().cpu().numpy().reshape(-1)
                predicted_jump = float(shear[1] - shear[0])
                target_jump = -float(point_load["value"])
                jump_metrics.append(
                    {
                        "position": location,
                        "shear_left": float(shear[0]),
                        "shear_right": float(shear[1]),
                        "predicted_jump": predicted_jump,
                        "target_jump": target_jump,
                        "absolute_error": abs(predicted_jump - target_jump),
                    }
                )
            metrics["point_load_jumps"] = jump_metrics
    if config["output"].get("save_plots", True):
        plot_beam(prediction, reference, output_directory / "fields.png")
    return metrics


def _plate_outputs(
    problem: PlateProblem,
    config: dict[str, Any],
    output_directory: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid_size = int(config["output"]["prediction_points"])
    line = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    sx, sy = torch.meshgrid(line, line, indexing="xy")
    coordinates = torch.stack([sx.flatten(), sy.flatten()], dim=1)
    with torch.no_grad():
        prediction = _numpy_prediction(problem, problem.predict(coordinates))
    save_table(prediction, output_directory / "predictions.csv")
    metrics: dict[str, Any] = {}
    validation_info: dict[str, Any] = {"method": "none"}
    if config["validation"].get("enabled", True):
        settings = config["validation"]
        requested_method = settings.get("plate_reference", "auto")
        reference = None
        method = requested_method
        if requested_method in {"auto", "navier"}:
            terms = int(settings.get("plate_navier_terms", 31))
            reference = plate_navier_reference(
                config["problem"], prediction["x"], prediction["y"], terms
            )
            if reference is not None:
                method = "navier"
        if reference is None and requested_method == "navier":
            raise ValueError(
                "The Navier reference supports only a uniformly loaded plate with "
                "four simply supported edges; use plate_reference: ritz or auto."
            )
        if reference is None and requested_method in {"auto", "ritz"}:
            method = "ritz"
            reference = plate_ritz_reference(
                config["problem"],
                prediction["x"],
                prediction["y"],
                int(settings.get("plate_ritz_order", 10)),
                int(settings.get("plate_ritz_quadrature", 32)),
            )
        if reference is not None:
            save_table(reference, output_directory / f"reference_{method}.csv")
            fields = ("w", "kx", "ky", "kxy", "Mx", "My", "Mxy")
            for field in fields:
                metrics[field] = error_metrics(prediction[field], reference[field])
            validation_info = {"method": method}
            if settings.get("plate_convergence_check", True):
                delta = max(int(settings.get("plate_convergence_delta", 2)), 1)
                if method == "navier":
                    high = int(settings.get("plate_navier_terms", 31))
                    low = max(3, high - 2 * delta)
                    comparison = plate_navier_reference(
                        config["problem"], prediction["x"], prediction["y"], low
                    )
                    validation_info["high_resolution"] = high
                    validation_info["low_resolution"] = low
                else:
                    high = int(settings.get("plate_ritz_order", 10))
                    low = max(3, high - delta)
                    comparison = plate_ritz_reference(
                        config["problem"],
                        prediction["x"],
                        prediction["y"],
                        low,
                        int(settings.get("plate_ritz_quadrature", 32)),
                    )
                    validation_info["high_resolution"] = high
                    validation_info["low_resolution"] = low
                validation_info["self_convergence"] = {
                    field: error_metrics(comparison[field], reference[field])
                    for field in fields
                }
    if config["output"].get("save_plots", True):
        plot_plate(prediction, grid_size, output_directory / "fields.png")
    return metrics, validation_info


def solve(
    config_or_path: dict[str, Any] | str | Path,
    output_directory: str | Path | None = None,
    initialization_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    validate_config(config)
    if output_directory is not None:
        config["output"]["directory"] = str(output_directory)
    run_directory = ensure_directory(config["output"]["directory"])
    save_resolved_config(config, run_directory / "config.resolved.yaml")

    training = config["training"]
    set_seed(int(training["seed"]))
    dtype = resolve_dtype(training["dtype"])
    device = resolve_device(training["device"])
    problem = _build_problem(config).to(device=device, dtype=dtype)
    checkpoint_source = None
    resume_source = None
    resume_state = None
    if initialization_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("Use either initialization_checkpoint or resume_checkpoint, not both.")
    if initialization_checkpoint is not None:
        checkpoint_path = Path(initialization_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "config" not in checkpoint:
            raise ValueError("Initialization checkpoint does not contain its resolved config.")
        validate_transfer_compatibility(checkpoint["config"], config)
        problem.load_state_dict(checkpoint["state_dict"])
        checkpoint_source = str(checkpoint_path.resolve())
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_state = torch.load(resume_path, map_location=device, weights_only=True)
        if resume_state.get("kind") != "structpinn_training_state":
            raise ValueError("The resume file is not a StructPINN training-state checkpoint.")
        expected_signature = resume_state.get("metadata", {}).get("resume_signature")
        if expected_signature is None:
            raise ValueError("Training-state checkpoint has no compatibility signature.")
        if expected_signature != _resume_signature(config):
            raise ValueError(
                "Resume config is incompatible with the saved training state. Only "
                "adam_epochs, device, logging, diagnostics, checkpoint cadence, validation, "
                "and output settings may change when resuming."
            )
        if int(training["adam_epochs"]) < int(resume_state["adam_epoch"]):
            raise ValueError("training.adam_epochs is smaller than the saved Adam epoch.")
        resume_source = str(resume_path.resolve())
    if (
        training.get("reference_anchor", {}).get("enabled", False)
        and checkpoint_source is None
        and resume_source is None
    ):
        raise ValueError("Reference-anchored fine-tuning requires an initialization checkpoint.")
    freeze_fields = list(training.get("freeze_fields", []))
    if freeze_fields:
        problem.network.freeze_fields(freeze_fields)
    print(f"Training {config['problem']['type']} problem on {device} with {training['dtype']}.")
    checkpoint_metadata = {
        "resume_signature": _resume_signature(config),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }
    result = train_problem(
        problem,
        training,
        device,
        dtype,
        checkpoint_directory=run_directory / "checkpoints",
        checkpoint_metadata=checkpoint_metadata,
        resume_state=resume_state,
    )
    problem.eval()
    save_history(result.history, run_directory / "history.csv")
    if config["output"].get("save_plots", True):
        plot_loss(result.history, run_directory / "loss.png")

    if isinstance(problem, BeamProblem):
        field_metrics = _beam_outputs(problem, config, run_directory, device, dtype)
        validation_reference = None
    else:
        field_metrics, validation_reference = _plate_outputs(
            problem, config, run_directory, device, dtype
        )
    diagnostics = _physics_diagnostics(problem, training, device, dtype)
    summary = {
        "problem_type": config["problem"]["type"],
        "device": str(device),
        "dtype": training["dtype"],
        "elapsed_seconds": result.elapsed_seconds,
        "stopped_early": result.stopped_early,
        "final_loss_weights": result.final_weights,
        "field_errors": field_metrics,
        "normalized_residuals": diagnostics,
        "initialized_from": checkpoint_source,
        "resumed_from": resume_source,
        "resumed_from_epoch": result.resumed_from_epoch,
        "training_checkpoint": (
            str((run_directory / "checkpoints" / "latest.pt").resolve())
            if (run_directory / "checkpoints" / "latest.pt").is_file()
            else None
        ),
        "frozen_fields": freeze_fields,
    }
    if validation_reference is not None:
        summary["validation_reference"] = validation_reference
    if isinstance(problem, PlateProblem) and problem.field_source == "displacement":
        summary["projected_normalized_residuals"] = _projected_plate_diagnostics(
            problem, training, device, dtype
        )
    save_json(summary, run_directory / "metrics.json")
    if config["output"].get("save_model", True):
        torch.save(
            {
                "state_dict": problem.state_dict(),
                "config": {key: value for key, value in config.items() if not key.startswith("_")},
                "summary": summary,
            },
            run_directory / "model.pt",
        )
    print(f"Finished. Results: {run_directory.resolve()}")
    return summary


def predict_checkpoint(
    checkpoint_path: str | Path,
    output_directory: str | Path,
    device_name: str = "auto",
    plate_field_source: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if plate_field_source is not None:
        config.setdefault("output", {})["plate_field_source"] = plate_field_source
    validate_config(config)
    set_seed(int(config["training"]["seed"]))
    config["training"]["device"] = str(device)
    dtype = resolve_dtype(config["training"]["dtype"])
    problem = _build_problem(config).to(device=device, dtype=dtype)
    problem.load_state_dict(checkpoint["state_dict"])
    problem.eval()
    run_directory = ensure_directory(output_directory)
    save_resolved_config(config, run_directory / "config.resolved.yaml")
    if isinstance(problem, BeamProblem):
        metrics = _beam_outputs(problem, config, run_directory, device, dtype)
        validation_reference = None
    else:
        metrics, validation_reference = _plate_outputs(
            problem, config, run_directory, device, dtype
        )
    diagnostics = _physics_diagnostics(problem, config["training"], device, dtype)
    training_summary = checkpoint.get("summary", {})
    evaluation_summary = {
        "problem_type": config["problem"]["type"],
        "device": str(device),
        "dtype": config["training"]["dtype"],
        "elapsed_seconds": training_summary.get("elapsed_seconds"),
        "stopped_early": training_summary.get("stopped_early"),
        "final_loss_weights": training_summary.get("final_loss_weights", {}),
        "field_errors": metrics,
        "normalized_residuals": diagnostics,
    }
    if validation_reference is not None:
        evaluation_summary["validation_reference"] = validation_reference
    if isinstance(problem, PlateProblem) and problem.field_source == "displacement":
        evaluation_summary["projected_normalized_residuals"] = _projected_plate_diagnostics(
            problem, config["training"], device, dtype
        )
    save_json(evaluation_summary, run_directory / "metrics.json")
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "field_errors": metrics,
    }
    save_json(summary, run_directory / "prediction_summary.json")
    print(f"Prediction exported to: {run_directory.resolve()}")
    return summary

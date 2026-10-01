from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "model": {
        "architecture": "shared",
        "hidden_layers": [64, 64, 64, 64],
        "activation": "silu",
    },
    "training": {
        "seed": 42,
        "diagnostic_seed": 1729,
        "dtype": "float64",
        "device": "auto",
        "adam_epochs": 5000,
        "learning_rate": 1.0e-3,
        "lbfgs_max_iter": 1000,
        "collocation_points": 256,
        "boundary_points": 64,
        "sampler": "sobol",
        "log_every": 100,
        "early_stopping": {
            "enabled": True,
            "patience": 1000,
            "min_relative_improvement": 1.0e-6,
        },
        "adaptive_sampling": {
            "enabled": True,
            "resample_every": 500,
            "candidate_pool": 4096,
            "power": 1.0,
            "offset": 0.1,
            "chunk_size": 1024,
        },
        "adaptive_weighting": {
            "enabled": True,
            "update_every": 100,
            "ema": 0.9,
            "min_weight": 0.05,
            "max_weight": 20.0,
        },
        "checkpointing": {
            "enabled": True,
            "every": 500,
            "keep_last": 2,
        },
    },
    "output": {
        "directory": "runs/default",
        "prediction_points": 401,
        "save_model": True,
        "save_plots": True,
    },
    "validation": {
        "enabled": True,
        "beam_fem_elements": 300,
        "plate_navier_terms": 31,
        "plate_reference": "auto",
        "plate_ritz_order": 10,
        "plate_ritz_quadrature": 32,
        "plate_convergence_check": True,
        "plate_convergence_delta": 2,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = _deep_merge(DEFAULTS, raw)
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if "problem" not in config:
        raise ValueError("Missing required 'problem' section.")
    problem = config["problem"]
    problem_type = problem.get("type")
    if problem_type not in {"beam", "plate"}:
        raise ValueError("problem.type must be 'beam' or 'plate'.")

    if problem_type == "beam":
        length = float(problem.get("geometry", {}).get("length", 0.0))
        if length <= 0:
            raise ValueError("Beam length must be positive.")
        material = problem.get("material", {})
        if "EI" not in material and "profile" not in material:
            raise ValueError("Beam material requires EI or an EI profile.")
        boundary = problem.get("boundary", {})
        for side in ("left", "right"):
            if side not in boundary:
                raise ValueError(f"Beam boundary is missing '{side}'.")
            if boundary[side].get("kind") not in {"fixed", "pinned", "free", "custom"}:
                raise ValueError(f"Unsupported beam boundary kind on {side}.")
    else:
        geometry = problem.get("geometry", {})
        if float(geometry.get("a", 0.0)) <= 0 or float(geometry.get("b", 0.0)) <= 0:
            raise ValueError("Plate dimensions a and b must be positive.")
        material = problem.get("material", {})
        if float(material.get("D", 0.0)) <= 0:
            raise ValueError("Plate bending stiffness D must be positive.")
        nu = float(material.get("nu", 0.0))
        if not -1.0 < nu < 0.5:
            raise ValueError("Poisson ratio must satisfy -1 < nu < 0.5.")
        boundary = problem.get("boundary", {})
        for edge in ("left", "right", "bottom", "top"):
            kind = boundary.get(edge, {}).get("kind")
            if kind not in {"simply_supported", "clamped", "free"}:
                raise ValueError(f"Unsupported or missing plate boundary kind on {edge}.")
        if problem.get("constitutive_enforcement", "soft") not in {"soft", "hard"}:
            raise ValueError("plate constitutive_enforcement must be soft or hard.")
        if problem.get("residual_scaling", "isotropic") not in {"isotropic", "anisotropic"}:
            raise ValueError("plate residual_scaling must be isotropic or anisotropic.")
        if problem.get("training_formulation", "mixed") not in {"mixed", "displacement"}:
            raise ValueError("plate training_formulation must be mixed or displacement.")
        if problem.get("boundary_compatibility_grouping", "orientation") not in {
            "all",
            "orientation",
            "edge",
        }:
            raise ValueError(
                "plate boundary_compatibility_grouping must be all, orientation, or edge."
            )

    training = config["training"]
    if training["dtype"] not in {"float32", "float64"}:
        raise ValueError("training.dtype must be float32 or float64.")
    if int(training["collocation_points"]) < 8:
        raise ValueError("At least 8 collocation points are required.")
    if training["sampler"] not in {"uniform", "sobol", "grid"}:
        raise ValueError("training.sampler must be uniform, sobol, or grid.")
    checkpointing = training.get("checkpointing", {})
    if int(checkpointing.get("every", 500)) < 1:
        raise ValueError("training.checkpointing.every must be positive.")
    if int(checkpointing.get("keep_last", 2)) < 0:
        raise ValueError("training.checkpointing.keep_last cannot be negative.")
    anchor = training.get("reference_anchor", {})
    if anchor.get("enabled", False):
        if problem_type != "plate" or problem.get("training_formulation", "mixed") != "displacement":
            raise ValueError(
                "training.reference_anchor requires a plate displacement formulation."
            )
        if training.get("adaptive_sampling", {}).get("enabled", True):
            raise ValueError(
                "training.reference_anchor requires adaptive_sampling.enabled=false "
                "because its teacher targets are cached at fixed points."
            )
        fields = anchor.get("fields", ["w", "kx", "ky", "kxy"])
        if not fields or not set(fields) <= {"w", "kx", "ky", "kxy"}:
            raise ValueError(
                "training.reference_anchor.fields must be a non-empty subset of "
                "w, kx, ky, kxy."
            )
        anchor["fields"] = list(fields)
        if float(anchor.get("weight", 1.0)) <= 0:
            raise ValueError("training.reference_anchor.weight must be positive.")
    if config.get("output", {}).get("plate_field_source", "mixed") not in {
        "mixed",
        "displacement",
    }:
        raise ValueError("output.plate_field_source must be mixed or displacement.")
    validation = config.get("validation", {})
    reference_method = validation.get("plate_reference", "auto")
    if reference_method not in {"auto", "navier", "ritz", "none"}:
        raise ValueError("validation.plate_reference must be auto, navier, ritz, or none.")
    if int(validation.get("plate_ritz_order", 10)) < 3:
        raise ValueError("validation.plate_ritz_order must be at least 3.")
    if int(validation.get("plate_ritz_quadrature", 32)) < int(
        validation.get("plate_ritz_order", 10)
    ) + 2:
        raise ValueError("validation.plate_ritz_quadrature must be at least order + 2.")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, allow_unicode=True, sort_keys=False)

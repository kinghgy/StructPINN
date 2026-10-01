from pathlib import Path

import numpy as np

from structpinn.config import load_config
from structpinn.validation import plate_navier_reference, plate_ritz_reference

ROOT = Path(__file__).resolve().parents[1]


def test_ritz_agrees_with_independent_navier_solution() -> None:
    problem = load_config(ROOT / "configs" / "plate_simply_supported.yaml")["problem"]
    x_line = np.linspace(0.0, float(problem["geometry"]["a"]), 17)
    y_line = np.linspace(0.0, float(problem["geometry"]["b"]), 17)
    x, y = np.meshgrid(x_line, y_line, indexing="xy")
    navier = plate_navier_reference(problem, x.ravel(), y.ravel(), terms=51)
    ritz = plate_ritz_reference(problem, x.ravel(), y.ravel(), order=8, quadrature=24)
    assert navier is not None
    for field, tolerance in {"w": 2.0e-4, "Mx": 3.0e-3, "My": 3.0e-3}.items():
        relative_error = np.linalg.norm(ritz[field] - navier[field]) / np.linalg.norm(
            navier[field]
        )
        assert relative_error < tolerance


def test_ritz_embeds_clamped_displacement_boundary() -> None:
    problem = load_config(ROOT / "configs" / "plate_clamped.yaml")["problem"]
    y = np.linspace(0.0, float(problem["geometry"]["b"]), 21)
    x = np.zeros_like(y)
    reference = plate_ritz_reference(problem, x, y, order=8, quadrature=24)
    assert np.max(np.abs(reference["w"])) < 1.0e-12
    assert all(np.all(np.isfinite(values)) for values in reference.values())


def test_ritz_rejects_unrestrained_free_plate() -> None:
    problem = load_config(ROOT / "configs" / "plate_clamped.yaml")["problem"]
    for edge in problem["boundary"]:
        problem["boundary"][edge]["kind"] = "free"
    try:
        plate_ritz_reference(problem, np.array([0.5]), np.array([0.5]))
    except ValueError as error:
        assert "rigid-body modes" in str(error)
    else:
        raise AssertionError("Expected a fully free static plate to be rejected.")

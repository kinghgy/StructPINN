from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial import Legendre, Polynomial

from ..loads import plate_load_numpy

_EDGE_POWER = {"free": 0, "simply_supported": 1, "clamped": 2}


def _envelope(left_kind: str, right_kind: str) -> Polynomial:
    left_power = _EDGE_POWER[left_kind]
    right_power = _EDGE_POWER[right_kind]
    return Polynomial([0.0, 1.0]) ** left_power * Polynomial([1.0, -1.0]) ** right_power


def _basis_1d(
    coordinate: np.ndarray,
    length: float,
    order: int,
    left_kind: str,
    right_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return basis values and first three physical derivatives."""

    s = np.asarray(coordinate, dtype=float) / length
    z = 2.0 * s - 1.0
    envelope = _envelope(left_kind, right_kind)
    envelope_values = [envelope.deriv(k)(s) for k in range(4)]
    values = [np.empty((order, s.size), dtype=float) for _ in range(4)]
    for index in range(order):
        polynomial = Legendre.basis(index)
        legendre = [
            (2.0**derivative) * polynomial.deriv(derivative)(z)
            for derivative in range(4)
        ]
        values[0][index] = envelope_values[0] * legendre[0]
        values[1][index] = (
            envelope_values[1] * legendre[0] + envelope_values[0] * legendre[1]
        ) / length
        values[2][index] = (
            envelope_values[2] * legendre[0]
            + 2.0 * envelope_values[1] * legendre[1]
            + envelope_values[0] * legendre[2]
        ) / length**2
        values[3][index] = (
            envelope_values[3] * legendre[0]
            + 3.0 * envelope_values[2] * legendre[1]
            + 3.0 * envelope_values[1] * legendre[2]
            + envelope_values[0] * legendre[3]
        ) / length**3
    return values[0], values[1], values[2], values[3]


def _tensor_basis(x_basis: np.ndarray, y_basis: np.ndarray) -> np.ndarray:
    return np.einsum("ip,jp->ijp", x_basis, y_basis).reshape(
        x_basis.shape[0] * y_basis.shape[0], -1
    )


def _quadrature(length: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * length * (nodes + 1.0), 0.5 * length * weights


def _validate_problem(problem: dict[str, Any], order: int, quadrature: int) -> None:
    if order < 3:
        raise ValueError("plate Ritz order must be at least 3.")
    if quadrature < order + 2:
        raise ValueError("plate Ritz quadrature must be at least order + 2.")
    boundary = problem["boundary"]
    kinds = [boundary[edge]["kind"] for edge in ("left", "right", "bottom", "top")]
    if all(kind == "free" for kind in kinds):
        raise ValueError(
            "A fully free static plate has rigid-body modes; add essential restraints "
            "before requesting a Ritz reference solution."
        )


def plate_ritz_reference(
    problem: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    order: int = 10,
    quadrature: int = 32,
) -> dict[str, np.ndarray]:
    """Rayleigh--Ritz reference for rectangular Kirchhoff plates.

    Essential conditions are embedded in tensor-product trial functions;
    moment and effective-shear conditions are imposed naturally by the energy
    principle.  The implementation is independent of PyTorch and the PINN.
    """

    _validate_problem(problem, order, quadrature)
    a = float(problem["geometry"]["a"])
    b = float(problem["geometry"]["b"])
    stiffness = float(problem["material"]["D"])
    poisson = float(problem["material"]["nu"])
    boundary = problem["boundary"]

    xq, x_weights = _quadrature(a, quadrature)
    yq, y_weights = _quadrature(b, quadrature)
    x_grid, y_grid = np.meshgrid(xq, yq, indexing="xy")
    weights = np.outer(y_weights, x_weights).reshape(-1)
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    x_values = _basis_1d(
        x_flat, a, order, boundary["left"]["kind"], boundary["right"]["kind"]
    )
    y_values = _basis_1d(
        y_flat, b, order, boundary["bottom"]["kind"], boundary["top"]["kind"]
    )
    basis = _tensor_basis(x_values[0], y_values[0])
    basis_xx = _tensor_basis(x_values[2], y_values[0])
    basis_yy = _tensor_basis(x_values[0], y_values[2])
    basis_xy = _tensor_basis(x_values[1], y_values[1])
    weighted = weights[None, :]
    matrix = stiffness * (
        (basis_xx * weighted) @ basis_xx.T
        + (basis_yy * weighted) @ basis_yy.T
        + poisson
        * ((basis_xx * weighted) @ basis_yy.T + (basis_yy * weighted) @ basis_xx.T)
        + 2.0 * (1.0 - poisson) * (basis_xy * weighted) @ basis_xy.T
    )
    load = plate_load_numpy(x_flat, y_flat, a, b, problem.get("load", {}))
    vector = basis @ (weights * load)
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number) or condition_number > 1.0e14:
        raise ValueError(
            "Ritz stiffness matrix is singular or ill-conditioned "
            f"(condition number {condition_number:.3e}); check plate restraints "
            "or lower the basis order."
        )
    coefficients = np.linalg.solve(matrix, vector)

    x_eval = np.asarray(x, dtype=float).reshape(-1)
    y_eval = np.asarray(y, dtype=float).reshape(-1)
    if x_eval.shape != y_eval.shape:
        raise ValueError("x and y evaluation coordinates must have matching shapes.")
    result = {
        "x": x_eval,
        "y": y_eval,
        **{
            name: np.empty_like(x_eval)
            for name in ("w", "kx", "ky", "kxy", "Mx", "My", "Mxy", "Qx", "Qy")
        },
    }
    # Chunk field evaluation so a dense output grid does not allocate several
    # order**2-by-point arrays at once for the whole grid.
    for start in range(0, x_eval.size, 8192):
        stop = min(start + 8192, x_eval.size)
        section = slice(start, stop)
        x_values = _basis_1d(
            x_eval[section],
            a,
            order,
            boundary["left"]["kind"],
            boundary["right"]["kind"],
        )
        y_values = _basis_1d(
            y_eval[section],
            b,
            order,
            boundary["bottom"]["kind"],
            boundary["top"]["kind"],
        )
        basis = _tensor_basis(x_values[0], y_values[0])
        basis_xx = _tensor_basis(x_values[2], y_values[0])
        basis_yy = _tensor_basis(x_values[0], y_values[2])
        basis_xy = _tensor_basis(x_values[1], y_values[1])
        w = coefficients @ basis
        w_xx = coefficients @ basis_xx
        w_yy = coefficients @ basis_yy
        w_xy = coefficients @ basis_xy
        result["w"][section] = w
        result["kx"][section] = -w_xx
        result["ky"][section] = -w_yy
        result["kxy"][section] = -2.0 * w_xy
        result["Mx"][section] = -stiffness * (w_xx + poisson * w_yy)
        result["My"][section] = -stiffness * (w_yy + poisson * w_xx)
        result["Mxy"][section] = -stiffness * (1.0 - poisson) * w_xy
        result["Qx"][section] = -stiffness * (
            coefficients
            @ (
                _tensor_basis(x_values[3], y_values[0])
                + _tensor_basis(x_values[1], y_values[2])
            )
        )
        result["Qy"][section] = -stiffness * (
            coefficients
            @ (
                _tensor_basis(x_values[0], y_values[3])
                + _tensor_basis(x_values[2], y_values[1])
            )
        )
    return result

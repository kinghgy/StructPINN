from __future__ import annotations

from typing import Any

import numpy as np

from ..loads import beam_load_numpy
from ..materials import beam_ei_numpy


def _shape(t: np.ndarray, element_length: float) -> np.ndarray:
    return np.vstack(
        [
            1.0 - 3.0 * t**2 + 2.0 * t**3,
            element_length * (t - 2.0 * t**2 + t**3),
            3.0 * t**2 - 2.0 * t**3,
            element_length * (-t**2 + t**3),
        ]
    )


def _shape_first(t: np.ndarray, element_length: float) -> np.ndarray:
    return np.vstack(
        [
            (-6.0 * t + 6.0 * t**2) / element_length,
            1.0 - 4.0 * t + 3.0 * t**2,
            (6.0 * t - 6.0 * t**2) / element_length,
            -2.0 * t + 3.0 * t**2,
        ]
    )


def _shape_second(t: np.ndarray, element_length: float) -> np.ndarray:
    return np.vstack(
        [
            (-6.0 + 12.0 * t) / element_length**2,
            (-4.0 + 6.0 * t) / element_length,
            (6.0 - 12.0 * t) / element_length**2,
            (-2.0 + 6.0 * t) / element_length,
        ]
    )


def _shape_third(t: np.ndarray, element_length: float) -> np.ndarray:
    return np.vstack(
        [
            np.full_like(t, 12.0 / element_length**3),
            np.full_like(t, 6.0 / element_length**2),
            np.full_like(t, -12.0 / element_length**3),
            np.full_like(t, 6.0 / element_length**2),
        ]
    )


def _essential_dofs(problem: dict[str, Any], node_count: int) -> dict[int, float]:
    boundary = problem["boundary"]
    constraints: dict[int, float] = {}
    for side, node in (("left", 0), ("right", node_count - 1)):
        condition = boundary[side]
        kind = condition["kind"]
        if kind in {"fixed", "pinned"}:
            constraints[2 * node] = float(condition.get("displacement", 0.0))
        if kind == "fixed":
            constraints[2 * node + 1] = float(condition.get("rotation", 0.0))
        if kind == "custom":
            if "displacement" in condition:
                constraints[2 * node] = float(condition["displacement"])
            if "rotation" in condition:
                constraints[2 * node + 1] = float(condition["rotation"])
    return constraints


def _apply_boundary_loads(problem: dict[str, Any], force: np.ndarray, node_count: int) -> None:
    """Apply prescribed internal end resultants as equivalent nodal loads.

    With M=-EI*w'' and Q=M', the left boundary contributes (-Q, +M)
    and the right boundary contributes (+Q, -M) to the global force vector.
    """

    for side, node, force_sign, moment_sign in (
        ("left", 0, -1.0, 1.0),
        ("right", node_count - 1, 1.0, -1.0),
    ):
        condition = problem["boundary"][side]
        kind = condition["kind"]
        if kind in {"free", "custom"} and "shear" in condition:
            force[2 * node] += force_sign * float(condition["shear"])
        if kind in {"pinned", "free", "custom"} and "moment" in condition:
            force[2 * node + 1] += moment_sign * float(condition["moment"])


def solve_beam_fem(
    problem: dict[str, Any],
    evaluation_x: np.ndarray,
    elements: int = 300,
) -> dict[str, np.ndarray]:
    """Independent Euler-Bernoulli beam finite-element reference solution."""

    length = float(problem["geometry"]["length"])
    material = problem["material"]
    load = problem.get("load", {})
    distributed = load.get("distributed", {"type": "uniform", "value": 0.0})
    point_loads = load.get("point_loads", [])
    nodes = np.linspace(0.0, length, elements + 1)
    element_length = length / elements
    dof_count = 2 * (elements + 1)
    stiffness = np.zeros((dof_count, dof_count), dtype=float)
    force = np.zeros(dof_count, dtype=float)

    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(4)
    local_t = 0.5 * (gauss_points + 1.0)
    local_weights = 0.5 * element_length * gauss_weights
    shapes = _shape(local_t, element_length)

    template = np.array(
        [
            [12.0, 6.0 * element_length, -12.0, 6.0 * element_length],
            [6.0 * element_length, 4.0 * element_length**2, -6.0 * element_length, 2.0 * element_length**2],
            [-12.0, -6.0 * element_length, 12.0, -6.0 * element_length],
            [6.0 * element_length, 2.0 * element_length**2, -6.0 * element_length, 4.0 * element_length**2],
        ]
    )

    for element in range(elements):
        x0 = nodes[element]
        midpoint_s = np.array([(x0 + 0.5 * element_length) / length])
        ei = float(beam_ei_numpy(midpoint_s, material)[0])
        local_stiffness = ei / element_length**3 * template
        physical_x = x0 + local_t * element_length
        q_values = beam_load_numpy(physical_x, length, distributed)
        local_force = np.sum(shapes * (q_values * local_weights)[None, :], axis=1)
        dofs = np.array([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        stiffness[np.ix_(dofs, dofs)] += local_stiffness
        force[dofs] += local_force

    for point_load in point_loads:
        position = float(point_load["position"])
        value = float(point_load["value"])
        element = min(int(position / element_length), elements - 1)
        t = np.array([(position - nodes[element]) / element_length])
        dofs = np.array([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        force[dofs] += _shape(t, element_length)[:, 0] * value

    _apply_boundary_loads(problem, force, elements + 1)

    constraints = _essential_dofs(problem, elements + 1)
    fixed = np.array(sorted(constraints), dtype=int)
    all_dofs = np.arange(dof_count)
    free = np.setdiff1d(all_dofs, fixed)
    displacement = np.zeros(dof_count, dtype=float)
    if fixed.size:
        displacement[fixed] = np.array([constraints[index] for index in fixed])
    reduced_force = force[free] - stiffness[np.ix_(free, fixed)] @ displacement[fixed]
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], reduced_force)

    evaluation_x = np.asarray(evaluation_x, dtype=float).reshape(-1)
    w = np.zeros_like(evaluation_x)
    theta = np.zeros_like(evaluation_x)
    moment = np.zeros_like(evaluation_x)
    shear = np.zeros_like(evaluation_x)
    for index, x_value in enumerate(evaluation_x):
        element = min(int(x_value / element_length), elements - 1)
        t = np.array([(x_value - nodes[element]) / element_length])
        dofs = np.array([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        local_displacement = displacement[dofs]
        w[index] = _shape(t, element_length)[:, 0] @ local_displacement
        theta[index] = _shape_first(t, element_length)[:, 0] @ local_displacement
        ei = float(beam_ei_numpy(np.array([x_value / length]), material)[0])
        moment[index] = -ei * (_shape_second(t, element_length)[:, 0] @ local_displacement)
        shear[index] = -ei * (_shape_third(t, element_length)[:, 0] @ local_displacement)

    return {
        "x": evaluation_x,
        "u": w,
        "theta": theta,
        "moment": moment,
        "shear": shear,
    }

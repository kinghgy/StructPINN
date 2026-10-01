from __future__ import annotations

from typing import Any

import numpy as np


def plate_navier_reference(
    problem: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    terms: int = 31,
) -> dict[str, np.ndarray] | None:
    """Truncated Navier reference for a simply supported plate under uniform load."""

    if any(
        problem["boundary"][edge]["kind"] != "simply_supported"
        for edge in ("left", "right", "bottom", "top")
    ):
        return None
    load = problem.get("load", {})
    if load.get("type", "uniform") != "uniform":
        return None

    a = float(problem["geometry"]["a"])
    b = float(problem["geometry"]["b"])
    D = float(problem["material"]["D"])
    nu = float(problem["material"]["nu"])
    q = float(load.get("value", 0.0))
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    result = {
        name: np.zeros_like(x)
        for name in ("w", "kx", "ky", "kxy", "Mx", "My", "Mxy", "Qx", "Qy")
    }
    last = terms if terms % 2 == 1 else terms - 1
    for m in range(1, last + 1, 2):
        alpha = m * np.pi / a
        sin_x = np.sin(alpha * x)
        cos_x = np.cos(alpha * x)
        for n in range(1, last + 1, 2):
            beta = n * np.pi / b
            sin_y = np.sin(beta * y)
            cos_y = np.cos(beta * y)
            q_mn = 16.0 * q / (m * n * np.pi**2)
            amplitude = q_mn / (D * (alpha**2 + beta**2) ** 2)
            sin_product = sin_x * sin_y
            cos_product = cos_x * cos_y
            result["w"] += amplitude * sin_product
            result["kx"] += alpha**2 * amplitude * sin_product
            result["ky"] += beta**2 * amplitude * sin_product
            result["kxy"] += -2.0 * alpha * beta * amplitude * cos_product
            result["Mx"] += D * (alpha**2 + nu * beta**2) * amplitude * sin_product
            result["My"] += D * (beta**2 + nu * alpha**2) * amplitude * sin_product
            result["Mxy"] += -D * (1.0 - nu) * alpha * beta * amplitude * cos_product
            result["Qx"] += D * alpha * (alpha**2 + beta**2) * amplitude * cos_x * sin_y
            result["Qy"] += D * beta * (alpha**2 + beta**2) * amplitude * sin_x * cos_y
    result["x"] = x
    result["y"] = y
    return result

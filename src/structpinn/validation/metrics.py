from __future__ import annotations

import numpy as np


def error_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    difference = prediction - reference
    denominator = np.linalg.norm(reference)
    relative_l2 = np.linalg.norm(difference) / denominator if denominator > 1.0e-14 else np.linalg.norm(difference)
    return {
        "relative_l2": float(relative_l2),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(np.abs(difference))),
    }

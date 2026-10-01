from pathlib import Path

import numpy as np

from structpinn.config import load_config
from structpinn.validation import plate_navier_reference

ROOT = Path(__file__).resolve().parents[1]


def test_navier_reference_satisfies_supported_edge() -> None:
    config = load_config(ROOT / "configs" / "plate_simply_supported.yaml")
    y = np.linspace(0.0, config["problem"]["geometry"]["b"], 11)
    x = np.zeros_like(y)
    reference = plate_navier_reference(config["problem"], x, y, terms=21)
    assert reference is not None
    assert np.max(np.abs(reference["w"])) < 1.0e-12
    assert np.max(np.abs(reference["Mx"])) < 1.0e-12

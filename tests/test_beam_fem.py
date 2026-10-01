from pathlib import Path

import numpy as np

from structpinn.config import load_config
from structpinn.validation import solve_beam_fem

ROOT = Path(__file__).resolve().parents[1]


def test_simply_supported_uniform_beam_fem_matches_analytic_midspan() -> None:
    config = load_config(ROOT / "configs" / "beam_uniform.yaml")
    result = solve_beam_fem(config["problem"], np.array([0.0, 0.5, 1.0]), elements=120)
    q = -2.0
    ei = 0.1
    expected_midspan = q / (24.0 * ei) * (0.5**4 - 2.0 * 0.5**3 + 0.5)
    assert abs(result["u"][0]) < 1.0e-10
    assert abs(result["u"][-1]) < 1.0e-10
    assert np.isclose(result["u"][1], expected_midspan, rtol=2.0e-3)


def test_cantilever_end_force_matches_analytic_solution() -> None:
    length = 2.0
    ei = 3.0
    end_force = -4.0
    problem = {
        "geometry": {"length": length},
        "material": {"EI": ei},
        "load": {"distributed": {"type": "uniform", "value": 0.0}, "point_loads": []},
        "boundary": {
            "left": {"kind": "fixed", "displacement": 0.0, "rotation": 0.0},
            "right": {"kind": "free", "moment": 0.0, "shear": end_force},
        },
    }
    result = solve_beam_fem(problem, np.array([length]), elements=40)
    expected_tip = end_force * length**3 / (3.0 * ei)
    expected_rotation = end_force * length**2 / (2.0 * ei)
    assert np.isclose(result["u"][0], expected_tip, rtol=1.0e-8)
    assert np.isclose(result["theta"][0], expected_rotation, rtol=1.0e-8)
    assert np.isclose(result["shear"][0], end_force, rtol=1.0e-8)

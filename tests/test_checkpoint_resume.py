from copy import deepcopy
from pathlib import Path

import torch

from structpinn.config import load_config
from structpinn.solver import solve, validate_transfer_compatibility

ROOT = Path(__file__).resolve().parents[1]


def _small_training_config() -> dict:
    config = load_config(ROOT / "configs" / "quick_smoke_test.yaml")
    config["model"]["hidden_layers"] = [8]
    config["training"]["adam_epochs"] = 6
    config["training"]["lbfgs_max_iter"] = 0
    config["training"]["collocation_points"] = 8
    config["training"]["boundary_points"] = 4
    config["training"]["adaptive_sampling"]["enabled"] = False
    config["training"]["adaptive_weighting"]["update_every"] = 2
    config["training"]["checkpointing"] = {"enabled": True, "every": 1, "keep_last": 1}
    config["training"]["log_every"] = 100
    config["validation"]["enabled"] = False
    config["output"]["prediction_points"] = 9
    config["output"]["save_plots"] = False
    return config


def test_adam_resume_reproduces_uninterrupted_parameters(tmp_path: Path) -> None:
    full = _small_training_config()
    solve(full, tmp_path / "full")

    partial = deepcopy(full)
    partial["training"]["adam_epochs"] = 3
    solve(partial, tmp_path / "partial")
    training_state_path = tmp_path / "partial" / "checkpoints" / "latest.pt"
    state = torch.load(training_state_path, map_location="cpu", weights_only=True)
    assert state["adam_epoch"] == 3
    assert state["optimizer_state_dict"]["state"]

    resumed = deepcopy(full)
    summary = solve(
        resumed,
        tmp_path / "resumed",
        resume_checkpoint=training_state_path,
    )
    full_model = torch.load(tmp_path / "full" / "model.pt", weights_only=True)
    resumed_model = torch.load(tmp_path / "resumed" / "model.pt", weights_only=True)
    for name, expected in full_model["state_dict"].items():
        torch.testing.assert_close(resumed_model["state_dict"][name], expected, rtol=0, atol=0)
    assert summary["resumed_from_epoch"] == 3


def test_transfer_allows_physics_change_but_rejects_network_change() -> None:
    source = _small_training_config()
    target = deepcopy(source)
    target["problem"]["geometry"]["length"] = 2.0
    target["problem"]["material"]["EI"] = 0.3
    target["problem"]["load"]["distributed"]["value"] = -2.0
    validate_transfer_compatibility(source, target)

    target["model"]["hidden_layers"] = [16]
    try:
        validate_transfer_compatibility(source, target)
    except ValueError as error:
        assert "network-incompatible" in str(error)
    else:
        raise AssertionError("Expected an incompatible warm start to be rejected.")

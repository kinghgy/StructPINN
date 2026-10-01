from pathlib import Path

import pytest
import yaml

from structpinn.config import load_config
from structpinn.experiments import _load_completed_run, aggregate_summaries, flatten_numeric

ROOT = Path(__file__).resolve().parents[1]


def test_flatten_numeric_ignores_booleans_and_non_finite_values() -> None:
    result = flatten_numeric({"a": {"b": 2}, "flag": True, "bad": float("nan")})
    assert result == {"a.b": 2.0}


def test_aggregate_uses_sample_standard_deviation_and_common_fields() -> None:
    result = aggregate_summaries(
        [
            {"elapsed_seconds": 10.0, "field_errors": {"w": {"relative_l2": 0.1}}},
            {
                "elapsed_seconds": 14.0,
                "field_errors": {"w": {"relative_l2": 0.3}, "Mx": {"relative_l2": 0.2}},
            },
        ]
    )
    elapsed = result["statistics"]["elapsed_seconds"]
    assert elapsed["mean"] == pytest.approx(12.0)
    assert elapsed["sample_std"] == pytest.approx(2.0**0.5 * 2.0)
    assert "field_errors.Mx.relative_l2" not in result["statistics"]


def test_aggregate_requires_at_least_one_run() -> None:
    with pytest.raises(ValueError, match="At least one"):
        aggregate_summaries([])


def test_resume_requires_the_same_resolved_config(tmp_path: Path) -> None:
    run_directory = tmp_path / "seed_11"
    run_directory.mkdir()
    config = load_config(ROOT / "configs" / "quick_smoke_test.yaml")
    config["training"]["seed"] = 11
    config["output"]["directory"] = str(run_directory)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with (run_directory / "config.resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(serializable, stream, sort_keys=False)
    (run_directory / "metrics.json").write_text("{}", encoding="utf-8")
    (run_directory / "model.pt").write_bytes(b"checkpoint-present")

    assert _load_completed_run(run_directory, 11, config) == {}
    config["training"]["learning_rate"] *= 0.5
    assert _load_completed_run(run_directory, 11, config) is None

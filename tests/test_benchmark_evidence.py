import csv
from pathlib import Path
from statistics import fmean, stdev

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_plate_comparison_is_recomputed_from_five_seed_evidence() -> None:
    baseline = _read_rows(BENCHMARKS / "plate_five_seed_projected_runs.csv")
    optimized = _read_rows(BENCHMARKS / "plate_five_seed_runs.csv")
    comparison = {
        row["metric"]: row
        for row in _read_rows(BENCHMARKS / "plate_five_seed_comparison.csv")
    }
    assert [int(row["seed"]) for row in baseline] == [11, 23, 37, 51, 67]
    assert [int(row["seed"]) for row in optimized] == [11, 23, 37, 51, 67]

    metric_keys = {
        "w_relative_l2": "field_errors.w.relative_l2",
        "kx_relative_l2": "field_errors.kx.relative_l2",
        "ky_relative_l2": "field_errors.ky.relative_l2",
        "kxy_relative_l2": "field_errors.kxy.relative_l2",
        "Mx_relative_l2": "field_errors.Mx.relative_l2",
        "My_relative_l2": "field_errors.My.relative_l2",
        "Mxy_relative_l2": "field_errors.Mxy.relative_l2",
        "projected_equilibrium_rms": "projected_normalized_residuals.equilibrium.rms",
    }
    for published_name, raw_name in metric_keys.items():
        baseline_values = [float(row[raw_name]) for row in baseline]
        optimized_values = [float(row[raw_name]) for row in optimized]
        published = comparison[published_name]
        baseline_mean = fmean(baseline_values)
        optimized_mean = fmean(optimized_values)
        assert float(published["projected_mean"]) == pytest.approx(baseline_mean, abs=5.0e-11)
        assert float(published["projected_sample_std"]) == pytest.approx(
            stdev(baseline_values), abs=5.0e-11
        )
        assert float(published["optimized_mean"]) == pytest.approx(optimized_mean, abs=5.0e-11)
        assert float(published["optimized_sample_std"]) == pytest.approx(
            stdev(optimized_values), abs=5.0e-11
        )
        change = 100.0 * (optimized_mean - baseline_mean) / baseline_mean
        assert float(published["mean_change_percent"]) == pytest.approx(change, abs=5.0e-4)


def test_device_benchmarks_pair_identical_seeds_and_losses() -> None:
    for filename in ("device_float32.csv", "device_float64.csv"):
        rows = _read_rows(BENCHMARKS / filename)
        assert len(rows) == 6
        for seed in (11, 12, 13):
            pair = [row for row in rows if int(row["seed"]) == seed]
            assert len(pair) == 2
            assert {row["device"].split(":")[0] for row in pair} == {"cpu", "cuda"}
            assert float(pair[0]["final_logged_loss"]) == pytest.approx(
                float(pair[1]["final_logged_loss"]), rel=1.0e-6
            )

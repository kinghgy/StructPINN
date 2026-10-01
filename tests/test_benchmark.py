import pytest

from structpinn.benchmark import benchmark_devices


@pytest.mark.parametrize(
    ("repetitions", "adam_epochs", "message"),
    [(0, 1, "repetitions"), (1, 0, "adam_epochs")],
)
def test_benchmark_rejects_non_positive_counts(
    repetitions: int, adam_epochs: int, message: str, tmp_path
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_devices("unused.yaml", ["cpu"], repetitions, adam_epochs, tmp_path)


def test_benchmark_rejects_duplicate_devices(tmp_path) -> None:
    with pytest.raises(ValueError, match="devices"):
        benchmark_devices("unused.yaml", ["cpu", "cpu"], 1, 1, tmp_path)


def test_benchmark_rejects_negative_lbfgs_iterations(tmp_path) -> None:
    with pytest.raises(ValueError, match="lbfgs_max_iter"):
        benchmark_devices("unused.yaml", ["cpu"], 1, 1, tmp_path, lbfgs_max_iter=-1)

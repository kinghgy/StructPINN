from pathlib import Path

from structpinn.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_are_valid() -> None:
    for path in (ROOT / "configs").rglob("*.yaml"):
        config = load_config(path)
        assert config["problem"]["type"] in {"beam", "plate"}

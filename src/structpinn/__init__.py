"""StructPINN: configurable physics-informed structural mechanics solvers."""

from .config import load_config, validate_config
from .solver import solve

__all__ = ["load_config", "solve", "validate_config"]
__version__ = "0.1.0"

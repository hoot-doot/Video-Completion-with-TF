"""Configuration loading and reproducibility helpers."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment specification."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> np.random.Generator:
    """Set Python/NumPy seeds and return the generator used by corruptions."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


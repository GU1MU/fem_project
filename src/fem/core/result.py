from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    """Result data for one solved model step."""
    model: Any
    step: Any
    U: np.ndarray
    reactions: np.ndarray
    name: str | None = None


@dataclass
class ModelResults:
    """Collection of solved model step results."""
    model: Any
    results: tuple[ModelResult, ...]

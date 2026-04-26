from __future__ import annotations

from pathlib import Path

from ..core.model import FEMModel
from .builder import build_model
from .parser import parse_file


def read(path: str | Path) -> FEMModel:
    """Read an Abaqus input file into a FEMModel."""
    return build_model(parse_file(path))

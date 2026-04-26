from __future__ import annotations

from typing import Any

from ..core.model import AnalysisStep


def static(name: str = "Step-1", **metadata: Any) -> AnalysisStep:
    """Create a static analysis step."""
    return AnalysisStep(str(name), procedure="static", metadata=metadata)


def add(model: Any, step: AnalysisStep) -> AnalysisStep:
    """Add a step to a model."""
    model.steps.append(step)
    return step

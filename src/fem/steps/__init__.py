from __future__ import annotations

from typing import Any, Sequence

from ..core.model import (
    AnalysisStep,
    DisplacementConstraint,
    NodalLoad,
    OutputRequest,
    Surface,
    SurfaceLoad,
)


def static(name: str = "Step-1", **metadata: Any) -> AnalysisStep:
    """Create a static analysis step."""
    return AnalysisStep(str(name), procedure="static", metadata=metadata)


def add(model: Any, step: AnalysisStep) -> AnalysisStep:
    """Add a step to a model."""
    model.steps.append(step)
    return step


def displacement(
    step: AnalysisStep,
    target: str | int,
    components: int | Sequence[int],
    value: float = 0.0,
) -> tuple[DisplacementConstraint, ...]:
    """Add displacement constraints to a step using 1-based components."""
    constraints = tuple(
        DisplacementConstraint(target, first, last, value)
        for first, last in _component_ranges(components)
    )
    step.boundaries = tuple(step.boundaries) + constraints
    return constraints


def nodal_load(
    step: AnalysisStep,
    target: str | int,
    component: int,
    value: float,
) -> NodalLoad:
    """Add a nodal load to a step using a 1-based component."""
    load = NodalLoad(target, component, value)
    step.cloads = tuple(step.cloads) + (load,)
    return load


def surface_traction(
    step: AnalysisStep,
    surface: str | Surface,
    vector: Sequence[float],
) -> SurfaceLoad:
    """Add a surface traction load to a step."""
    surface_name = surface.name if isinstance(surface, Surface) else str(surface)
    load = SurfaceLoad(surface_name, vector, load_type="traction")
    step.surface_loads = tuple(step.surface_loads) + (load,)
    return load


def surface_pressure(
    step: AnalysisStep,
    surface: str | Surface,
    magnitude: float,
) -> SurfaceLoad:
    """Add an inward pressure load to a step."""
    surface_name = surface.name if isinstance(surface, Surface) else str(surface)
    load = SurfaceLoad(surface_name, magnitude=magnitude, load_type="pressure")
    step.surface_loads = tuple(step.surface_loads) + (load,)
    return load


def output(
    step: AnalysisStep,
    kind: str,
    target: str,
    variables: Sequence[str] = (),
    **metadata: Any,
) -> OutputRequest:
    """Add an output request to a step."""
    request = OutputRequest(kind, target, variables, metadata)
    step.outputs = tuple(step.outputs) + (request,)
    return request


def _component_ranges(components: int | Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return compact contiguous component ranges."""
    if isinstance(components, int):
        values = (int(components),)
    else:
        values = tuple(int(component) for component in components)

    if not values:
        raise ValueError("components must not be empty")
    for component in values:
        if component < 1:
            raise ValueError("components are 1-based and must be positive")

    sorted_values = tuple(sorted(set(values)))
    ranges: list[tuple[int, int]] = []
    start = sorted_values[0]
    previous = start
    for component in sorted_values[1:]:
        if component == previous + 1:
            previous = component
            continue
        ranges.append((start, previous))
        start = previous = component
    ranges.append((start, previous))
    return tuple(ranges)

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.model import AnalysisStep, ElementFace, SurfaceLoad
from .condition import BoundaryCondition


def get_step(model: Any, step: str | int | AnalysisStep | None = None) -> AnalysisStep | None:
    """Return a model step by name or index."""
    if step is None:
        for candidate in model.steps:
            if candidate.name.lower() != "initial":
                return candidate
        return model.steps[0] if model.steps else None
    if isinstance(step, AnalysisStep):
        return step
    if isinstance(step, int):
        return model.steps[step]
    for candidate in model.steps:
        if candidate.name == step:
            return candidate
    raise KeyError(f"analysis step {step} is not defined")


def boundary_for_step(model: Any, step: str | int | AnalysisStep | None = None) -> BoundaryCondition:
    """Build solver boundary data for one model step."""
    selected_step = get_step(model, step)
    if selected_step is None:
        return BoundaryCondition()

    boundary = BoundaryCondition()
    for constraint in _step_boundaries(model, selected_step):
        for node_id in _resolve_node_target(model, constraint.target):
            for component in range(
                constraint.first_component,
                constraint.last_component + 1,
            ):
                _validate_component(model, component)
                boundary.add_displacement(
                    node_id,
                    component - 1,
                    constraint.value,
                    model.mesh,
                )

    for load in selected_step.cloads:
        _validate_component(model, load.component)
        for node_id in _resolve_node_target(model, load.target):
            boundary.add_nodal_force(
                node_id,
                load.component - 1,
                load.value,
                model.mesh,
            )

    for surface_load in selected_step.surface_loads:
        if surface_load.surface not in model.surfaces:
            raise KeyError(f"surface {surface_load.surface} is not defined")
        for face in model.surfaces[surface_load.surface].faces:
            if surface_load.load_type == "pressure":
                vector = _pressure_vector(model, face, surface_load)
            elif surface_load.load_type == "traction":
                vector = surface_load.vector
            else:
                raise ValueError(f"unsupported surface load type: {surface_load.load_type}")
            boundary.add_surface_traction(face.elem_id, face.local_index, *vector)

    return boundary


def _resolve_node_target(model: Any, target: str | int) -> tuple[int, ...]:
    """Resolve a node id or named node set."""
    if isinstance(target, int):
        return (target,)
    if target not in model.node_sets:
        raise KeyError(f"node set {target} is not defined")
    return model.node_sets[target].node_ids


def _step_boundaries(model: Any, step: AnalysisStep) -> tuple:
    """Return initial boundaries inherited by the selected step."""
    initial = next(
        (candidate for candidate in model.steps if candidate.name.lower() == "initial"),
        None,
    )
    if initial is None or initial is step:
        return tuple(step.boundaries)
    return tuple(initial.boundaries) + tuple(step.boundaries)


def _pressure_vector(
    model: Any,
    face: ElementFace,
    surface_load: SurfaceLoad,
) -> tuple[float, ...]:
    """Return an inward pressure vector for one surface face."""
    if surface_load.magnitude is None:
        raise ValueError("pressure surface load requires a magnitude")

    node_lookup = {node.id: node for node in model.mesh.nodes}
    coords = []
    for node_id in face.node_ids:
        node = node_lookup[node_id]
        coords.append([float(node.x), float(node.y), float(getattr(node, "z", 0.0))])
    if len(coords) < 3:
        raise ValueError(f"surface face {face} must contain at least 3 nodes for pressure")

    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[1], dtype=float)
    p2 = np.array(coords[2], dtype=float)
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError(f"surface face {face} has zero normal")
    return tuple(float(value) for value in -surface_load.magnitude * normal / norm)


def _validate_component(model: Any, component: int) -> None:
    """Validate a 1-based component against mesh DOFs."""
    if component < 1 or component > model.mesh.dofs_per_node:
        raise ValueError(
            f"component {component} is invalid for mesh with "
            f"{model.mesh.dofs_per_node} DOFs per node"
        )

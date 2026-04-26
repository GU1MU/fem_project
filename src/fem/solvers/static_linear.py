from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary.condition import BoundaryCondition
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..core.model import AnalysisStep, ElementFace, SurfaceLoad
from ..core.result import ModelResult, ModelResults
from . import linear


def solve(
    model: Any,
    step: str | int | AnalysisStep | None = None,
    output_dir: str | Path = "results",
    name: str | None = None,
) -> ModelResult:
    """Solve one linear static model step."""
    materials.apply_sections(model)
    selected_step = get_step(model, step)
    boundary = boundary_for_step(model, selected_step)
    K = assemble_global_stiffness_sparse(model.mesh)
    F = build_load_vector(model.mesh, boundary)
    K_mod, F_mod = apply_dirichlet(K, F, boundary)
    U = linear.solve(K_mod, F_mod)
    reactions = K @ U - F
    return ModelResult(
        model,
        selected_step,
        U,
        reactions,
        boundary,
        output_dir=output_dir,
        name=name,
    )


def solve_all(
    model: Any,
    selected_steps: Any = None,
    output_dir: str | Path = "results",
    name: str | None = None,
) -> ModelResults:
    """Solve multiple non-initial model steps."""
    steps = _solve_all_steps(model, selected_steps)
    multi_step = len(steps) > 1
    results = tuple(
        solve(
            model,
            step,
            output_dir=output_dir,
            name=_result_name(model, step, name, multi_step),
        )
        for step in steps
    )
    return ModelResults(model, results)


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
        return model.boundary if model.boundary is not None else BoundaryCondition()

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


def _solve_all_steps(model: Any, steps: Any) -> tuple[AnalysisStep | None, ...]:
    """Resolve solve_all step selectors."""
    if steps is None:
        runnable = tuple(step for step in model.steps if step.name.lower() != "initial")
        if runnable:
            return runnable
        if model.steps:
            return (model.steps[0],)
        return (None,)
    if isinstance(steps, (str, int, AnalysisStep)):
        return (get_step(model, steps),)
    return tuple(get_step(model, step) for step in steps)


def _result_name(
    model: Any,
    step: AnalysisStep | None,
    name: str | None,
    multi_step: bool,
) -> str | None:
    """Return a non-conflicting result name for solve_all."""
    if not multi_step:
        return name
    base = name or model.name or "result"
    step_name = step.name if step is not None else "step"
    return f"{base}_{step_name}"

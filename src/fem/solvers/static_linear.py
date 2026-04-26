from __future__ import annotations

from typing import Any

from .. import materials
from ..assemble import assemble_global_stiffness_sparse
from ..boundary.constraints import apply_dirichlet
from ..boundary.loads import build_load_vector
from ..boundary.step import boundary_for_step, get_step
from ..core.model import AnalysisStep
from ..core.result import ModelResult, ModelResults
from . import linear


def solve(
    model: Any,
    step: str | int | AnalysisStep | None = None,
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
        name=name,
    )


def solve_all(
    model: Any,
    selected_steps: Any = None,
    name: str | None = None,
) -> ModelResults:
    """Solve multiple non-initial model steps."""
    steps = _solve_all_steps(model, selected_steps)
    multi_step = len(steps) > 1
    results = tuple(
        solve(
            model,
            step,
            name=_result_name(model, step, name, multi_step),
        )
        for step in steps
    )
    return ModelResults(model, results)


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

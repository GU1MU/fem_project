from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._common import add_kernel_load, require_element, validate_vector
from .condition import ElementLoad


def add_forces(
    mesh: Any,
    loads: list[ElementLoad],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble constant element body forces."""
    for load in loads:
        elem = require_element(elem_lookup, load.elem_id)
        validate_vector(load.vector, spatial_dim, "body force")
        add_kernel_load(mesh, elem, node_lookup, F, "body_force", load.vector)


def add_gravity(
    mesh: Any,
    gravity: tuple[float, ...] | None,
    F: np.ndarray,
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble gravity as density-scaled body force."""
    if gravity is None:
        return

    validate_vector(gravity, spatial_dim, "gravity")
    for elem in mesh.elements:
        rho = elem.props.get("rho")
        if rho is not None:
            vector = tuple(float(rho) * value for value in gravity)
            add_kernel_load(mesh, elem, node_lookup, F, "body_force", vector)

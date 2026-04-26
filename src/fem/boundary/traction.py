from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._common import add_kernel_load, require_element, validate_vector
from .condition import SurfaceTraction


def add_forces(
    mesh: Any,
    tractions: list[SurfaceTraction],
    F: np.ndarray,
    elem_lookup: Dict[int, Any],
    node_lookup: Dict[int, Any],
    spatial_dim: int,
) -> None:
    """Assemble element edge or face tractions."""
    method_name = method_for_dim(spatial_dim)
    for traction in tractions:
        elem = require_element(elem_lookup, traction.elem_id)
        validate_vector(traction.vector, spatial_dim, "surface traction")
        add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            method_name,
            traction.vector,
            local_index=traction.local_index,
        )


def method_for_dim(dim: int) -> str:
    """Return kernel traction method for the mesh dimension."""
    if dim == 2:
        return "edge_traction"
    if dim == 3:
        return "face_traction"
    raise ValueError(f"unsupported mesh spatial dimension: {dim}")

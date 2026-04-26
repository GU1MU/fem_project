from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..elements import get_element_kernel

from .condition import BoundaryCondition


def build_load_vector(mesh: Any, bc: BoundaryCondition) -> np.ndarray:
    """Build global load vector from boundary conditions."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)
    _add_nodal_forces(F, bc.nodal_forces, num_dofs)

    elem_lookup = {elem.id: elem for elem in mesh.elements}
    node_lookup = {node.id: node for node in mesh.nodes}
    spatial_dim = _spatial_dim(mesh)

    for load in bc.body_forces:
        elem = _require_element(elem_lookup, load.elem_id)
        _validate_vector(load.vector, spatial_dim, "body force")
        _add_kernel_load(mesh, elem, node_lookup, F, "body_force", load.vector)

    if bc.gravity is not None:
        gravity = bc.gravity
        _validate_vector(gravity, spatial_dim, "gravity")
        for elem in mesh.elements:
            rho = elem.props.get("rho")
            if rho is not None:
                vector = tuple(float(rho) * value for value in gravity)
                _add_kernel_load(mesh, elem, node_lookup, F, "body_force", vector)

    traction_method = _traction_method(spatial_dim)
    for traction in bc.surface_tractions:
        elem = _require_element(elem_lookup, traction.elem_id)
        _validate_vector(traction.vector, spatial_dim, "surface traction")
        _add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            traction_method,
            traction.vector,
            local_index=traction.local_index,
        )

    return F


def _add_nodal_forces(F: np.ndarray, nodal_forces: Dict[int, float], num_dofs: int) -> None:
    """Assemble nodal forces into F."""
    for dof_id, value in nodal_forces.items():
        if dof_id < 0 or dof_id >= num_dofs:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {num_dofs})")
        F[dof_id] += float(value)


def _require_element(elem_lookup: Dict[int, Any], elem_id: int) -> Any:
    """Return element by id."""
    elem = elem_lookup.get(elem_id)
    if elem is None:
        raise KeyError(f"Element {elem_id} not found in mesh")
    return elem


def _spatial_dim(mesh: Any) -> int:
    """Return mesh coordinate dimension from node fields."""
    if not mesh.nodes:
        raise ValueError("mesh must contain at least one node")
    return 3 if hasattr(mesh.nodes[0], "z") else 2


def _traction_method(dim: int) -> str:
    """Return kernel traction method for the mesh dimension."""
    if dim == 2:
        return "edge_traction"
    if dim == 3:
        return "face_traction"
    raise ValueError(f"unsupported mesh spatial dimension: {dim}")


def _validate_vector(vector: tuple[float, ...], expected_size: int, name: str) -> None:
    """Validate a load vector size against mesh coordinates."""
    if len(vector) != expected_size:
        raise ValueError(
            f"{name} vector must have {expected_size} components, got {len(vector)}"
        )


def _add_kernel_load(
    mesh: Any,
    elem: Any,
    node_lookup: Dict[int, Any],
    F: np.ndarray,
    method_name: str,
    vector: tuple[float, ...],
    local_index: int | None = None,
) -> None:
    """Assemble one element load through an element kernel method."""
    kernel = get_element_kernel(elem.type)
    method = getattr(kernel, method_name, None)
    if method is None:
        raise NotImplementedError(
            f"Unsupported element type for {method_name} assembly: {elem.type}"
        )

    if local_index is None:
        fe = method(mesh, elem, vector, node_lookup)
    else:
        fe = method(mesh, elem, local_index, vector, node_lookup)
    F[mesh.element_dofs(elem)] += fe

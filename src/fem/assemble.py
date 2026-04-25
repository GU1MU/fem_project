from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .elements import get_element_kernel


def assemble_global_stiffness(
    mesh_or_num_dofs: Any = None,
    elements: Iterable | None = None,
    get_element_dofs: Callable[[object], Sequence[int]] | None = None,
    compute_element_stiffness: Callable[[object], np.ndarray] | None = None,
    *,
    num_dofs: int | None = None,
) -> np.ndarray:
    """Assemble a dense global stiffness matrix from mesh or callbacks."""
    if num_dofs is not None:
        if mesh_or_num_dofs is not None:
            raise TypeError("provide either mesh_or_num_dofs or num_dofs, not both")
        mesh_or_num_dofs = num_dofs
    if mesh_or_num_dofs is None:
        raise TypeError("dense assembly requires mesh or num_dofs")

    if elements is None and get_element_dofs is None and compute_element_stiffness is None:
        return _assemble_dense_from_mesh(mesh_or_num_dofs)

    if elements is None or get_element_dofs is None or compute_element_stiffness is None:
        raise TypeError("dense assembly requires either mesh or callback arguments")

    return _assemble_dense_from_callbacks(
        int(mesh_or_num_dofs),
        elements,
        get_element_dofs,
        compute_element_stiffness,
    )


def assemble_global_stiffness_sparse(
    mesh_or_num_dofs: Any = None,
    num_elements: int | None = None,
    get_element_dofs: Callable[[int], Sequence[int]] | None = None,
    compute_element_stiffness: Callable[[int], np.ndarray] | None = None,
    *,
    num_dofs: int | None = None,
) -> csr_matrix:
    """Assemble a sparse global stiffness matrix from mesh or callbacks."""
    if num_dofs is not None:
        if mesh_or_num_dofs is not None:
            raise TypeError("provide either mesh_or_num_dofs or num_dofs, not both")
        mesh_or_num_dofs = num_dofs
    if mesh_or_num_dofs is None:
        raise TypeError("sparse assembly requires mesh or num_dofs")

    if (
        num_elements is None
        and get_element_dofs is None
        and compute_element_stiffness is None
    ):
        return _assemble_sparse_from_mesh(mesh_or_num_dofs)

    if num_elements is None or get_element_dofs is None or compute_element_stiffness is None:
        raise TypeError("sparse assembly requires either mesh or callback arguments")

    return _assemble_sparse_from_callbacks(
        int(mesh_or_num_dofs),
        int(num_elements),
        get_element_dofs,
        compute_element_stiffness,
    )


def _assemble_dense_from_mesh(mesh: Any) -> np.ndarray:
    """Assemble dense stiffness using element kernels registered on the mesh."""
    node_lookup = {node.id: node for node in mesh.nodes}
    return _assemble_dense_from_callbacks(
        mesh.num_dofs,
        mesh.elements,
        lambda elem: mesh.element_dofs(elem),
        lambda elem: get_element_kernel(elem.type).stiffness(
            mesh,
            elem,
            node_lookup=node_lookup,
        ),
    )


def _assemble_sparse_from_mesh(mesh: Any) -> csr_matrix:
    """Assemble sparse stiffness using element kernels registered on the mesh."""
    node_lookup = {node.id: node for node in mesh.nodes}
    return _assemble_sparse_from_callbacks(
        mesh.num_dofs,
        len(mesh.elements),
        lambda eid: mesh.element_dofs(mesh.elements[eid]),
        lambda eid: get_element_kernel(mesh.elements[eid].type).stiffness(
            mesh,
            mesh.elements[eid],
            node_lookup=node_lookup,
        ),
    )


def _assemble_dense_from_callbacks(
    num_dofs: int,
    elements: Iterable,
    get_element_dofs: Callable[[object], Sequence[int]],
    compute_element_stiffness: Callable[[object], np.ndarray],
) -> np.ndarray:
    """Assemble dense stiffness from callback functions."""
    K = np.zeros((num_dofs, num_dofs), dtype=float)

    for elem in elements:
        Ke = compute_element_stiffness(elem)
        dofs = list(get_element_dofs(elem))
        _validate_element_stiffness(Ke, dofs, num_dofs, elem)

        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                K[I, J] += Ke[a, b]

    return K


def _assemble_sparse_from_callbacks(
    num_dofs: int,
    num_elements: int,
    get_element_dofs: Callable[[int], Sequence[int]],
    compute_element_stiffness: Callable[[int], np.ndarray],
) -> csr_matrix:
    """Assemble sparse stiffness from index-based callback functions."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for eid in range(num_elements):
        Ke = compute_element_stiffness(eid)
        dofs = list(get_element_dofs(eid))
        _validate_element_stiffness(Ke, dofs, num_dofs, eid)

        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                rows.append(I)
                cols.append(J)
                data.append(float(Ke[a, b]))

    return coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()


def _validate_element_stiffness(
    Ke: np.ndarray,
    dofs: Sequence[int],
    num_dofs: int,
    elem_label: object,
) -> None:
    """Validate element stiffness shape and DOF bounds."""
    nd = len(dofs)
    if Ke.shape != (nd, nd):
        raise ValueError(
            f"element {elem_label} stiffness shape {Ke.shape} does not match {nd} DOFs"
        )

    for dof in dofs:
        if dof < 0 or dof >= num_dofs:
            raise IndexError(
                f"element {elem_label} DOF index {dof} out of bounds [0, {num_dofs})"
            )

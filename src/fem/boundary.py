from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from scipy.sparse import csr_matrix

from .elements import get_element_kernel


@dataclass
class BoundaryCondition2D:
    """Store 2D Dirichlet BCs and loads."""
    prescribed_displacements: Dict[int, float] = field(default_factory=dict)
    nodal_forces: Dict[int, float] = field(default_factory=dict)
    body_forces: List[Tuple[int, float, float]] = field(default_factory=list)
    surface_tractions: List[Tuple[int, int, float, float]] = field(default_factory=list)
    gravity: Optional[Tuple[float, float]] = None

    def add_displacement_dof(self, dof_id: int, value: float = 0.0) -> None:
        """Add prescribed displacement on a dof."""
        self.prescribed_displacements[int(dof_id)] = float(value)

    def add_displacement(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add prescribed displacement by node and component."""
        self.add_displacement_dof(mesh.global_dof(node_id, component), value)

    def add_fixed_support(self, node_id: int, components: Optional[Iterable[int]], mesh: Any) -> None:
        """Fix selected components on a node."""
        if components is None:
            components = range(mesh.dofs_per_node)
        for component in components:
            self.add_displacement(node_id, component, 0.0, mesh)

    def add_nodal_force_dof(self, dof_id: int, value: float) -> None:
        """Add nodal force on a dof."""
        _accumulate_dof_value(self.nodal_forces, dof_id, value)

    def add_nodal_force(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add nodal force by node and component."""
        self.add_nodal_force_dof(mesh.global_dof(node_id, component), value)

    def add_body_force_element(self, elem_id: int, bx: float, by: float) -> None:
        """Add constant body force on an element."""
        self.body_forces.append((int(elem_id), float(bx), float(by)))

    def add_surface_traction(self, elem_id: int, local_edge: int, tx: float, ty: float) -> None:
        """Add constant edge traction on an element edge."""
        self.surface_tractions.append((int(elem_id), int(local_edge), float(tx), float(ty)))

    def set_gravity(self, gx: float, gy: float) -> None:
        """Set global gravity acceleration."""
        self.gravity = (float(gx), float(gy))


@dataclass
class BoundaryCondition3D:
    """Store 3D Dirichlet BCs and loads."""
    prescribed_displacements: Dict[int, float] = field(default_factory=dict)
    nodal_forces: Dict[int, float] = field(default_factory=dict)
    body_forces: List[Tuple[int, float, float, float]] = field(default_factory=list)
    surface_tractions: List[Tuple[int, int, float, float, float]] = field(default_factory=list)
    gravity: Optional[Tuple[float, float, float]] = None

    def add_displacement_dof(self, dof_id: int, value: float = 0.0) -> None:
        """Add prescribed displacement on a dof."""
        self.prescribed_displacements[int(dof_id)] = float(value)

    def add_displacement(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add prescribed displacement by node and component."""
        self.add_displacement_dof(mesh.global_dof(node_id, component), value)

    def add_fixed_support(self, node_id: int, components: Optional[Iterable[int]], mesh: Any) -> None:
        """Fix selected components on a node."""
        if components is None:
            components = range(mesh.dofs_per_node)
        for component in components:
            self.add_displacement(node_id, component, 0.0, mesh)

    def add_nodal_force_dof(self, dof_id: int, value: float) -> None:
        """Add nodal force on a dof."""
        _accumulate_dof_value(self.nodal_forces, dof_id, value)

    def add_nodal_force(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add nodal force by node and component."""
        self.add_nodal_force_dof(mesh.global_dof(node_id, component), value)

    def add_body_force_element(self, elem_id: int, bx: float, by: float, bz: float) -> None:
        """Add constant body force on an element."""
        self.body_forces.append((int(elem_id), float(bx), float(by), float(bz)))

    def add_surface_traction(self, elem_id: int, local_face: int, tx: float, ty: float, tz: float) -> None:
        """Add constant face traction on an element face."""
        self.surface_tractions.append((int(elem_id), int(local_face), float(tx), float(ty), float(tz)))

    def set_gravity(self, gx: float, gy: float, gz: float) -> None:
        """Set global gravity acceleration."""
        self.gravity = (float(gx), float(gy), float(gz))


def apply_dirichlet_bc(K: csr_matrix, F: np.ndarray, bc: BoundaryCondition2D) -> Tuple[csr_matrix, np.ndarray]:
    """Apply Dirichlet BCs to sparse K and F."""
    return _apply_dirichlet_bc(K, F, bc.prescribed_displacements)


def apply_dirichlet_bc_3d(K: csr_matrix, F: np.ndarray, bc: BoundaryCondition3D) -> Tuple[csr_matrix, np.ndarray]:
    """Apply 3D Dirichlet BCs to sparse K and F."""
    return _apply_dirichlet_bc(K, F, bc.prescribed_displacements)


def build_load_vector(mesh: Any, bc: BoundaryCondition2D) -> np.ndarray:
    """Build global 2D load vector."""
    return _build_load_vector(
        mesh,
        bc,
        traction_method="edge_traction",
        missing_element_error=KeyError,
        strict_kernel=False,
    )


def build_load_vector_3d(mesh: Any, bc: BoundaryCondition3D) -> np.ndarray:
    """Build global 3D load vector."""
    return _build_load_vector(
        mesh,
        bc,
        traction_method="face_traction",
        missing_element_error=ValueError,
        strict_kernel=True,
    )


def _accumulate_dof_value(values: Dict[int, float], dof_id: int, value: float) -> None:
    """Accumulate a scalar value on a DOF map."""
    dof_id = int(dof_id)
    values[dof_id] = values.get(dof_id, 0.0) + float(value)


def _apply_dirichlet_bc(
    K: csr_matrix,
    F: np.ndarray,
    prescribed_displacements: Dict[int, float],
) -> Tuple[csr_matrix, np.ndarray]:
    """Apply prescribed displacements by zeroing rows and columns."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")

    n = K.shape[0]
    F = np.asarray(F, dtype=float).ravel()
    if F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")

    K_mod = K.copy().tolil()
    F_mod = F.copy()

    for dof_id, value in prescribed_displacements.items():
        if dof_id < 0 or dof_id >= n:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {n})")
        value = float(value)
        if value != 0.0:
            F_mod -= value * K_mod[:, dof_id].toarray().ravel()
        K_mod[dof_id, :] = 0.0
        K_mod[:, dof_id] = 0.0
        K_mod[dof_id, dof_id] = 1.0
        F_mod[dof_id] = value

    return K_mod.tocsr(), F_mod


def _build_load_vector(
    mesh: Any,
    bc: Any,
    traction_method: str,
    missing_element_error: Type[Exception],
    strict_kernel: bool,
) -> np.ndarray:
    """Build global load vector from a boundary condition object."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)
    _add_nodal_forces(F, bc.nodal_forces, num_dofs)

    elem_lookup = {elem.id: elem for elem in mesh.elements}
    node_lookup = {node.id: node for node in mesh.nodes}

    for item in bc.body_forces:
        elem_id, *vector = item
        elem = _require_element(elem_lookup, elem_id, missing_element_error)
        _add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            "body_force",
            tuple(float(v) for v in vector),
            strict_kernel,
        )

    if bc.gravity is not None:
        gravity = tuple(float(v) for v in bc.gravity)
        for elem in mesh.elements:
            rho = _get_density(elem.props)
            if rho is not None:
                _add_kernel_load(
                    mesh,
                    elem,
                    node_lookup,
                    F,
                    "body_force",
                    tuple(rho * value for value in gravity),
                    strict_kernel,
                )

    for item in bc.surface_tractions:
        elem_id, local_index, *vector = item
        elem = _require_element(elem_lookup, elem_id, missing_element_error)
        _add_kernel_load(
            mesh,
            elem,
            node_lookup,
            F,
            traction_method,
            tuple(float(v) for v in vector),
            strict_kernel,
            local_index=int(local_index),
        )

    return F


def _add_nodal_forces(F: np.ndarray, nodal_forces: Dict[int, float], num_dofs: int) -> None:
    """Assemble nodal forces into F."""
    for dof_id, value in nodal_forces.items():
        if dof_id < 0 or dof_id >= num_dofs:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {num_dofs})")
        F[dof_id] += float(value)


def _require_element(
    elem_lookup: Dict[int, Any],
    elem_id: int,
    error_type: Type[Exception],
) -> Any:
    """Return element by id or raise the configured error."""
    elem = elem_lookup.get(elem_id)
    if elem is None:
        raise error_type(f"Element {elem_id} not found in mesh")
    return elem


def _get_density(props: Dict[str, Any]) -> Optional[float]:
    """Get density from element props with rho/rou compatibility."""
    if "rho" in props:
        return float(props["rho"])
    if "rou" in props:
        return float(props["rou"])
    return None


def _add_kernel_load(
    mesh: Any,
    elem: Any,
    node_lookup: Dict[int, Any],
    F: np.ndarray,
    method_name: str,
    vector: tuple[float, ...],
    strict_kernel: bool,
    local_index: int | None = None,
) -> None:
    """Assemble one element load through an element kernel method."""
    try:
        kernel = get_element_kernel(elem.type)
    except NotImplementedError:
        if strict_kernel:
            raise
        return

    method = getattr(kernel, method_name, None)
    if method is None:
        if strict_kernel:
            raise NotImplementedError(
                f"Unsupported element type for {method_name} assembly: {elem.type}"
            )
        return

    if local_index is None:
        fe = method(mesh, elem, vector, node_lookup)
    else:
        fe = method(mesh, elem, local_index, vector, node_lookup)
    F[mesh.element_dofs(elem)] += fe

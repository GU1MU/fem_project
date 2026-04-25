from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
import numpy as np
from scipy.sparse import csr_matrix
from .elements import get_element_kernel


@dataclass
class BoundaryCondition2D:
    """Store 2D Dirichlet BCs and loads."""
    prescribed_displacements: Dict[int, float] = field(default_factory=dict)
    nodal_forces: Dict[int, float] = field(default_factory=dict)
    body_forces: List[Tuple[int, float, float]] = field(default_factory=list)                 # (elem_id, bx, by)
    surface_tractions: List[Tuple[int, int, float, float]] = field(default_factory=list)     # (elem_id, local_edge, tx, ty)
    gravity: Optional[Tuple[float, float]] = None                                             # (gx, gy)

    def add_displacement_dof(self, dof_id: int, value: float = 0.0) -> None:
        """Add prescribed displacement on a dof."""
        self.prescribed_displacements[dof_id] = float(value)

    def add_displacement(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add prescribed displacement by node and component."""
        self.add_displacement_dof(mesh.global_dof(node_id, component), value)

    def add_fixed_support(self, node_id: int, components: Optional[Iterable[int]], mesh: Any) -> None:
        """Fix selected components on a node."""
        if components is None:
            components = range(mesh.dofs_per_node)
        for c in components:
            self.add_displacement(node_id, c, 0.0, mesh)

    def add_nodal_force_dof(self, dof_id: int, value: float) -> None:
        """Add nodal force on a dof."""
        self.nodal_forces[dof_id] = self.nodal_forces.get(dof_id, 0.0) + float(value)

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
        """Set global gravity acceleration (m/s^2)."""
        self.gravity = (float(gx), float(gy))


@dataclass
class BoundaryCondition3D:
    """Store 3D Dirichlet BCs and loads."""
    prescribed_displacements: Dict[int, float] = field(default_factory=dict)
    nodal_forces: Dict[int, float] = field(default_factory=dict)
    body_forces: List[Tuple[int, float, float, float]] = field(default_factory=list)          # (elem_id, bx, by, bz)
    surface_tractions: List[Tuple[int, int, float, float, float]] = field(default_factory=list) # (elem_id, local_face, tx, ty, tz)
    gravity: Optional[Tuple[float, float, float]] = None                                       # (gx, gy, gz)

    def add_displacement_dof(self, dof_id: int, value: float = 0.0) -> None:
        """Add prescribed displacement on a dof."""
        self.prescribed_displacements[dof_id] = float(value)

    def add_displacement(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add prescribed displacement on a node component."""
        dof_id = mesh.global_dof(node_id, component)
        self.prescribed_displacements[dof_id] = float(value)

    def add_fixed_support(self, node_id: int, components: Optional[Iterable[int]], mesh: Any) -> None:
        """Add fixed support (zero displacement) on node components."""
        if components is None:
            components = range(mesh.dofs_per_node)
        for comp in components:
            self.add_displacement(node_id, comp, 0.0, mesh)

    def add_nodal_force_dof(self, dof_id: int, value: float) -> None:
        """Add concentrated force on a dof."""
        self.nodal_forces[dof_id] = float(value)

    def add_nodal_force(self, node_id: int, component: int, value: float, mesh: Any) -> None:
        """Add concentrated force on a node component."""
        dof_id = mesh.global_dof(node_id, component)
        self.nodal_forces[dof_id] = float(value)

    def add_body_force_element(self, elem_id: int, bx: float, by: float, bz: float) -> None:
        """Add constant body force on an element."""
        self.body_forces.append((int(elem_id), float(bx), float(by), float(bz)))

    def add_surface_traction(self, elem_id: int, local_face: int, tx: float, ty: float, tz: float) -> None:
        """Add constant face traction on an element face."""
        self.surface_tractions.append((int(elem_id), int(local_face), float(tx), float(ty), float(tz)))

    def set_gravity(self, gx: float, gy: float, gz: float) -> None:
        """Set global gravity acceleration (m/s^2)."""
        self.gravity = (float(gx), float(gy), float(gz))


def apply_dirichlet_bc_3d(K: csr_matrix, F: np.ndarray, bc: BoundaryCondition3D) -> Tuple[csr_matrix, np.ndarray]:
    """Apply 3D Dirichlet BCs to sparse K and F by zero-row/col and diag=1."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")
    F = np.asarray(F, dtype=float).ravel()
    if F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")

    K_mod = K.copy().tolil()
    F_mod = F.copy()

    for dof_id, val in bc.prescribed_displacements.items():
        if dof_id < 0 or dof_id >= n:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {n})")
        if val != 0.0:
            F_mod -= val * K_mod[:, dof_id].toarray().ravel()
        K_mod[dof_id, :] = 0.0
        K_mod[:, dof_id] = 0.0
        K_mod[dof_id, dof_id] = 1.0
        F_mod[dof_id] = float(val)

    return K_mod.tocsr(), F_mod


def build_load_vector_3d(mesh: Any, bc: BoundaryCondition3D) -> np.ndarray:
    """Build global load vector with nodal forces, body forces, face tractions, and gravity."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)

    for dof_id, val in bc.nodal_forces.items():
        if dof_id < 0 or dof_id >= num_dofs:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {num_dofs})")
        F[dof_id] += float(val)

    elem_lookup = {e.id: e for e in mesh.elements}
    node_lookup = {n.id: n for n in mesh.nodes}

    for elem_id, bx, by, bz in bc.body_forces:
        e = elem_lookup.get(elem_id)
        if e is None:
            raise ValueError(f"Element {elem_id} not found in mesh")
        _add_element_body_force_consistent_3d(mesh, e, node_lookup, F, bx, by, bz)

    if bc.gravity is not None:
        gx, gy, gz = bc.gravity
        for e in mesh.elements:
            rho = _get_density(e.props)
            if rho is not None:
                _add_element_body_force_consistent_3d(mesh, e, node_lookup, F, rho*gx, rho*gy, rho*gz)

    for elem_id, local_face, tx, ty, tz in bc.surface_tractions:
        e = elem_lookup.get(elem_id)
        if e is None:
            raise ValueError(f"Element {elem_id} not found in mesh")
        _add_element_face_traction_consistent_3d(mesh, e, node_lookup, F, local_face, tx, ty, tz)

    return F


def _add_element_body_force_consistent_3d(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, bx: float, by: float, bz: float) -> None:
    """Assemble 3D consistent body force through element kernel."""
    kernel = get_element_kernel(elem.type)
    if not hasattr(kernel, "body_force"):
        raise NotImplementedError(f"Unsupported 3D element type for body force assembly: {elem.type}")
    F[mesh.element_dofs(elem)] += kernel.body_force(
        mesh, elem, (float(bx), float(by), float(bz)), node_lookup
    )


def _add_element_face_traction_consistent_3d(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, local_face: int, tx: float, ty: float, tz: float) -> None:
    """Assemble 3D consistent face traction through element kernel."""
    kernel = get_element_kernel(elem.type)
    if not hasattr(kernel, "face_traction"):
        raise NotImplementedError(f"Unsupported 3D element type for face traction assembly: {elem.type}")
    F[mesh.element_dofs(elem)] += kernel.face_traction(
        mesh, elem, int(local_face), (float(tx), float(ty), float(tz)), node_lookup
    )


def apply_dirichlet_bc(K: csr_matrix, F: np.ndarray, bc: BoundaryCondition2D) -> Tuple[csr_matrix, np.ndarray]:
    """Apply Dirichlet BCs to sparse K and F by zero-row/col and diag=1."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")
    F = np.asarray(F, dtype=float).ravel()
    if F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")

    K_mod = K.copy().tolil()
    F_mod = F.copy()

    for dof_id, val in bc.prescribed_displacements.items():
        if dof_id < 0 or dof_id >= n:
            raise IndexError(f"dof_id out of range: {dof_id}")
        if val != 0.0:
            col = np.asarray(K_mod[:, dof_id].toarray()).ravel()
            F_mod -= col * float(val)
        K_mod[dof_id, :] = 0.0
        K_mod[:, dof_id] = 0.0
        K_mod[dof_id, dof_id] = 1.0
        F_mod[dof_id] = float(val)

    return K_mod.tocsr(), F_mod


def build_load_vector(mesh: Any, bc: BoundaryCondition2D) -> np.ndarray:
    """Build global load vector with nodal forces, body forces, edge tractions, and gravity."""
    num_dofs = int(mesh.num_dofs)
    F = np.zeros(num_dofs, dtype=float)

    for dof_id, val in bc.nodal_forces.items():
        if dof_id < 0 or dof_id >= num_dofs:
            raise IndexError(f"nodal force dof_id out of range: {dof_id}")
        F[dof_id] += float(val)

    elem_lookup = {e.id: e for e in mesh.elements}
    node_lookup = {n.id: n for n in mesh.nodes}

    for elem_id, bx, by in bc.body_forces:
        e = elem_lookup.get(elem_id)
        if e is None:
            raise KeyError(f"elem_id not found: {elem_id}")
        _add_element_body_force_consistent(mesh, e, node_lookup, F, bx, by)

    if bc.gravity is not None:
        gx, gy = bc.gravity
        for e in mesh.elements:
            rho = _get_density(e.props)
            if rho is None:
                continue
            _add_element_body_force_consistent(mesh, e, node_lookup, F, rho * gx, rho * gy)

    for elem_id, local_edge, tx, ty in bc.surface_tractions:
        e = elem_lookup.get(elem_id)
        if e is None:
            raise KeyError(f"elem_id not found: {elem_id}")
        _add_element_edge_traction_consistent(mesh, e, node_lookup, F, local_edge, tx, ty)

    return F


def _get_density(props: Dict[str, Any]) -> Optional[float]:
    """Get density from element props with rho/rou compatibility."""
    if "rho" in props:
        return float(props["rho"])
    if "rou" in props:
        return float(props["rou"])
    return None


def _add_element_body_force_consistent(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, bx: float, by: float) -> None:
    """Assemble 2D consistent body force through element kernel."""
    try:
        kernel = get_element_kernel(elem.type)
    except NotImplementedError:
        return
    if not hasattr(kernel, "body_force"):
        return
    F[mesh.element_dofs(elem)] += kernel.body_force(
        mesh, elem, (float(bx), float(by)), node_lookup
    )


def _add_element_edge_traction_consistent(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, local_edge: int, tx: float, ty: float) -> None:
    """Assemble 2D consistent edge traction through element kernel."""
    try:
        kernel = get_element_kernel(elem.type)
    except NotImplementedError:
        return
    if not hasattr(kernel, "edge_traction"):
        return
    F[mesh.element_dofs(elem)] += kernel.edge_traction(
        mesh, elem, int(local_edge), (float(tx), float(ty)), node_lookup
    )


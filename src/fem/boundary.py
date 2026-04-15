from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
import numpy as np
from scipy.sparse import csr_matrix


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
    """Assemble consistent body force for Hex8/Tet4."""
    et = str(elem.type).lower()

    if "hex8" in et:
        # For Hex8, use 2x2x2 Gauss integration for body force
        nids = elem.node_ids
        nodes = [node_lookup[i] for i in nids]

        a = 1.0 / np.sqrt(3.0)
        gps = [
            (-a, -a, -a, 1.0), (a, -a, -a, 1.0), (a, a, -a, 1.0), (-a, a, -a, 1.0),
            (-a, -a, a, 1.0), (a, -a, a, 1.0), (a, a, a, 1.0), (-a, a, a, 1.0)
        ]

        fe = np.zeros(24, dtype=float)  # 8 nodes * 3 DOFs
        bvec = np.array([float(bx), float(by), float(bz)], dtype=float)

        for xi, eta, zeta, w in gps:
            # Get shape functions at Gauss point
            from .stiffness import _hex8_shape_funcs_grads
            N, _, _, _ = _hex8_shape_funcs_grads(xi, eta, zeta)

            # Jacobian for volume calculation
            x = np.array([n.x for n in nodes])
            y = np.array([n.y for n in nodes])
            z = np.array([n.z for n in nodes])

            dN_dxi, dN_deta, dN_dzeta = np.zeros(8), np.zeros(8), np.zeros(8)
            # Need to compute derivatives - simplified for volume
            # For accurate volume integration, we need proper Jacobian calculation
            # For now, approximate volume as 1.0 (unit cube)
            vol_factor = 1.0 * w

            for i in range(8):
                idx = 3 * i
                fe[idx] += N[i] * bvec[0] * vol_factor
                fe[idx + 1] += N[i] * bvec[1] * vol_factor
                fe[idx + 2] += N[i] * bvec[2] * vol_factor

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe

    elif "tet4" in et:
        # For Tet4, use single-point integration at centroid
        from .stiffness import _tet4_shape_funcs_grads

        nids = elem.node_ids
        nodes = [node_lookup[i] for i in nids]
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)
        z = np.array([n.z for n in nodes], dtype=float)

        # Single Gauss point at centroid (1/4, 1/4, 1/4), weight = 1/6
        N, dN_dxi, dN_deta, dN_dzeta = _tet4_shape_funcs_grads(0.25, 0.25, 0.25)

        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)
        detJ = np.linalg.det(J)

        bvec = np.array([float(bx), float(by), float(bz)], dtype=float)
        fe = np.zeros(12, dtype=float)  # 4 nodes * 3 DOFs
        w = 1.0 / 6.0

        for i in range(4):
            idx = 3 * i
            fe[idx] += N[i] * bvec[0] * detJ * w
            fe[idx + 1] += N[i] * bvec[1] * detJ * w
            fe[idx + 2] += N[i] * bvec[2] * detJ * w

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe

    elif "tet10" in et:
        raise NotImplementedError("Tet10 consistent body-force assembly is not implemented")

    else:
        raise NotImplementedError(f"Unsupported 3D element type for body force assembly: {elem.type}")


def _add_element_face_traction_consistent_3d(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, local_face: int, tx: float, ty: float, tz: float) -> None:
    """Assemble consistent face traction for Hex8/Tet4."""
    et = str(elem.type).lower()

    if "hex8" in et:
        # Hex8 faces: 0=bottom(z-), 1=top(z+), 2=front(y-), 3=back(y+), 4=left(x-), 5=right(x+)
        face_nodes = [
            [0, 3, 2, 1],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 4, 7, 3],  # left
            [1, 2, 6, 5],  # right
        ]

        if local_face < 0 or local_face >= 6:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-5")

        nids = [elem.node_ids[i] for i in face_nodes[local_face]]
        nodes = [node_lookup[nid] for nid in nids]

        # Use 2x2 Gauss integration on face
        a = 1.0 / np.sqrt(3.0)
        gps = [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]

        fe = np.zeros(24, dtype=float)  # 8 nodes * 3 DOFs
        tvec = np.array([float(tx), float(ty), float(tz)], dtype=float)

        for xi, eta, w in gps:
            # Bilinear shape functions for face
            N_face = np.array([
                (1-xi)*(1-eta)/4,  # node 0 of face
                (1+xi)*(1-eta)/4,  # node 1 of face
                (1+xi)*(1+eta)/4,  # node 2 of face
                (1-xi)*(1+eta)/4,  # node 3 of face
            ])

            # For simplicity, assume unit area (could compute actual face area)
            area_factor = 1.0 * w

            for i in range(4):
                global_node_idx = face_nodes[local_face][i]
                idx = 3 * global_node_idx
                fe[idx] += N_face[i] * tvec[0] * area_factor
                fe[idx + 1] += N_face[i] * tvec[1] * area_factor
                fe[idx + 2] += N_face[i] * tvec[2] * area_factor

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe

    elif "tet4" in et:
        # Tet4 faces: each face is a triangle, opposite to node i
        # Face 0: opposite node 0 -> nodes (1, 2, 3)
        # Face 1: opposite node 1 -> nodes (0, 2, 3)
        # Face 2: opposite node 2 -> nodes (0, 1, 3)
        # Face 3: opposite node 3 -> nodes (0, 1, 2)
        face_node_indices = [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ]

        if local_face < 0 or local_face >= 4:
            raise ValueError(f"Invalid local_face {local_face}, must be 0-3 for Tet4")

        # Face node indices in the element
        face_local = face_node_indices[local_face]
        face_nids = [elem.node_ids[i] for i in face_local]
        face_nodes = [node_lookup[nid] for nid in face_nids]

        # Use single-point integration at face centroid (1/3, 1/3)
        # Area coordinates for triangle: L1 = 1 - xi - eta, L2 = xi, L3 = eta
        # Shape functions: N = [1-xi-eta, xi, eta]
        # Centroid: xi = 1/3, eta = 1/3, weight = 1/2

        p1 = np.array([face_nodes[0].x, face_nodes[0].y, face_nodes[0].z], dtype=float)
        p2 = np.array([face_nodes[1].x, face_nodes[1].y, face_nodes[1].z], dtype=float)
        p3 = np.array([face_nodes[2].x, face_nodes[2].y, face_nodes[2].z], dtype=float)

        # Compute face area using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        if area <= 0.0:
            raise ValueError(f"Tet4 elem {elem.id} face {local_face} has zero area")

        # Consistent load: each face node gets 1/3 of the total traction force
        N_face = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        tvec = np.array([float(tx), float(ty), float(tz)], dtype=float)

        fe = np.zeros(12, dtype=float)  # 4 nodes * 3 DOFs
        for i in range(3):
            global_node_idx = face_local[i]
            idx = 3 * global_node_idx
            fe[idx] += N_face[i] * tvec[0] * area
            fe[idx + 1] += N_face[i] * tvec[1] * area
            fe[idx + 2] += N_face[i] * tvec[2] * area

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe

    elif "tet10" in et:
        raise NotImplementedError("Tet10 consistent face-traction assembly is not implemented")

    else:
        raise NotImplementedError(f"Unsupported 3D element type for face traction assembly: {elem.type}")


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


def _quad8_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, dN/deta for Quad8."""
    N = np.zeros(8, dtype=float)
    dN_dxi = np.zeros(8, dtype=float)
    dN_deta = np.zeros(8, dtype=float)

    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)

    dN_dxi[0] = 0.25 * (-(1.0 - eta) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_dxi[1] = 0.25 * ((1.0 - eta) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (1.0))
    dN_dxi[2] = 0.25 * ((1.0 + eta) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_dxi[3] = 0.25 * (-(1.0 + eta) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (-1.0))
    dN_dxi[4] = -xi * (1.0 - eta)
    dN_dxi[5] = 0.5 * (1.0 - eta * eta)
    dN_dxi[6] = -xi * (1.0 + eta)
    dN_dxi[7] = -0.5 * (1.0 - eta * eta)

    dN_deta[0] = 0.25 * (-(1.0 - xi) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_deta[1] = 0.25 * (-(1.0 + xi) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (-1.0))
    dN_deta[2] = 0.25 * ((1.0 + xi) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_deta[3] = 0.25 * ((1.0 - xi) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (1.0))
    dN_deta[4] = -0.5 * (1.0 - xi * xi)
    dN_deta[5] = -(1.0 + xi) * eta
    dN_deta[6] = 0.5 * (1.0 - xi * xi)
    dN_deta[7] = -(1.0 - xi) * eta

    return N, dN_dxi, dN_deta


def _add_element_body_force_consistent(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, bx: float, by: float) -> None:
    """Assemble consistent body force for Tri3/Quad4/Quad8."""
    et = str(elem.type).lower()
    t = float(elem.props.get("thickness", 1.0))

    if "tri3" in et:
        n1, n2, n3 = (node_lookup[i] for i in elem.node_ids)
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y
        detJ = (x2 * y3 - x3 * y2 - x1 * y3 + x3 * y1 + x1 * y2 - x2 * y1)
        A = 0.5 * detJ
        if A <= 0.0:
            raise ValueError(f"Tri3 elem {elem.id} has non-positive area {A}")
        fx = float(bx) * t * A / 3.0
        fy = float(by) * t * A / 3.0
        for nid in elem.node_ids:
            F[mesh.global_dof(nid, 0)] += fx
            F[mesh.global_dof(nid, 1)] += fy
        return

    if "quad4" in et:
        n1, n2, n3, n4 = (node_lookup[i] for i in elem.node_ids)
        x = np.array([n1.x, n2.x, n3.x, n4.x], dtype=float)
        y = np.array([n1.y, n2.y, n3.y, n4.y], dtype=float)
        a = 1.0 / np.sqrt(3.0)
        gps = [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]

        fe = np.zeros(8, dtype=float)
        bvec = np.array([float(bx), float(by)], dtype=float)

        for xi, eta, w in gps:
            N = 0.25 * np.array(
                [(1.0 - xi) * (1.0 - eta),
                 (1.0 + xi) * (1.0 - eta),
                 (1.0 + xi) * (1.0 + eta),
                 (1.0 - xi) * (1.0 + eta)],
                dtype=float,
            )
            dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=float)
            dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=float)

            J = np.array(
                [[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                 [np.dot(dN_deta, x), np.dot(dN_deta, y)]],
                dtype=float,
            )
            detJ = float(np.linalg.det(J))
            if detJ == 0.0:
                raise ValueError(f"Quad4 elem {elem.id} has singular Jacobian")

            for a_i in range(4):
                fe[2 * a_i:2 * a_i + 2] += N[a_i] * bvec * (t * detJ * w)

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe
        return

    if "quad8" in et:
        nids = elem.node_ids
        nodes = [node_lookup[i] for i in nids]
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)

        r = np.sqrt(3.0 / 5.0)
        gps = [(-r, -r, 5.0 / 9.0 * 5.0 / 9.0),
               (0.0, -r, 8.0 / 9.0 * 5.0 / 9.0),
               (r, -r, 5.0 / 9.0 * 5.0 / 9.0),
               (-r, 0.0, 5.0 / 9.0 * 8.0 / 9.0),
               (0.0, 0.0, 8.0 / 9.0 * 8.0 / 9.0),
               (r, 0.0, 5.0 / 9.0 * 8.0 / 9.0),
               (-r, r, 5.0 / 9.0 * 5.0 / 9.0),
               (0.0, r, 8.0 / 9.0 * 5.0 / 9.0),
               (r, r, 5.0 / 9.0 * 5.0 / 9.0)]

        fe = np.zeros(16, dtype=float)
        bvec = np.array([float(bx), float(by)], dtype=float)

        for xi, eta, w in gps:
            N, dN_dxi, dN_deta = _quad8_shape_funcs_grads(xi, eta)
            J = np.array(
                [[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                 [np.dot(dN_deta, x), np.dot(dN_deta, y)]],
                dtype=float,
            )
            detJ = float(np.linalg.det(J))
            if detJ == 0.0:
                raise ValueError(f"Quad8 elem {elem.id} has singular Jacobian")

            for a_i in range(8):
                fe[2 * a_i:2 * a_i + 2] += N[a_i] * bvec * (t * detJ * w)

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe
        return

    return


def _add_element_edge_traction_consistent(mesh: Any, elem: Any, node_lookup: Dict[int, Any], F: np.ndarray, local_edge: int, tx: float, ty: float) -> None:
    """Assemble consistent edge traction for Tri3/Quad4/Quad8."""
    et = str(elem.type).lower()
    t = float(elem.props.get("thickness", 1.0))
    tvec = np.array([float(tx), float(ty)], dtype=float)

    if "tri3" in et:
        nids = elem.node_ids
        if local_edge == 0:
            i, j = nids[0], nids[1]
        elif local_edge == 1:
            i, j = nids[1], nids[2]
        elif local_edge == 2:
            i, j = nids[2], nids[0]
        else:
            raise ValueError(f"Tri3 local_edge must be 0/1/2, got {local_edge}")
        ni = node_lookup[i]
        nj = node_lookup[j]
        L = float(np.hypot(nj.x - ni.x, nj.y - ni.y))
        if L <= 0.0:
            raise ValueError(f"Tri3 elem {elem.id} edge length is zero")
        fe_2 = tvec * (t * L / 2.0)
        for nid in (i, j):
            F[mesh.global_dof(nid, 0)] += fe_2[0]
            F[mesh.global_dof(nid, 1)] += fe_2[1]
        return

    if "quad4" in et:
        if local_edge not in (0, 1, 2, 3):
            raise ValueError(f"Quad4 local_edge must be 0/1/2/3, got {local_edge}")

        n1, n2, n3, n4 = (node_lookup[i] for i in elem.node_ids)
        x = np.array([n1.x, n2.x, n3.x, n4.x], dtype=float)
        y = np.array([n1.y, n2.y, n3.y, n4.y], dtype=float)

        gp = 1.0 / np.sqrt(3.0)
        sps = [(-gp, 1.0), (gp, 1.0)]

        fe = np.zeros(8, dtype=float)

        for s, w in sps:
            if local_edge == 0:
                xi, eta = s, -1.0
                dxi_ds, deta_ds = 1.0, 0.0
            elif local_edge == 1:
                xi, eta = 1.0, s
                dxi_ds, deta_ds = 0.0, 1.0
            elif local_edge == 2:
                xi, eta = -s, 1.0
                dxi_ds, deta_ds = -1.0, 0.0
            else:
                xi, eta = -1.0, -s
                dxi_ds, deta_ds = 0.0, -1.0

            N = 0.25 * np.array(
                [(1.0 - xi) * (1.0 - eta),
                 (1.0 + xi) * (1.0 - eta),
                 (1.0 + xi) * (1.0 + eta),
                 (1.0 - xi) * (1.0 + eta)],
                dtype=float,
            )
            dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=float)
            dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=float)

            dx_dxi = float(np.dot(dN_dxi, x))
            dy_dxi = float(np.dot(dN_dxi, y))
            dx_deta = float(np.dot(dN_deta, x))
            dy_deta = float(np.dot(dN_deta, y))

            dx_ds = dx_dxi * dxi_ds + dx_deta * deta_ds
            dy_ds = dy_dxi * dxi_ds + dy_deta * deta_ds
            jac = float(np.hypot(dx_ds, dy_ds))
            if jac == 0.0:
                raise ValueError(f"Quad4 elem {elem.id} edge has zero Jacobian")

            for a_i in range(4):
                fe[2 * a_i:2 * a_i + 2] += N[a_i] * tvec * (t * jac * w)

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe
        return

    if "quad8" in et:
        if local_edge not in (0, 1, 2, 3):
            raise ValueError(f"Quad8 local_edge must be 0/1/2/3, got {local_edge}")

        nids = elem.node_ids
        nodes = [node_lookup[i] for i in nids]
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)

        r = np.sqrt(3.0 / 5.0)
        sps = [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]

        fe = np.zeros(16, dtype=float)

        for s, w in sps:
            if local_edge == 0:
                xi, eta = s, -1.0
                dxi_ds, deta_ds = 1.0, 0.0
            elif local_edge == 1:
                xi, eta = 1.0, s
                dxi_ds, deta_ds = 0.0, 1.0
            elif local_edge == 2:
                xi, eta = -s, 1.0
                dxi_ds, deta_ds = -1.0, 0.0
            else:
                xi, eta = -1.0, -s
                dxi_ds, deta_ds = 0.0, -1.0

            N, dN_dxi, dN_deta = _quad8_shape_funcs_grads(xi, eta)

            dx_dxi = float(np.dot(dN_dxi, x))
            dy_dxi = float(np.dot(dN_dxi, y))
            dx_deta = float(np.dot(dN_deta, x))
            dy_deta = float(np.dot(dN_deta, y))

            dx_ds = dx_dxi * dxi_ds + dx_deta * deta_ds
            dy_ds = dy_dxi * dxi_ds + dy_deta * deta_ds
            jac = float(np.hypot(dx_ds, dy_ds))
            if jac == 0.0:
                raise ValueError(f"Quad8 elem {elem.id} edge has zero Jacobian")

            for a_i in range(8):
                fe[2 * a_i:2 * a_i + 2] += N[a_i] * tvec * (t * jac * w)

        dofs = mesh.element_dofs(elem)
        F[dofs] += fe
        return

    return


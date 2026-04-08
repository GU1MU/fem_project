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


from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from ..materials import compute_plane_elastic_matrix


def quad8_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, and dN/deta for Quad8."""
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


def quad8_gauss_points(gauss_order: int):
    """Return Gauss points for Quad8."""
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    if gauss_order == 3:
        r = np.sqrt(3.0 / 5.0)
        one_d = [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]
        pts = []
        for xi, wx in one_d:
            for eta, wy in one_d:
                pts.append((xi, eta, wx * wy))
        return pts
    raise ValueError("gauss_order must be 2 or 3 for Quad8")


class Quad8PlaneKernel:
    """Quad8 plane stress/strain element kernel."""
    type_names = ("Quad8Plane", "Quad8", "CPS8", "CPE8")

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 3,
    ) -> np.ndarray:
        """Return Quad8 plane element stiffness."""
        if len(elem.node_ids) != 8:
            raise ValueError(f"Quad8 needs 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

        D, t = self._material_data(elem)
        Ke = np.zeros((16, 16), dtype=float)
        for xi, eta, w in quad8_gauss_points(gauss_order):
            B, detJ = self._B_matrix(mesh, elem, xi, eta, node_lookup)
            Ke += (B.T @ D @ B) * (t * detJ * w)
        return Ke

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return stress at one natural coordinate point."""
        D, _ = self._material_data(elem)
        B, _ = self._B_matrix(mesh, elem, xi, eta, node_lookup)
        return D @ (B @ U[mesh.element_dofs(elem)])

    def _material_data(self, elem: Any):
        """Return D matrix and thickness from element props."""
        try:
            E = float(elem.props["E"])
            nu = float(elem.props["nu"])
            t = float(elem.props.get("thickness", 1.0))
        except KeyError as e:
            raise KeyError(f"elem {elem.id} missing '{e.args[0]}' in props={elem.props}")

        pt = str(elem.props.get("plane_type", "stress")).lower()
        D = compute_plane_elastic_matrix(E, nu, pt)
        return D, t

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        xi: float,
        eta: float,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and detJ at one natural coordinate point."""
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        nodes = [node_lookup[i] for i in elem.node_ids]
        x = np.array([n.x for n in nodes], dtype=float)
        y = np.array([n.y for n in nodes], dtype=float)

        _, dN_dxi, dN_deta = quad8_shape_funcs_grads(xi, eta)
        J = np.array(
            [[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
             [np.dot(dN_deta, x), np.dot(dN_deta, y)]],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError(f"elem {elem.id} singular Jacobian")

        dN_xy = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
        B = np.zeros((3, 16), dtype=float)
        for a_i in range(8):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        return B, detJ

from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from ..materials import compute_plane_elastic_matrix


def quad4_shape_grad_xi_eta(xi: float, eta: float) -> np.ndarray:
    """Return dN/dxi and dN/deta for bilinear Quad4."""
    dN_dxi = np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    ) * 0.25
    dN_deta = np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    ) * 0.25
    return np.vstack([dN_dxi, dN_deta])


def quad4_gauss_points(gauss_order: int):
    """Return Gauss points for Quad4."""
    if gauss_order == 1:
        return [(0.0, 0.0, 4.0)]
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    raise ValueError("gauss_order must be 1 or 2")


class Quad4PlaneKernel:
    """Quad4 plane stress/strain element kernel."""
    type_names = ("Quad4Plane", "Quad4", "CPS4", "CPE4", "CPS4R", "CPE4R")

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
        gauss_order: int = 2,
    ) -> np.ndarray:
        """Return Quad4 plane element stiffness."""
        B_data = self._integration_data(mesh, elem, node_lookup, gauss_order)
        D, t = self._material_data(elem)
        Ke = np.zeros((8, 8), dtype=float)

        for B, detJ, w in B_data:
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
        B = self._B_matrix(mesh, elem, xi, eta, node_lookup)[0]
        Ue = U[mesh.element_dofs(elem)]
        return D @ (B @ Ue)

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

    def _integration_data(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None,
        gauss_order: int,
    ):
        """Return B, detJ, and weight at integration points."""
        if len(elem.node_ids) != 4:
            raise ValueError(f"Quad4 needs 4 nodes, elem {elem.id} node_ids={elem.node_ids}")
        return [
            (*self._B_matrix(mesh, elem, xi, eta, node_lookup), w)
            for xi, eta, w in quad4_gauss_points(gauss_order)
        ]

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

        dN = quad4_shape_grad_xi_eta(xi, eta)
        J = np.array(
            [[np.dot(dN[0], x), np.dot(dN[0], y)],
             [np.dot(dN[1], x), np.dot(dN[1], y)]],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError(f"elem {elem.id} singular Jacobian")

        dN_xy = np.linalg.inv(J) @ dN
        B = np.zeros((3, 8), dtype=float)
        for a_i in range(4):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        return B, detJ

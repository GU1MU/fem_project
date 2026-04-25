from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup
from ..materials import compute_plane_elastic_matrix


class Tri3PlaneKernel:
    """Tri3 plane stress/strain element kernel."""
    type_names = ("Tri3Plane", "Tri3", "CPS3", "CPE3")

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Tri3 plane element stiffness."""
        B, area = self._B_matrix(mesh, elem, node_lookup)
        D, t = self._material_data(elem)
        return t * area * (B.T @ D @ B)

    def stress_at(
        self,
        mesh: Any,
        elem: Any,
        U: np.ndarray,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return constant Tri3 stress."""
        B, _ = self._B_matrix(mesh, elem, node_lookup)
        D, _ = self._material_data(elem)
        return D @ (B @ U[mesh.element_dofs(elem)])

    def _material_data(self, elem: Any):
        """Return D matrix and thickness from element props."""
        try:
            E = float(elem.props["E"])
            nu = float(elem.props["nu"])
            t = float(elem.props["thickness"])
        except KeyError as e:
            raise KeyError(
                f"元素 {elem.id} 的 props 缺少 {e.args[0]}，当前 props={elem.props}"
            )

        pt = str(elem.props.get("plane_type", "stress")).lower()
        D = compute_plane_elastic_matrix(E, nu, pt)
        return D, t

    def _B_matrix(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None,
    ):
        """Return B matrix and area for Tri3."""
        if len(elem.node_ids) != 3:
            raise ValueError(
                f"Tri3 单元必须有 3 个节点，elem {elem.id} node_ids={elem.node_ids}"
            )
        if node_lookup is None:
            node_lookup = build_node_lookup(mesh)

        try:
            n1, n2, n3 = (node_lookup[nid] for nid in elem.node_ids)
        except KeyError as e:
            raise KeyError(f"在 mesh.nodes 中找不到 id={e.args[0]} 的节点")

        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        x3, y3 = n3.x, n3.y

        detJ = (
            x2 * y3 - x3 * y2
            - x1 * y3 + x3 * y1
            + x1 * y2 - x2 * y1
        )
        area = 0.5 * detJ
        if area <= 0.0:
            raise ValueError(
                f"元素 {elem.id} 的面积 A={area} <= 0，请检查节点顺序或是否退化"
            )

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3],
        ], dtype=float)
        return B, area

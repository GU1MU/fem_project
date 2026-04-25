from __future__ import annotations

from typing import Any

import numpy as np

from .base import build_node_lookup


def line2_geometry(mesh: Any, elem: Any, node_lookup: dict[int, Any] | None = None):
    """Return length and direction cosines for a 2-node line element."""
    if len(elem.node_ids) != 2:
        raise ValueError(f"Line2 element must have 2 nodes, elem {elem.id} node_ids={elem.node_ids}")
    if node_lookup is None:
        node_lookup = build_node_lookup(mesh)

    ni = node_lookup[elem.node_ids[0]]
    nj = node_lookup[elem.node_ids[1]]
    dx = nj.x - ni.x
    dy = nj.y - ni.y
    length = (dx**2 + dy**2) ** 0.5
    if length == 0.0:
        raise ValueError(f"Line2 element {elem.id} has zero length")
    return length, dx / length, dy / length


class Truss2DKernel:
    """Two-node planar truss element kernel."""
    type_names = ("Truss2D",)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Truss2D element stiffness."""
        try:
            A = float(elem.props["area"])
            E = float(elem.props["E"])
        except KeyError as e:
            raise KeyError(f"元素 {elem.id} 缺少属性 {e.args[0]}，props={elem.props}")

        L, c, s = line2_geometry(mesh, elem, node_lookup)
        k = E * A / L
        return k * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ], dtype=float)


class Beam2DKernel:
    """Two-node Euler-Bernoulli beam element kernel."""
    type_names = ("Beam2D",)

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return Beam2D element stiffness."""
        try:
            E = float(elem.props["E"])
            A = float(elem.props["area"])
            I = float(elem.props["Izz"])
        except KeyError as e:
            raise KeyError(
                f"元素 {elem.id} 的 props 缺少 {e.args[0]}，当前 props={elem.props}"
            )

        L, c, s = line2_geometry(mesh, elem, node_lookup)
        EA_L = E * A / L
        EI_L3 = E * I / (L**3)
        EI_L2 = E * I / (L**2)
        EI_L = E * I / L

        k_local = np.array([
            [EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
            [0.0, 12 * EI_L3, 6 * EI_L2, 0.0, -12 * EI_L3, 6 * EI_L2],
            [0.0, 6 * EI_L2, 4 * EI_L, 0.0, -6 * EI_L2, 2 * EI_L],
            [-EA_L, 0.0, 0.0, EA_L, 0.0, 0.0],
            [0.0, -12 * EI_L3, -6 * EI_L2, 0.0, 12 * EI_L3, -6 * EI_L2],
            [0.0, 6 * EI_L2, 2 * EI_L, 0.0, -6 * EI_L2, 4 * EI_L],
        ], dtype=float)

        T = np.array([
            [c, -s, 0.0, 0.0, 0.0, 0.0],
            [s, c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, -s, 0.0],
            [0.0, 0.0, 0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        return T.T @ k_local @ T

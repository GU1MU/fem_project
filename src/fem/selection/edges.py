from __future__ import annotations

import builtins
from typing import Any

from .nodes import _coord_matches


def boundary(mesh: Any) -> list[tuple[int, int, list[int]]]:
    """Return boundary edges as (elem_id, local_edge, node_ids)."""
    edge_count: dict[tuple[int, ...], int] = {}
    edge_store: dict[tuple[int, ...], list[tuple[int, int, list[int]]]] = {}

    for elem in mesh.elements:
        for local_edge, node_ids in enumerate(_element_edge_node_ids(elem)):
            key = _edge_key(node_ids)
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_store.setdefault(key, []).append((elem.id, local_edge, node_ids))

    result = []
    for key, count in edge_count.items():
        if count == 1:
            result.extend(edge_store[key])
    return result


def all(mesh: Any) -> list[tuple[int, int, list[int]]]:
    """Return all edges as (elem_id, local_edge, node_ids)."""
    result = []
    for elem in mesh.elements:
        for local_edge, node_ids in enumerate(_element_edge_node_ids(elem)):
            result.append((elem.id, local_edge, node_ids))
    return result


def by_x(
    mesh: Any,
    x_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match x within tol."""
    return by_coord(mesh, x=x_value, tol=tol, boundary_only=boundary_only)


def by_y(
    mesh: Any,
    y_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match y within tol."""
    return by_coord(mesh, y=y_value, tol=tol, boundary_only=boundary_only)


def by_z(
    mesh: Any,
    z_value: float,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match z within tol."""
    return by_coord(mesh, z=z_value, tol=tol, boundary_only=boundary_only)


def by_coord(
    mesh: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tol: float = 1e-8,
    boundary_only: bool = True,
) -> list[tuple[int, int, list[int]]]:
    """Return edges whose all nodes match given coordinates."""
    if x is None and y is None and z is None:
        raise ValueError("at least one coordinate must be provided")
    candidates = boundary(mesh) if boundary_only else all(mesh)
    node_lookup = {node.id: node for node in mesh.nodes}
    return [
        (elem_id, local_edge, node_ids)
        for elem_id, local_edge, node_ids in candidates
        if builtins.all(
            _coord_matches(node_lookup[node_id], x, y, z, tol)
            for node_id in node_ids
        )
    ]


def _element_edge_node_ids(elem: Any) -> list[list[int]]:
    """Return local edge node id lists for common 2D elements."""
    etype = str(elem.type).lower()
    node_ids = elem.node_ids

    if "tri3" in etype and len(node_ids) == 3:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[0]],
        ]

    if "quad4" in etype and len(node_ids) == 4:
        return [
            [node_ids[0], node_ids[1]],
            [node_ids[1], node_ids[2]],
            [node_ids[2], node_ids[3]],
            [node_ids[3], node_ids[0]],
        ]

    if "quad8" in etype and len(node_ids) == 8:
        return [
            [node_ids[0], node_ids[4], node_ids[1]],
            [node_ids[1], node_ids[5], node_ids[2]],
            [node_ids[2], node_ids[6], node_ids[3]],
            [node_ids[3], node_ids[7], node_ids[0]],
        ]

    return []


def _edge_key(node_ids: list[int]) -> tuple[int, ...]:
    """Return topology key for an edge."""
    return tuple(sorted((node_ids[0], node_ids[-1])))

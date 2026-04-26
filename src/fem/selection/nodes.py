from __future__ import annotations

from typing import Any


def by_x(mesh: Any, x_value: float, tol: float = 1e-8) -> list[int]:
    """Return node ids whose x matches target within tol."""
    return [node.id for node in mesh.nodes if abs(node.x - x_value) <= tol]


def by_y(mesh: Any, y_value: float, tol: float = 1e-8) -> list[int]:
    """Return node ids whose y matches target within tol."""
    return [node.id for node in mesh.nodes if abs(node.y - y_value) <= tol]


def by_z(mesh: Any, z_value: float, tol: float = 1e-8) -> list[int]:
    """Return node ids whose z matches target within tol."""
    return [
        node.id for node in mesh.nodes
        if hasattr(node, "z") and abs(node.z - z_value) <= tol
    ]


def by_coord(
    mesh: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tol: float = 1e-8,
) -> list[int]:
    """Return node ids whose given coordinates match targets within tol."""
    if x is None and y is None and z is None:
        raise ValueError("at least one coordinate must be provided")
    return [
        node.id for node in mesh.nodes
        if _coord_matches(node, x, y, z, tol)
    ]


def in_box(
    mesh: Any,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> list[int]:
    """Return node ids inside a bounding box; None bounds mean open."""
    result = []
    for node in mesh.nodes:
        if xmin is not None and node.x < xmin:
            continue
        if xmax is not None and node.x > xmax:
            continue
        if ymin is not None and node.y < ymin:
            continue
        if ymax is not None and node.y > ymax:
            continue
        if zmin is not None and (not hasattr(node, "z") or node.z < zmin):
            continue
        if zmax is not None and (not hasattr(node, "z") or node.z > zmax):
            continue
        result.append(node.id)
    return result


def in_circle(mesh: Any, x: float, y: float, r: float, tol: float = 1e-8) -> list[int]:
    """Return node ids inside a circle in the xy-plane."""
    if r < 0.0:
        raise ValueError("r must be non-negative")
    r_eff = r + tol
    r2 = r_eff * r_eff
    return [
        node.id for node in mesh.nodes
        if (node.x - x) ** 2 + (node.y - y) ** 2 <= r2
    ]


def boundary(mesh: Any, tol: float = 1e-8) -> dict[str, list[int]]:
    """Return node ids on coordinate-extreme boundaries."""
    if not mesh.nodes:
        return {}

    xs = [node.x for node in mesh.nodes]
    ys = [node.y for node in mesh.nodes]
    result = {
        "left": by_x(mesh, min(xs), tol),
        "right": by_x(mesh, max(xs), tol),
        "bottom": by_y(mesh, min(ys), tol),
        "top": by_y(mesh, max(ys), tol),
    }

    if hasattr(mesh.nodes[0], "z"):
        zs = [node.z for node in mesh.nodes]
        result["back"] = by_z(mesh, min(zs), tol)
        result["front"] = by_z(mesh, max(zs), tol)

    return result


def nearest(mesh: Any, x: float, y: float, z: float | None = None) -> int | None:
    """Return the nearest node id to a coordinate."""
    best_id = None
    best_dist2 = None
    for node in mesh.nodes:
        dz = 0.0 if z is None or not hasattr(node, "z") else node.z - z
        dist2 = (node.x - x) ** 2 + (node.y - y) ** 2 + dz ** 2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_id = node.id
    return best_id


def _coord_matches(
    node: Any,
    x: float | None,
    y: float | None,
    z: float | None,
    tol: float,
) -> bool:
    """Return whether a node matches all provided coordinates."""
    if x is not None and abs(node.x - x) > tol:
        return False
    if y is not None and abs(node.y - y) > tol:
        return False
    if z is not None and (not hasattr(node, "z") or abs(node.z - z) > tol):
        return False
    return True

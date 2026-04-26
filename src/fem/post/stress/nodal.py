from __future__ import annotations

import csv
from typing import Dict, Sequence

import numpy as np

from ...mesh import HexMesh3D, Mesh3DProtocol, PlaneMesh2D
from ._common import (
    PLANE_NODAL_HEADER,
    SOLID_NODAL_HEADER,
    element_volume,
    matches,
    nodal_stress,
    node_lookup,
    validated_u,
    write_zero_solid_node,
)
from .invariants import von_mises_3d, von_mises_plane


def by_type(
    type_key: str,
    mesh,
    U: Sequence[float],
    path: str,
    gauss_order: int | None = None,
) -> None:
    """Export nodal stresses by normalized element type key."""
    if type_key == "truss2d":
        raise ValueError("Nodal stress export is not available for Truss2D elements")
    if type_key == "tri3":
        tri3(mesh, U, path)
    elif type_key == "quad4":
        quad4(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "quad8":
        quad8(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "hex8":
        hex8(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "tet4":
        tet4(mesh, U, path)
    elif type_key == "tet10":
        tet10(mesh, U, path)
    else:
        raise ValueError(f"Unsupported stress element type key: {type_key!r}")


def tri3(mesh: PlaneMesh2D, U: Sequence[float], path: str) -> None:
    """Export Tri3 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "tri3")


def quad4(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "quad4", gauss_order)


def quad8(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 nodal stresses averaged from elements."""
    _plane(mesh, U, path, "quad8", gauss_order)


def hex8(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Hex8 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "hex8", gauss_order=gauss_order)


def tet4(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet4 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "tet4", weighted=True)


def tet10(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet10 nodal stresses averaged from connected elements."""
    _solid(mesh, U, path, "tet10", weighted=True)


def _plane(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
) -> None:
    """Export plane nodal stresses averaged from element nodal stresses."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = "stress"
    nu_ref = 0.0

    for elem in mesh.elements:
        if not matches(elem, type_key):
            continue
        node_vals, plane_type, nu_ref = nodal_stress(mesh, elem, U, lookup, gauss_order)
        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PLANE_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                sig_x, sig_y, tau_xy = (sums[nid] / counts[nid]).tolist()
            writer.writerow([
                nid,
                node.x,
                node.y,
                sig_x,
                sig_y,
                tau_xy,
                von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu_ref),
            ])


def _solid(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
    weighted: bool = False,
) -> None:
    """Export solid nodal stresses averaged from connected element nodes."""
    U = validated_u(mesh, U)
    lookup = node_lookup(mesh)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOLID_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = lookup[nid]
            stress_sum = np.zeros(6, dtype=float)
            weight_sum = 0.0
            for elem in mesh.elements:
                if not matches(elem, type_key) or nid not in elem.node_ids:
                    continue
                node_vals = nodal_stress(mesh, elem, U, lookup, gauss_order)
                local_idx = elem.node_ids.index(nid)
                weight = element_volume(mesh, elem, lookup) if weighted else 1.0
                stress_sum += weight * np.asarray(node_vals[local_idx], dtype=float)
                weight_sum += weight

            if weight_sum == 0.0:
                write_zero_solid_node(writer, nid, node)
                continue

            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress_sum / weight_sum
            writer.writerow([
                nid,
                node.x,
                node.y,
                node.z,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])

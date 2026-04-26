from __future__ import annotations

import csv
from typing import Any, Dict, Sequence

import numpy as np

from ...elements import get_element_kernel
from ...mesh import HexMesh3D, Mesh2DProtocol, Mesh3DProtocol, Node3D, PlaneMesh2D, TrussMesh2D
from .invariants import von_mises_3d, von_mises_plane


_TET_CENTROID = (0.25, 0.25, 0.25)
_SOLID_HEADER = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]
_SOLID_NODAL_HEADER = [
    "node_id", "x", "y", "z",
    "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises",
]
_PLANE_ELEMENT_HEADER = ["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"]
_PLANE_NODAL_HEADER = ["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"]


def element(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses to CSV. Element type is inferred for single-type meshes."""
    type_key = _resolve_type_key(mesh, element_type)
    if type_key == "truss2d":
        truss2d_element(mesh, U, path)
    elif type_key == "tri3":
        tri3_plane_element(mesh, U, path)
    elif type_key == "quad4":
        quad4_plane_element(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "quad8":
        quad8_plane_element(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "hex8":
        hex8_element(mesh, U, path)
    elif type_key == "tet4":
        tet4_element(mesh, U, path)
    elif type_key == "tet10":
        tet10_element(mesh, U, path)
    else:
        raise ValueError(f"Unsupported stress element type: {element_type!r}")


def nodal(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export nodal stresses to CSV. Element type is inferred for single-type meshes."""
    type_key = _resolve_type_key(mesh, element_type)
    if type_key == "truss2d":
        raise ValueError("Nodal stress export is not available for Truss2D elements")
    if type_key == "tri3":
        tri3_nodal(mesh, U, path)
    elif type_key == "quad4":
        quad4_nodal(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "quad8":
        quad8_nodal(mesh, U, path, 3 if gauss_order is None else gauss_order)
    elif type_key == "hex8":
        hex8_nodal(mesh, U, path, 2 if gauss_order is None else gauss_order)
    elif type_key == "tet4":
        tet4_nodal(mesh, U, path)
    elif type_key == "tet10":
        tet10_nodal(mesh, U, path)
    else:
        raise ValueError(f"Unsupported stress element type: {element_type!r}")


def truss2d_element(mesh: TrussMesh2D, U: Sequence[float], path: str) -> None:
    """Export Truss2D element axial strain/stress and mises to CSV."""
    U = _validated_u(mesh, U)
    node_lookup = _node_lookup(mesh)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elem_id",
            "node_i",
            "node_j",
            "axial_strain",
            "axial_stress",
            "mises_stress",
        ])
        for elem in mesh.elements:
            ni_id, nj_id = elem.node_ids
            axial_strain, axial_stress, mises_stress = get_element_kernel(
                elem.type
            ).element_stress(mesh, elem, U, node_lookup)
            writer.writerow([
                elem.id,
                ni_id,
                nj_id,
                axial_strain,
                axial_stress,
                mises_stress,
            ])


def tri3_plane_element(mesh: PlaneMesh2D, U: Sequence[float], path: str) -> None:
    """Export Tri3 element-nodal stresses without averaging."""
    _export_plane_element(mesh, U, path, "tri3")


def quad4_plane_element(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 element-nodal stresses without averaging."""
    _export_plane_element(mesh, U, path, "quad4", gauss_order)


def quad8_plane_element(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 element-nodal stresses without averaging."""
    _export_plane_element(mesh, U, path, "quad8", gauss_order)


def tri3_nodal(mesh: PlaneMesh2D, U: Sequence[float], path: str) -> None:
    """Export Tri3 nodal stresses averaged from elements."""
    _export_plane_nodal(mesh, U, path, "tri3")


def quad4_nodal(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 nodal stresses averaged from elements."""
    _export_plane_nodal(mesh, U, path, "quad4", gauss_order)


def quad8_nodal(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 nodal stresses averaged from elements."""
    _export_plane_nodal(mesh, U, path, "quad8", gauss_order)


def hex8_element(mesh: HexMesh3D, U: Sequence[float], path: str) -> None:
    """Export Hex8 centroid stresses to CSV."""
    _export_solid_element(mesh, U, path, "hex8", (0.0, 0.0, 0.0))


def hex8_nodal(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Hex8 nodal stresses averaged from connected elements."""
    _export_solid_nodal(mesh, U, path, "hex8", gauss_order=gauss_order)


def tet4_element(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet4 centroid stresses to CSV."""
    _export_solid_element(mesh, U, path, "tet4", _TET_CENTROID)


def tet4_nodal(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet4 nodal stresses averaged from connected elements."""
    _export_solid_nodal(mesh, U, path, "tet4", weighted=True)


def tet10_element(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet10 centroid stresses to CSV."""
    _export_solid_element(mesh, U, path, "tet10", _TET_CENTROID)


def tet10_nodal(mesh: Mesh3DProtocol, U: Sequence[float], path: str) -> None:
    """Export Tet10 nodal stresses averaged from connected elements."""
    _export_solid_nodal(mesh, U, path, "tet10", weighted=True)


def _export_plane_element(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
) -> None:
    """Export plane element-nodal stresses without averaging."""
    U = _validated_u(mesh, U)
    node_lookup = _node_lookup(mesh)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_PLANE_ELEMENT_HEADER)
        for elem in mesh.elements:
            if not _matches(elem, type_key):
                continue
            node_vals, plane_type, nu = _nodal_stress(mesh, elem, U, node_lookup, gauss_order)
            for local_idx, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[local_idx - 1].tolist()
                writer.writerow([
                    elem.id,
                    nid,
                    local_idx,
                    sig_x,
                    sig_y,
                    tau_xy,
                    von_mises_plane(sig_x, sig_y, tau_xy, plane_type, nu),
                ])


def _export_plane_nodal(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
) -> None:
    """Export plane nodal stresses averaged from element nodal stresses."""
    U = _validated_u(mesh, U)
    node_lookup = _node_lookup(mesh)
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = "stress"
    nu_ref = 0.0

    for elem in mesh.elements:
        if not _matches(elem, type_key):
            continue
        node_vals, plane_type, nu_ref = _nodal_stress(mesh, elem, U, node_lookup, gauss_order)
        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_PLANE_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = node_lookup[nid]
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


def _export_solid_element(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_key: str,
    natural_coords: tuple[float, float, float],
) -> None:
    """Export solid element stresses at one natural coordinate point."""
    U = _validated_u(mesh, U)
    node_lookup = _node_lookup(mesh)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SOLID_HEADER)
        for elem in mesh.elements:
            if not _matches(elem, type_key):
                continue
            stress = get_element_kernel(elem.type).stress_at(
                mesh,
                elem,
                U,
                *natural_coords,
                node_lookup,
            )
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stress
            writer.writerow([
                elem.id,
                sig_x,
                sig_y,
                sig_z,
                tau_xy,
                tau_yz,
                tau_zx,
                von_mises_3d(sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx),
            ])


def _export_solid_nodal(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    type_key: str,
    gauss_order: int | None = None,
    weighted: bool = False,
) -> None:
    """Export solid nodal stresses averaged from connected element nodes."""
    U = _validated_u(mesh, U)
    node_lookup = _node_lookup(mesh)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SOLID_NODAL_HEADER)
        for nid in mesh.node_ids:
            node = node_lookup[nid]
            stress_sum = np.zeros(6, dtype=float)
            weight_sum = 0.0
            for elem in mesh.elements:
                if not _matches(elem, type_key) or nid not in elem.node_ids:
                    continue
                node_vals = _solid_nodal_stress(mesh, elem, U, node_lookup, gauss_order)
                local_idx = elem.node_ids.index(nid)
                weight = _element_volume(mesh, elem, node_lookup) if weighted else 1.0
                stress_sum += weight * np.asarray(node_vals[local_idx], dtype=float)
                weight_sum += weight

            if weight_sum == 0.0:
                _write_zero_solid_node(writer, nid, node)
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


def _nodal_stress(
    mesh: Mesh2DProtocol,
    elem: Any,
    U: np.ndarray,
    node_lookup: dict[int, Any],
    gauss_order: int | None,
):
    """Return element nodal stress through the element kernel."""
    kernel = get_element_kernel(elem.type)
    if gauss_order is None:
        return kernel.nodal_stress(mesh, elem, U, node_lookup)
    return kernel.nodal_stress(mesh, elem, U, node_lookup, gauss_order)


def _solid_nodal_stress(
    mesh: Mesh3DProtocol,
    elem: Any,
    U: np.ndarray,
    node_lookup: dict[int, Any],
    gauss_order: int | None,
) -> np.ndarray:
    """Return solid element nodal stress through the element kernel."""
    kernel = get_element_kernel(elem.type)
    if gauss_order is None:
        return kernel.nodal_stress(mesh, elem, U, node_lookup)
    return kernel.nodal_stress(mesh, elem, U, node_lookup, gauss_order)


def _element_volume(
    mesh: Mesh3DProtocol,
    elem: Any,
    node_lookup: dict[int, Any],
) -> float:
    """Return element volume through the element kernel."""
    return float(get_element_kernel(elem.type).volume(mesh, elem, node_lookup))


def _validated_u(mesh: Any, U: Sequence[float]) -> np.ndarray:
    """Validate and flatten a global displacement vector."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")
    return U


def _node_lookup(mesh: Any) -> dict[int, Any]:
    """Return node lookup keyed by node id."""
    return {node.id: node for node in mesh.nodes}


def _matches(elem: Any, type_key: str) -> bool:
    """Return whether an element type matches a stress exporter key."""
    return type_key in str(elem.type).lower()


def _resolve_type_key(mesh: Any, element_type: str | None) -> str:
    """Resolve a normalized stress exporter key."""
    if element_type is not None:
        type_key = _type_key(element_type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {element_type!r}")
        return type_key

    type_keys = {_type_key(elem.type) for elem in mesh.elements}
    type_keys.discard(None)
    if not type_keys:
        raise ValueError("Cannot infer stress element type from mesh")
    if len(type_keys) > 1:
        raise ValueError("Mixed element meshes require an explicit element_type")
    return type_keys.pop()


def _type_key(element_type: Any) -> str | None:
    """Normalize mesh element type names to stress exporter keys."""
    etype = str(element_type).lower()
    if "truss" in etype:
        return "truss2d"
    if "tri3" in etype:
        return "tri3"
    if "quad4" in etype:
        return "quad4"
    if "quad8" in etype:
        return "quad8"
    if "hex8" in etype:
        return "hex8"
    if "tet10" in etype:
        return "tet10"
    if "tet4" in etype:
        return "tet4"
    return None


def _write_zero_solid_node(writer: csv.writer, nid: int, node: Node3D) -> None:
    """Write a zero stress row for an unconnected solid node."""
    writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])

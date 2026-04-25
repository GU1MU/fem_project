import csv
from typing import Dict, Sequence

import numpy as np

from ..elements import get_element_kernel
from ..mesh import HexMesh3D, Mesh2DProtocol, Mesh3DProtocol, Node3D, PlaneMesh2D, TrussMesh2D


_TET4_CENTROID = (0.25, 0.25, 0.25)


def export_truss2d_element_stress(
    mesh: TrussMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Truss2D element axial strain/stress and mises to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

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


def _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order):
    """Return Quad4 nodal stress by extrapolation."""
    return get_element_kernel(elem.type).nodal_stress(mesh, elem, U, node_lookup, gauss_order)


def _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order):
    """Return Quad8 nodal stress by extrapolation."""
    return get_element_kernel(elem.type).nodal_stress(mesh, elem, U, node_lookup, gauss_order)


def _tri3_element_stress(mesh, elem, U, node_lookup):
    """Return Tri3 constant stress and plane type."""
    node_vals, plane_type, nu = get_element_kernel(elem.type).nodal_stress(mesh, elem, U, node_lookup)
    return node_vals[0], plane_type, nu


def export_tri3_plane_element_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tri3 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            if not str(elem.type).lower().startswith("tri3"):
                continue

            sigma, plane_type, nu = _tri3_element_stress(mesh, elem, U, node_lookup)
            sig_x, sig_y, tau_xy = sigma.tolist()

            if plane_type == "stress":
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
            else:
                sig_z = nu * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )

            for i, nid in enumerate(elem.node_ids, start=1):
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_tri3_nodal_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tri3 nodal stresses averaged from elements."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        if not str(elem.type).lower().startswith("tri3"):
            continue

        sigma, pt, nu = _tri3_element_stress(mesh, elem, U, node_lookup)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for nid in elem.node_ids:
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + sigma
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def export_quad4_plane_element_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            etype = str(elem.type).lower()
            if "quad4" not in etype:
                continue
            if len(elem.node_ids) != 4:
                raise ValueError(f"Quad4 elem must have 4 nodes, elem {elem.id} node_ids={elem.node_ids}")

            node_vals, plane_type, nu = _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)

            for i, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[i - 1].tolist()
                if plane_type == "stress":
                    mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
                else:
                    sig_z = nu * (sig_x + sig_y)
                    mises = np.sqrt(
                        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                        + 3.0 * tau_xy**2
                    )
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_quad8_plane_element_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            etype = str(elem.type).lower()
            if "quad8" not in etype:
                continue
            if len(elem.node_ids) != 8:
                raise ValueError(f"Quad8 elem must have 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

            node_vals, plane_type, nu = _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)

            for i, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[i - 1].tolist()
                if plane_type == "stress":
                    mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
                else:
                    sig_z = nu * (sig_x + sig_y)
                    mises = np.sqrt(
                        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                        + 3.0 * tau_xy**2
                    )
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_quad4_nodal_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 nodal stresses averaged from element extrapolation."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        etype = str(elem.type).lower()
        if "quad4" not in etype:
            continue
        if len(elem.node_ids) != 4:
            raise ValueError(f"Quad4 elem must have 4 nodes, elem {elem.id} node_ids={elem.node_ids}")

        node_vals, pt, nu = _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def export_quad8_nodal_stress(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 nodal stresses averaged from element extrapolation."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        etype = str(elem.type).lower()
        if "quad8" not in etype:
            continue
        if len(elem.node_ids) != 8:
            raise ValueError(f"Quad8 elem must have 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

        node_vals, pt, nu = _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def export_hex8_element_stress(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Hex8 element stresses to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in mesh.elements:
            if elem.type.lower() != "hex8":
                continue

            # Compute element stresses at centroid (xi=0, eta=0, zeta=0)
            stresses = _compute_hex8_element_stress_at_point(mesh, elem, U, node_lookup, 0.0, 0.0, 0.0)
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stresses
            mises = np.sqrt(0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2 +
                                   6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))

            writer.writerow([elem.id, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def _compute_hex8_element_stress_at_point(
    mesh: HexMesh3D,
    elem,
    U: np.ndarray,
    node_lookup: Dict[int, Node3D],
    xi: float,
    eta: float,
    zeta: float,
) -> tuple:
    """Compute stresses at a point in Hex8 element."""
    return get_element_kernel(elem.type).stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)


def export_hex8_nodal_stress(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Hex8 nodal stresses (extrapolated from Gauss points) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["node_id", "x", "y", "z", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node = node_lookup[nid]

            # Find elements connected to this node
            connected_elems = [elem for elem in mesh.elements if nid in elem.node_ids]

            if not connected_elems:
                # Node not connected to any element
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            # Average stresses from connected elements
            stress_sum = np.zeros(6)
            count = 0

            for elem in connected_elems:
                et = str(elem.type).lower()
                if et != "hex8":
                    continue
                node_vals = get_element_kernel(elem.type).nodal_stress(
                    mesh, elem, U, node_lookup, gauss_order
                )
                local_idx = elem.node_ids.index(nid)
                stress_sum += node_vals[local_idx]
                count += 1

            if count == 0:
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            avg_stress = stress_sum / count
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = avg_stress
            mises = np.sqrt(0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2 +
                                   6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))

            writer.writerow([nid, node.x, node.y, node.z, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


# ============================================================
# Tet4 (3D tetrahedral) stress computation and export
# ============================================================

_TET4_CENTROID = (0.25, 0.25, 0.25)


def _tet4_element_volume(elem, node_lookup: Dict[int, Node3D]) -> float:
    """Return Tet4 element volume."""
    return get_element_kernel(elem.type).volume(None, elem, node_lookup)


def _tet10_element_volume(elem, node_lookup: Dict[int, Node3D]) -> float:
    """Return Tet10 element volume using the same 4-point rule as stiffness integration."""
    return get_element_kernel(elem.type).volume(None, elem, node_lookup)


def _compute_tet4_element_stress_at_point(
    mesh: Mesh3DProtocol,
    elem,
    U: np.ndarray,
    node_lookup: Dict[int, Node3D],
    xi: float,
    eta: float,
    zeta: float,
) -> tuple:
    """Compute 3D stresses at a point in a Tet4 element."""
    return get_element_kernel(elem.type).stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)


def export_tet4_element_stress(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tet4 element stresses (at centroid) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in mesh.elements:
            et = str(elem.type).lower()
            if "tet4" not in et:
                continue

            stresses = _compute_tet4_element_stress_at_point(
                mesh, elem, U, node_lookup, *_TET4_CENTROID
            )
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stresses
            mises = np.sqrt(
                0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            )
            writer.writerow([elem.id, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def export_tet4_nodal_stress(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tet4 nodal stresses (averaged from elements) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["node_id", "x", "y", "z", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node = node_lookup[nid]

            connected_elems = [elem for elem in mesh.elements if nid in elem.node_ids]

            if not connected_elems:
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            stress_sum = np.zeros(6, dtype=float)
            weight_sum = 0.0

            for elem in connected_elems:
                et = str(elem.type).lower()
                if "tet4" not in et:
                    continue
                node_vals = get_element_kernel(elem.type).nodal_stress(mesh, elem, U, node_lookup)
                local_idx = elem.node_ids.index(nid)
                stress = node_vals[local_idx]
                weight = _tet4_element_volume(elem, node_lookup)
                stress_sum += weight * np.asarray(stress, dtype=float)
                weight_sum += weight

            if weight_sum == 0.0:
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            avg = stress_sum / weight_sum
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = avg
            mises = np.sqrt(
                0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            )
            writer.writerow([nid, node.x, node.y, node.z, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def _compute_tet10_element_stress_at_point(
    mesh: Mesh3DProtocol,
    elem,
    U: np.ndarray,
    node_lookup: Dict[int, Node3D],
    xi: float,
    eta: float,
    zeta: float,
) -> tuple:
    """Compute 3D stresses at a point in a Tet10 element."""
    return get_element_kernel(elem.type).stress_at(mesh, elem, U, xi, eta, zeta, node_lookup)


def export_tet10_element_stress(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tet10 element stresses (at centroid) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in mesh.elements:
            et = str(elem.type).lower()
            if "tet10" not in et:
                continue

            stresses = _compute_tet10_element_stress_at_point(
                mesh, elem, U, node_lookup, *_TET4_CENTROID
            )
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stresses
            mises = np.sqrt(
                0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            )
            writer.writerow([elem.id, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def export_tet10_nodal_stress(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tet10 nodal stresses (averaged from elements) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["node_id", "x", "y", "z", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node = node_lookup[nid]

            connected_elems = [elem for elem in mesh.elements if nid in elem.node_ids]

            if not connected_elems:
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            stress_sum = np.zeros(6, dtype=float)
            weight_sum = 0.0

            for elem in connected_elems:
                et = str(elem.type).lower()
                if "tet10" not in et:
                    continue
                local_idx = elem.node_ids.index(nid)
                node_vals = get_element_kernel(elem.type).nodal_stress(mesh, elem, U, node_lookup)
                stress = node_vals[local_idx]
                weight = _tet10_element_volume(elem, node_lookup)
                stress_sum += weight * np.asarray(stress, dtype=float)
                weight_sum += weight

            if weight_sum == 0.0:
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            avg = stress_sum / weight_sum
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = avg
            mises = np.sqrt(
                0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            )
            writer.writerow([nid, node.x, node.y, node.z, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])

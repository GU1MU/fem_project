import csv
from typing import Sequence, Optional, List, Dict
import numpy as np
from .elements import get_element_kernel
from .mesh import Mesh2DProtocol, TrussMesh2D, PlaneMesh2D, Node2D, Mesh3DProtocol, HexMesh3D, TetMesh3D, Node3D
from .vtk_export import (
    _build_vtk_cells,
    _convert_element_stress_fields_to_polar,
    _convert_nodal_stress_fields_to_polar,
    _polar_basis,
    _polar_displacement,
    _polar_stress,
    _write_vtk,
    convert_nodal_displacement_to_polar,
    export_vtk_from_csv,
    export_vtk_from_csv_3d,
)


def export_nodal_displacements_csv(
    mesh: Mesh2DProtocol,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 2:
            component_names = ["ux", "uy"]
        elif dofs_per_node == 3:
            component_names = ["ux", "uy", "uz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y"] + component_names

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node: Node2D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y] + disp_vals)


def export_truss2d_element_stress_csv(
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


def export_tri3_plane_element_stress_csv(
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


def export_tri3_nodal_stress_csv(
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


def export_quad4_plane_element_stress_csv(
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


def export_quad8_plane_element_stress_csv(
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


def export_quad4_nodal_stress_csv(
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


def export_quad8_nodal_stress_csv(
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

def extract_path_data(
    mesh: Mesh2DProtocol,
    start_id: int,
    end_id: int,
    points: int,
    target: str,
    path: str = "xydata.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
    normalized: bool = False,
) -> None:
    """Extract path data to CSV."""
    if points < 2:
        raise ValueError("points must be >= 2")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    if start_id not in node_lookup or end_id not in node_lookup:
        raise ValueError("start_id or end_id not in mesh nodes")

    start = np.array([node_lookup[start_id].x, node_lookup[start_id].y], dtype=float)
    end = np.array([node_lookup[end_id].x, node_lookup[end_id].y], dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        raise ValueError("start_id and end_id define zero length path")
    direction = vec / length

    def _read_nodal_fields(csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV requires node_id column, got {reader.fieldnames}")

            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in {"node_id", "x", "y"}
            ]
            data: Dict[int, Dict[str, float]] = {}

            for row in reader:
                nid = int(row["node_id"])
                values: Dict[str, float] = {}
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        values[name] = float(val_str)
                    except ValueError:
                        values[name] = 0.0
                data[nid] = values
            return field_names, data

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(stress_csv_path)

    source_data = None
    if disp_csv_path is not None and target in disp_fields:
        source_data = disp_data
    elif stress_csv_path is not None and target in stress_fields:
        source_data = stress_data

    if source_data is None:
        raise ValueError(f"target {target} not found in provided CSV files")

    candidates = [
        node for node in mesh.nodes
        if node.id in source_data and target in source_data[node.id]
    ]
    if not candidates:
        raise ValueError("no nodes with target data available")

    selected_ids: List[int] = []
    for i in range(points):
        t = i / (points - 1)
        pos = start + t * vec
        best_id = None
        best_dist = None
        for node in candidates:
            dx = node.x - pos[0]
            dy = node.y - pos[1]
            dist2 = dx * dx + dy * dy
            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best_id = node.id
        if best_id is None:
            raise ValueError("failed to select nodes along path")
        selected_ids.append(best_id)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "x", "y", target])

        for nid in selected_ids:
            node = node_lookup[nid]
            val = source_data[nid].get(target)
            if val is None:
                raise ValueError(f"node {nid} missing target {target}")

            proj = np.dot(np.array([node.x, node.y], dtype=float) - start, direction)
            dist = proj / length if normalized else proj
            writer.writerow([dist, node.x, node.y, val])


def extract_circle_data(
    center: Sequence[float],
    radius: float,
    points: int,
    target: str,
    csv_path: str,
    save_path: str,
) -> None:
    """Extract target data on a circle to CSV."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    if points < 2:
        raise ValueError("points must be >= 2")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("CSV requires x and y columns")
        if target not in reader.fieldnames:
            raise ValueError(f"target {target} not found in CSV header")

        rows = []
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                continue
            rows.append((x, y, row))

    if not rows:
        raise ValueError("no valid rows in CSV")

    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", target])

        for theta in angles:
            px = cx + radius * np.cos(theta)
            py = cy + radius * np.sin(theta)

            best_row = None
            best_dist = None
            for x, y, row in rows:
                dx = x - px
                dy = y - py
                dist2 = dx * dx + dy * dy
                if best_dist is None or dist2 < best_dist:
                    best_dist = dist2
                    best_row = row

            if best_row is None:
                continue
            writer.writerow([best_row.get("x", ""), best_row.get("y", ""), best_row.get(target, "")])


def extract_nodes_data(
    mesh: Mesh2DProtocol,
    node_ids: Sequence[int],
    targets: Sequence[str],
    path: str = "nodes_data.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
) -> None:
    """Extract nodal target data to CSV."""
    if not node_ids:
        raise ValueError("node_ids is empty")
    if not targets:
        raise ValueError("targets is empty")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    for nid in node_ids:
        if nid not in node_lookup:
            raise ValueError(f"node_id {nid} not in mesh")

    def _read_nodal_fields(csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV requires node_id column, got {reader.fieldnames}")

            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in {"node_id", "x", "y"}
            ]
            data: Dict[int, Dict[str, float]] = {}

            for row in reader:
                nid = int(row["node_id"])
                values: Dict[str, float] = {}
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        values[name] = float(val_str)
                    except ValueError:
                        values[name] = 0.0
                data[nid] = values
            return field_names, data

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(stress_csv_path)

    target_sources: Dict[str, Dict[int, Dict[str, float]]] = {}
    for target in targets:
        if disp_csv_path is not None and target in disp_fields:
            target_sources[target] = disp_data
        elif stress_csv_path is not None and target in stress_fields:
            target_sources[target] = stress_data
        else:
            raise ValueError(f"target {target} not found in provided CSV files")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y"] + list(targets))

        for nid in node_ids:
            node = node_lookup[nid]
            row = [nid, node.x, node.y]
            for target in targets:
                source = target_sources[target]
                if nid not in source or target not in source[nid]:
                    raise ValueError(f"node {nid} missing target {target}")
                row.append(source[nid][target])
            writer.writerow(row)


def convert_nodal_solution_into_polar_coord(
    csv_path: str,
    center: Sequence[float],
    out_path: str,
) -> None:
    """Convert nodal displacement or stress CSV into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        fields = list(reader.fieldnames)
        has_disp = "ux" in fields and "uy" in fields
        has_stress = "sig_x" in fields and "sig_y" in fields and "tau_xy" in fields

        if has_disp and has_stress:
            raise ValueError("CSV has both displacement and stress columns")
        if not has_disp and not has_stress:
            raise ValueError("CSV missing displacement or stress columns")

        if "x" not in fields or "y" not in fields:
            raise ValueError("CSV requires x and y columns")

        if has_disp:
            mapping = {"ux": "ur", "uy": "ut"}
        else:
            mapping = {"sig_x": "sig_r", "sig_y": "sig_t", "tau_xy": "tau_rt"}

        out_fields = [mapping.get(name, name) for name in fields]
        rows = list(reader)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_fields)

        for row in rows:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                raise ValueError("x or y is not numeric")

            dx = x - cx
            dy = y - cy
            r = (dx * dx + dy * dy) ** 0.5
            if r == 0.0:
                c = 1.0
                s = 0.0
            else:
                c = dx / r
                s = dy / r

            ux_val = uy_val = None
            sx_val = sy_val = txy_val = None
            if has_disp:
                try:
                    ux_val = float(row["ux"])
                    uy_val = float(row["uy"])
                except ValueError:
                    raise ValueError("ux or uy is not numeric")

                ur = c * ux_val + s * uy_val
                ut = -s * ux_val + c * uy_val

            if has_stress:
                try:
                    sx_val = float(row["sig_x"])
                    sy_val = float(row["sig_y"])
                    txy_val = float(row["tau_xy"])
                except ValueError:
                    raise ValueError("sig_x, sig_y, or tau_xy is not numeric")

                sig_r = c * c * sx_val + s * s * sy_val + 2.0 * s * c * txy_val
                sig_t = s * s * sx_val + c * c * sy_val - 2.0 * s * c * txy_val
                tau_rt = -s * c * sx_val + s * c * sy_val + (c * c - s * s) * txy_val

            out_row = []
            for name in fields:
                if has_disp and name == "ux":
                    out_row.append(ur)
                elif has_disp and name == "uy":
                    out_row.append(ut)
                elif has_stress and name == "sig_x":
                    out_row.append(sig_r)
                elif has_stress and name == "sig_y":
                    out_row.append(sig_t)
                elif has_stress and name == "tau_xy":
                    out_row.append(tau_rt)
                else:
                    out_row.append(row.get(name, ""))

            writer.writerow(out_row)


def export_nodal_displacements_csv_3d(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export 3D nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 3:
            component_names = ["ux", "uy", "uz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y", "z"] + component_names

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node: Node3D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y, node.z] + disp_vals)


def export_hex8_element_stress_csv(
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


def export_hex8_nodal_stress_csv(
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


def export_tet4_element_stress_csv(
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


def export_tet4_nodal_stress_csv(
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


def export_tet10_element_stress_csv(
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


def export_tet10_nodal_stress_csv(
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


def export_nodal_displacements_csv_3d_tet(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
) -> None:
    """Export 3D tetrahedral nodal displacements to CSV (alias for generic 3D export)."""
    export_nodal_displacements_csv_3d(mesh, U, path)


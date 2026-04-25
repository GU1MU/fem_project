import csv
from typing import Dict, Optional, Sequence

import numpy as np

from .mesh import Mesh2DProtocol, Mesh3DProtocol


def _build_vtk_cells(mesh, node_id_to_pt_idx: Dict[int, int]):
    """Build VTK connectivity for supported element types."""
    cells = []
    cell_types = []
    elems_for_cell = []
    for elem in mesh.elements:
        etype = str(elem.type).lower()
        vtk_conn = None
        vtk_type = None

        if "truss" in etype or "beam" in etype:
            if len(elem.node_ids) != 2:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [2] + pt_ids
            vtk_type = 3

        elif "tri3" in etype:
            if len(elem.node_ids) != 3:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [3] + pt_ids
            vtk_type = 5

        elif "quad4" in etype:
            if len(elem.node_ids) != 4:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [4] + pt_ids
            vtk_type = 9

        elif "quad8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 23

        elif "tet4" in etype:
            if len(elem.node_ids) != 4:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [4] + pt_ids
            vtk_type = 10

        elif "tet10" in etype:
            if len(elem.node_ids) != 10:
                continue
            # Abaqus C3D10 and VTK quadratic tetrahedra use the same edge order.
            vtk_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            pt_ids = [node_id_to_pt_idx[elem.node_ids[i]] for i in vtk_order]
            vtk_conn = [10] + pt_ids
            vtk_type = 24

        elif "hex8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 12

        else:
            continue

        cells.append(vtk_conn)
        cell_types.append(vtk_type)
        elems_for_cell.append(elem)
    return cells, cell_types, elems_for_cell


def _polar_basis(x: float, y: float, center: Sequence[float]):
    """Return cos/sin of polar basis at (x, y)."""
    dx = x - float(center[0])
    dy = y - float(center[1])
    r = (dx * dx + dy * dy) ** 0.5
    if r == 0.0:
        return 1.0, 0.0
    return dx / r, dy / r


def _polar_displacement(c: float, s: float, ux: float, uy: float):
    """Return (ur, ut) from (ux, uy)."""
    ur = c * ux + s * uy
    ut = -s * ux + c * uy
    return ur, ut


def _polar_stress(c: float, s: float, sig_x: float, sig_y: float, tau_xy: float):
    """Return (sig_r, sig_t, tau_rt) from (sig_x, sig_y, tau_xy)."""
    sig_r = c * c * sig_x + s * s * sig_y + 2.0 * s * c * tau_xy
    sig_t = s * s * sig_x + c * c * sig_y - 2.0 * s * c * tau_xy
    tau_rt = -s * c * sig_x + s * c * sig_y + (c * c - s * s) * tau_xy
    return sig_r, sig_t, tau_rt


def convert_nodal_displacement_to_polar(
    mesh: Mesh2DProtocol,
    node_disp: Dict[int, Dict[str, float]],
    center: Sequence[float],
) -> Dict[int, Dict[str, float]]:
    """Convert nodal displacement dict into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")

    node_lookup = {node.id: node for node in mesh.nodes}
    polar_disp: Dict[int, Dict[str, float]] = {}

    for node in mesh.nodes:
        disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
        c, s = _polar_basis(node.x, node.y, center)
        ur, ut = _polar_displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[node.id] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    for nid, disp in node_disp.items():
        if nid in polar_disp:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        c, s = _polar_basis(node.x, node.y, center)
        ur, ut = _polar_displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[nid] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    return polar_disp


def _convert_nodal_stress_fields_to_polar(
    mesh: Mesh2DProtocol,
    nodal_fields: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert nodal stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(nodal_fields) or polar_names.intersection(nodal_fields):
        return nodal_fields

    node_lookup = {node.id: node for node in mesh.nodes}
    new_fields = {name: vals for name, vals in nodal_fields.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for node in mesh.nodes:
        nid = node.id
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = _polar_basis(node.x, node.y, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    for nid in nodal_fields["sig_x"]:
        if nid in sig_r:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = _polar_basis(node.x, node.y, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields


def _convert_element_stress_fields_to_polar(
    mesh: Mesh2DProtocol,
    field_data: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert element stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(field_data) or polar_names.intersection(field_data):
        return field_data

    node_lookup = {node.id: node for node in mesh.nodes}
    elem_lookup = {elem.id: elem for elem in mesh.elements}

    new_fields = {name: vals for name, vals in field_data.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for eid, elem in elem_lookup.items():
        sx = float(field_data["sig_x"].get(eid, 0.0))
        sy = float(field_data["sig_y"].get(eid, 0.0))
        txy = float(field_data["tau_xy"].get(eid, 0.0))
        xs = [node_lookup[nid].x for nid in elem.node_ids if nid in node_lookup]
        ys = [node_lookup[nid].y for nid in elem.node_ids if nid in node_lookup]
        if not xs or not ys:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        c, s = _polar_basis(cx, cy, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[eid] = sr
        sig_t[eid] = st
        tau_rt[eid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields


def _write_vtk(
    mesh,
    cells,
    cell_types,
    elems_for_cell,
    node_disp,
    field_data,
    vtk_path: str,
    nodal_fields: Optional[Dict[str, Dict[int, float]]] = None,
):
    """Write VTK file from displacement and field dictionaries."""
    nodes = mesh.nodes
    num_points = len(nodes)
    num_cells = len(cells)

    cell_field_arrays: Dict[str, np.ndarray] = {}
    for field_name, field_dict in field_data.items():
        arr = np.zeros(num_cells, dtype=float)
        for cidx, elem in enumerate(elems_for_cell):
            eid = elem.id
            arr[cidx] = float(field_dict.get(eid, 0.0))
        cell_field_arrays[field_name] = arr

    is_3d = len(nodes) > 0 and hasattr(nodes[0], "z")

    with open(vtk_path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM results from CSV\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for node in nodes:
            if is_3d:
                f.write(f"{node.x} {node.y} {node.z}\n")
            else:
                f.write(f"{node.x} {node.y} 0.0\n")

        total_ints = sum(len(conn) for conn in cells)
        f.write(f"\nCELLS {num_cells} {total_ints}\n")
        for conn in cells:
            f.write(" ".join(str(v) for v in conn) + "\n")

        f.write(f"\nCELL_TYPES {num_cells}\n")
        for ct in cell_types:
            f.write(f"{ct}\n")

        f.write(f"\nPOINT_DATA {num_points}\n")
        f.write("VECTORS displacement float\n")
        for node in nodes:
            disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rz": 0.0})
            if is_3d:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} {disp.get('uz', 0.0)}\n")
            else:
                f.write(f"{disp.get('ux', 0.0)} {disp.get('uy', 0.0)} 0.0\n")

        if not is_3d and getattr(mesh, "dofs_per_node", 0) >= 3:
            has_any_rz = any(abs(d.get("rz", 0.0)) > 0.0 for d in node_disp.values())
            if has_any_rz:
                f.write("\nSCALARS rotz float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{node_disp.get(node.id, {}).get('rz', 0.0)}\n")

        if nodal_fields:
            for field_name, field_dict in nodal_fields.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{float(field_dict.get(node.id, 0.0))}\n")

        if cell_field_arrays:
            f.write(f"\nCELL_DATA {num_cells}\n")
            for field_name, arr in cell_field_arrays.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in arr:
                    f.write(f"{val}\n")


def export_vtk_from_csv(
    mesh,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
    polar: bool = False,
    polar_center: Optional[Sequence[float]] = None,
) -> None:
    """Convert displacement + element stress CSV to VTK."""
    node_disp: Dict[int, Dict[str, float]] = {}

    with open(disp_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"node_id", "ux", "uy"}
        if len(mesh.nodes) > 0 and hasattr(mesh.nodes[0], "z"):
            required_cols.add("uz")
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"Disp CSV requires columns {required_cols}, got {reader.fieldnames}")

        has_rz = "rz" in reader.fieldnames
        has_uz = "uz" in reader.fieldnames

        for row in reader:
            nid = int(row["node_id"])
            ux = float(row["ux"])
            uy = float(row["uy"])
            rz = float(row["rz"]) if has_rz and row.get("rz", "") != "" else 0.0
            uz = float(row["uz"]) if has_uz and row.get("uz", "") != "" else 0.0
            node_disp[nid] = {"ux": ux, "uy": uy, "uz": uz, "rz": rz}

    for node in mesh.nodes:
        if node.id not in node_disp:
            node_disp[node.id] = {"ux": 0.0, "uy": 0.0, "rz": 0.0}

    if polar:
        if polar_center is None:
            raise ValueError("export_vtk_from_csv: polar_center required when polar=True")
        node_disp = convert_nodal_displacement_to_polar(mesh, node_disp, polar_center)

    nodal_fields: Dict[str, Dict[int, float]] = {}
    if nodal_stress_csv_path is not None:
        with open(nodal_stress_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"Nodal stress CSV requires 'node_id', got {reader.fieldnames}")

            ignore_exact = {"node_id", "x", "y", "z"}
            field_names = [name for name in (reader.fieldnames or []) if name not in ignore_exact]

            for name in field_names:
                nodal_fields[name] = {}

            for row in reader:
                nid = int(row["node_id"])
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    nodal_fields[name][nid] = val

    if polar and nodal_fields:
        nodal_fields = _convert_nodal_stress_fields_to_polar(mesh, nodal_fields, polar_center)

    field_data: Dict[str, Dict[int, float]] = {}
    if elem_csv_path is not None:
        with open(elem_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "elem_id" not in (reader.fieldnames or []):
                raise ValueError(f"Element stress CSV requires 'elem_id', got {reader.fieldnames}")

            ignore_prefixes = ("node", "nid")
            ignore_exact = {"elem_id", "local_node"}

            stress_field_names = [
                name for name in (reader.fieldnames or [])
                if name not in ignore_exact and not name.startswith(ignore_prefixes)
            ]

            for name in stress_field_names:
                field_data[name] = {}

            for row in reader:
                eid = int(row["elem_id"])
                for name in stress_field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    field_data[name][eid] = val

    if polar and field_data:
        field_data = _convert_element_stress_fields_to_polar(mesh, field_data, polar_center)

    node_id_to_pt_idx: Dict[int, int] = {node.id: i for i, node in enumerate(mesh.nodes)}
    cells, cell_types, elems_for_cell = _build_vtk_cells(mesh, node_id_to_pt_idx)
    if not cells:
        raise ValueError("export_vtk_from_csv: no supported elements")

    _write_vtk(mesh, cells, cell_types, elems_for_cell, node_disp, field_data, vtk_path, nodal_fields)


def export_vtk_from_csv_3d(
    mesh: Mesh3DProtocol,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
) -> None:
    """Backward-compatible wrapper for 3D VTK export."""
    export_vtk_from_csv(
        mesh=mesh,
        disp_csv_path=disp_csv_path,
        elem_csv_path=elem_csv_path,
        vtk_path=vtk_path,
        nodal_stress_csv_path=nodal_stress_csv_path,
    )

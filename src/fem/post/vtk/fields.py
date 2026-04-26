from __future__ import annotations

import csv
from typing import Dict


def read_displacement(mesh, path: str) -> Dict[int, Dict[str, float]]:
    """Read nodal displacement CSV into a node keyed field dict."""
    node_disp: Dict[int, Dict[str, float]] = {}

    with open(path, "r", encoding="utf-8") as f:
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

    return node_disp


def read_nodal_stress(path: str) -> Dict[str, Dict[int, float]]:
    """Read nodal stress CSV into field dictionaries."""
    nodal_fields: Dict[str, Dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
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

    return nodal_fields


def read_element_stress(path: str) -> Dict[str, Dict[int, float]]:
    """Read element stress CSV into field dictionaries."""
    field_data: Dict[str, Dict[int, float]] = {}
    counts: Dict[str, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
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
            counts[name] = {}

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
                field_data[name][eid] = field_data[name].get(eid, 0.0) + val
                counts[name][eid] = counts[name].get(eid, 0) + 1

    for name, values in field_data.items():
        for eid, total in list(values.items()):
            values[eid] = total / counts[name][eid]

    return field_data

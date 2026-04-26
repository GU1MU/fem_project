import csv
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ...core.mesh import Mesh2DProtocol, Mesh3DProtocol, Node2D, Node3D


def _export_nodal_displacement_2d(
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
            component_names = ["ux", "uy", "rz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y"] + component_names

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node: Node2D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y] + disp_vals)


def _export_nodal_displacement_3d(
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

    path = _prepare_path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.node_ids:
            node: Node3D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y, node.z] + disp_vals)


def nodal(
    mesh,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export nodal displacements to CSV. Node coordinates define 2D or 3D output."""
    if mesh.nodes and hasattr(mesh.nodes[0], "z"):
        _export_nodal_displacement_3d(mesh, U, path, component_names)
    else:
        _export_nodal_displacement_2d(mesh, U, path, component_names)


def _prepare_path(path: str | Path) -> Path:
    """Create output parent directory and return a Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

from __future__ import annotations

from typing import Optional, Sequence

from . import cells, fields, polar as polar_fields, writer


def from_csv(
    mesh,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
    polar: bool = False,
    polar_center: Optional[Sequence[float]] = None,
) -> None:
    """Convert displacement and stress CSV files to VTK."""
    node_disp = fields.read_displacement(mesh, disp_csv_path)

    if polar:
        if polar_center is None:
            raise ValueError("from_csv: polar_center required when polar=True")
        node_disp = polar_fields.convert_nodal_displacement(mesh, node_disp, polar_center)

    nodal_fields = {}
    if nodal_stress_csv_path is not None:
        nodal_fields = fields.read_nodal_stress(nodal_stress_csv_path)
    if polar and nodal_fields:
        nodal_fields = polar_fields.convert_nodal_stress_fields(mesh, nodal_fields, polar_center)

    field_data = {}
    if elem_csv_path is not None:
        field_data = fields.read_element_stress(elem_csv_path)
    if polar and field_data:
        field_data = polar_fields.convert_element_stress_fields(mesh, field_data, polar_center)

    vtk_cells, cell_types, elems_for_cell = cells.build(mesh)
    if not vtk_cells:
        raise ValueError("from_csv: no supported elements")

    writer.write(
        mesh=mesh,
        cells=vtk_cells,
        cell_types=cell_types,
        elems_for_cell=elems_for_cell,
        node_disp=node_disp,
        field_data=field_data,
        path=vtk_path,
        nodal_fields=nodal_fields,
    )

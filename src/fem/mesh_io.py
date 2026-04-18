from __future__ import annotations

import csv
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .abaqus import (
    AbaqusInpModel,
    InpBoundarySpec,
    InpCloadSpec,
    InpDloadSpec,
    InpElement,
    InpElementBlock,
    InpMaterial,
    InpModelData,
    InpNode,
    InpSection,
    InpStaticStep,
    InpUnhandledStepSpec,
)
from .abaqus.convert import (
    _build_boundary_from_inp_model as _build_boundary_from_inp_model_impl,
    _build_mesh_from_inp_model as _build_mesh_from_inp_model_impl,
    _read_abaqus_inp_as_model_data as _read_abaqus_inp_as_model_data_impl,
)
from .abaqus import legacy as _abaqus_legacy
from .abaqus.parser import read_abaqus_inp_model as _read_abaqus_inp_model_impl
from .mesh import (
    BeamMesh2D,
    Element2D,
    Element3D,
    HexMesh3D,
    Node2D,
    Node3D,
    PlaneMesh2D,
    TetMesh3D,
    TrussMesh2D,
)


for _abaqus_model_type in (
    AbaqusInpModel,
    InpBoundarySpec,
    InpCloadSpec,
    InpDloadSpec,
    InpElement,
    InpElementBlock,
    InpMaterial,
    InpModelData,
    InpNode,
    InpSection,
    InpStaticStep,
    InpUnhandledStepSpec,
):
    # Keep the public import surface behavior anchored on fem.mesh_io.
    _abaqus_model_type.__module__ = __name__


def read_abaqus_inp_model(inp_path: str) -> AbaqusInpModel:
    """Public INP reader entry point (mesh_io-owned wrapper)."""

    return _read_abaqus_inp_model_impl(inp_path)


def build_mesh_from_inp_model(model: AbaqusInpModel) -> Any:
    """Public mesh_io wrapper for internal Abaqus mesh conversion."""

    return _build_mesh_from_inp_model_impl(model)


def build_boundary_from_inp_model(
    model: AbaqusInpModel,
    mesh: Any,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> Any:
    """Public mesh_io wrapper for internal Abaqus boundary conversion."""

    return _build_boundary_from_inp_model_impl(
        model,
        mesh,
        step_name=step_name,
        step_index=step_index,
    )


def read_abaqus_inp_as_model_data(
    inp_path: str,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> InpModelData:
    """Public mesh_io wrapper for internal Abaqus model-data conversion."""

    return _read_abaqus_inp_as_model_data_impl(
        inp_path,
        step_name=step_name,
        step_index=step_index,
    )


def read_materials_as_dict(path: str) -> Dict[int, Dict[str, str]]:
    """Read material CSV into a dict keyed by material_id."""
    materials: Dict[int, Dict[str, str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row:
                continue

            mid_raw = row.get("material_id")
            if mid_raw is None or mid_raw.strip() == "":
                continue

            mid = int(mid_raw)
            materials[mid] = {k: (v.strip() if v is not None else "") for k, v in row.items()}

    return materials


def _get_float_from_material(
    mat_row: Dict[str, str],
    keys: List[str],
) -> Optional[float]:
    
    # 做一个 key.lower() -> 原始 key 的映射，方便大小写不敏感
    lower_map = {k.lower(): k for k in mat_row.keys()}

    for key in keys:
        kl = key.lower()
        if kl in lower_map:
            raw = mat_row[lower_map[kl]]
            if raw == "":
                continue
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def read_truss2d_csv(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TrussMesh2D:
    """Read a Truss2D mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)

    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 节点表头
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 单元表头
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                if len(row) < 5:
                    raise ValueError(f"第 {line_no} 行单元格式错误: {row!r}")
                eid = int(row[0])
                ni = int(row[1])
                nj = int(row[2])
                area = float(row[3])
                mid = int(row[4])

                props: Dict[str, object] = {
                    "area": area,
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[ni, nj],
                        type="Truss2D",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return TrussMesh2D(nodes=nodes, elements=elements)


def read_beam2d_csv(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> BeamMesh2D:
    """Read a Beam2D mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)

    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 表头：节点
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 表头：单元
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                # elem_id,node_i,node_j,area,Izz,material_id
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Beam 单元格式错误: {row!r}")
                eid = int(row[0])
                ni = int(row[1])
                nj = int(row[2])
                area = float(row[3])
                Izz = float(row[4])
                mid = int(row[5])

                props: Dict[str, object] = {
                    "area": area,        # A
                    "Izz": Izz,          # I
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[ni, nj],
                        type="Beam2D",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("beam mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("beam mesh csv 中没有读到单元")

    return BeamMesh2D(nodes=nodes, elements=elements)


def read_tri3_2d_csv(
    mesh_path: str,
    material_path: Optional[str] = None,
    plane_type: str = "stress",   
) -> PlaneMesh2D:
    """Read a Tri3 plane mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        from .mesh_io import read_materials_as_dict, _get_float_from_material
        materials_dict = read_materials_as_dict(material_path)

    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    mode: Optional[str] = None  

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            if row[0] == "node_id":
                mode = "nodes"
                continue

            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 3:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                nodes.append(Node2D(id=nid, x=x, y=y))

            elif mode == "elements":
                # elem_id,node1,node2,node3,thickness,material_id
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Tri3 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                thickness = float(row[4])
                mid = int(row[5])

                props: Dict[str, object] = {
                    "thickness": thickness,
                    "material_id": mid,
                    "plane_type": plane_type,  
                }

                if materials_dict:
                    from .mesh_io import _get_float_from_material

                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        E_val = _get_float_from_material(mat_row, ["E"])
                        nu_val = _get_float_from_material(mat_row, ["nu"])
                        rho_val = _get_float_from_material(mat_row, ["rho"])

                        if E_val is None or nu_val is None:
                            raise KeyError(
                                f"材料 {mid} 未找到 E/nu 信息，mat_row={mat_row}"
                            )

                        props["E"] = E_val
                        props["nu"] = nu_val
                        if rho_val is not None:
                            props["rho"] = rho_val

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[n1, n2, n3],
                        type="Tri3Plane",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("plane tri3 mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("plane tri3 mesh csv 中没有读到单元")

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_tri3_2d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
) -> PlaneMesh2D:
    """Read a Tri3 plane mesh from Abaqus .inp files."""

    return _abaqus_legacy._read_tri3_2d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        default_thickness=default_thickness,
        plane_type=plane_type,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_quad4_2d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
    enforce_parallelogram: bool = False,
    tol: float = 1e-10,
) -> PlaneMesh2D:
    """Read Quad4 plane mesh (CPS4/CPE4) from Abaqus INP file."""

    return _abaqus_legacy._read_quad4_2d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        default_thickness=default_thickness,
        plane_type=plane_type,
        fix_orientation=fix_orientation,
        enforce_parallelogram=enforce_parallelogram,
        tol=tol,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_quad8_2d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> PlaneMesh2D:
    """Read Quad8 plane mesh (CPS8/CPE8) from Abaqus INP file."""

    return _abaqus_legacy._read_quad8_2d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        default_thickness=default_thickness,
        plane_type=plane_type,
        fix_orientation=fix_orientation,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_tet10_3d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
) -> TetMesh3D:
    """Read a Tet10 3D mesh from Abaqus .inp file (C3D10 elements).

    Node ordering (Abaqus convention):
        Corner nodes:  1-4
        Edge midnodes: 5=edge(1,2), 6=edge(3,4), 7=edge(1,4),
                       8=edge(1,3), 9=edge(2,4), 10=edge(2,3)
    """

    return _abaqus_legacy._read_tet10_3d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_hex8_csv(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a Hex8 mesh CSV with optional materials."""

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            # 节点表头
            if row[0] == "node_id":
                mode = "nodes"
                continue

            # 单元表头
            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                nodes.append(Node3D(id=nid, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 10:
                    raise ValueError(f"第 {line_no} 行 Hex8 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                n4 = int(row[4])
                n5 = int(row[5])
                n6 = int(row[6])
                n7 = int(row[7])
                n8 = int(row[8])
                mid = int(row[9])

                props: Dict[str, object] = {
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_nu = _get_float_from_material(mat_row, ["nu", "poisson"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_nu is not None:
                            props["nu"] = raw_nu
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4, n5, n6, n7, n8],
                        type="Hex8",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return HexMesh3D(nodes=nodes, elements=elements)


def read_tet4_3d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
) -> TetMesh3D:
    """Read a Tet4 3D mesh from Abaqus .inp file (C3D4 / C3D4T elements)."""

    return _abaqus_legacy._read_tet4_3d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_hex8_3d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a Hex8 3D mesh from Abaqus .inp file (C3D8 elements)."""

    return _abaqus_legacy._read_hex8_3d_abaqus(
        inp_path=inp_path,
        material_id=material_id,
        material_path=material_path,
        read_materials_as_dict=read_materials_as_dict,
        get_float_from_material=_get_float_from_material,
    )

def read_tet4_csv(
    mesh_path: str,
    material_path: Optional[str] = None,
) -> TetMesh3D:
    """Read a Tet4 mesh CSV with optional materials."""
    import numpy as np

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)

    nodes: List[Node3D] = []
    elements: List[Element3D] = []

    mode: Optional[str] = None

    with open(mesh_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, start=1):
            row = [col.strip() for col in row]

            if not row or all(col == "" for col in row):
                continue

            if row[0].startswith("#"):
                continue

            if row[0] == "node_id":
                mode = "nodes"
                continue

            if row[0] == "elem_id":
                mode = "elements"
                continue

            if mode == "nodes":
                if len(row) < 4:
                    raise ValueError(f"第 {line_no} 行节点格式错误: {row!r}")
                nid = int(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                nodes.append(Node3D(id=nid, x=x, y=y, z=z))

            elif mode == "elements":
                if len(row) < 6:
                    raise ValueError(f"第 {line_no} 行 Tet4 单元格式错误: {row!r}")
                eid = int(row[0])
                n1 = int(row[1])
                n2 = int(row[2])
                n3 = int(row[3])
                n4 = int(row[4])
                mid = int(row[5])

                props: Dict[str, object] = {
                    "material_id": mid,
                }

                if materials_dict:
                    mat_row = materials_dict.get(mid)
                    if mat_row is not None:
                        raw_E = _get_float_from_material(mat_row, ["E"])
                        raw_nu = _get_float_from_material(mat_row, ["nu", "poisson"])
                        raw_rho = _get_float_from_material(mat_row, ["rho"])
                        if raw_E is not None:
                            props["E"] = raw_E
                        if raw_nu is not None:
                            props["nu"] = raw_nu
                        if raw_rho is not None:
                            props["rho"] = raw_rho

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Tet4",
                        props=props,
                    )
                )

            else:
                raise ValueError(
                    f"在未识别出表头前遇到数据行（第 {line_no} 行）: {row!r}"
                )

    if not nodes:
        raise ValueError("mesh csv 中没有读到节点")
    if not elements:
        raise ValueError("mesh csv 中没有读到单元")

    return TetMesh3D(nodes=nodes, elements=elements)

from __future__ import  annotations

import csv
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple
from .mesh import Node2D, Element2D, TrussMesh2D, BeamMesh2D, PlaneMesh2D, Node3D, Element3D, HexMesh3D, TetMesh3D


@dataclass(frozen=True)
class InpNode:
    id: int
    coordinates: Tuple[float, ...]


@dataclass(frozen=True)
class InpElement:
    id: int
    node_ids: Tuple[int, ...]


@dataclass
class InpElementBlock:
    element_type: str
    elset: Optional[str] = None
    elements: List[InpElement] = field(default_factory=list)


@dataclass
class InpMaterial:
    name: str
    elastic: Optional[Tuple[float, ...]] = None
    density: Optional[float] = None


@dataclass(frozen=True)
class InpSection:
    section_type: str
    elset: Optional[str]
    material_name: Optional[str]
    parameters: Dict[str, str]
    data: List[Tuple[str, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class InpBoundarySpec:
    target: str
    boundary_type: Optional[str] = None
    first_dof: Optional[int] = None
    last_dof: Optional[int] = None
    value: Optional[float] = None
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InpCloadSpec:
    target: str
    dof: int
    magnitude: float
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InpDloadSpec:
    target: str
    load_type: str
    magnitude: float
    components: Tuple[float, ...] = ()
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class InpUnhandledStepSpec:
    keyword: str
    parameters: Dict[str, str]
    data_lines: List[Tuple[str, ...]] = field(default_factory=list)


@dataclass
class InpStaticStep:
    name: Optional[str]
    static_parameters: Tuple[float, ...] = ()
    boundary_specs: List[InpBoundarySpec] = field(default_factory=list)
    cload_specs: List[InpCloadSpec] = field(default_factory=list)
    dload_specs: List[InpDloadSpec] = field(default_factory=list)
    unhandled_specs: List[InpUnhandledStepSpec] = field(default_factory=list)


@dataclass
class AbaqusInpModel:
    part_name: Optional[str] = None
    instance_name: Optional[str] = None
    nodes: Dict[int, InpNode] = field(default_factory=dict)
    element_blocks: List[InpElementBlock] = field(default_factory=list)
    nsets: Dict[str, Dict[str, List[int]]] = field(
        default_factory=lambda: {"part": {}, "assembly": {}}
    )
    elsets: Dict[str, Dict[str, List[int]]] = field(
        default_factory=lambda: {"part": {}, "assembly": {}}
    )
    materials: Dict[str, InpMaterial] = field(default_factory=dict)
    sections: List[InpSection] = field(default_factory=list)
    steps: List[InpStaticStep] = field(default_factory=list)


def _parse_inp_keyword(line: str) -> Tuple[str, Dict[str, str]]:
    parts = [part.strip() for part in line.strip()[1:].split(",")]
    keyword = parts[0].upper()
    params: Dict[str, str] = {}

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip().lower()] = value.strip()

    return keyword, params


def _split_inp_data_line(line: str) -> List[str]:
    return [part.strip() for part in line.split(",")]


def _extend_named_id_set(
    target: Dict[str, Dict[str, List[int]]],
    scope: str,
    name: Optional[str],
    values: List[int],
) -> None:
    if name is None:
        return
    target.setdefault(scope, {}).setdefault(name, []).extend(values)


def _require_non_empty_fields(
    parts: List[str],
    block_name: str,
    expected_min_fields: int,
) -> List[str]:
    if len(parts) < expected_min_fields or any(part == "" for part in parts):
        raise ValueError(f"Malformed {block_name} data: {parts!r}")
    return parts


def _trim_trailing_empty_fields(parts: List[str]) -> List[str]:
    trimmed = list(parts)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def _parse_set_values(parts: List[str], block_name: str, is_generate: bool) -> List[int]:
    normalized_parts = _trim_trailing_empty_fields(parts)

    if is_generate:
        fields = _require_non_empty_fields(normalized_parts, block_name, 3)
        if len(fields) != 3:
            raise ValueError(f"Malformed {block_name} generate data: {parts!r}")

        start, end, increment = (int(value) for value in fields)
        if increment == 0:
            raise ValueError(f"Malformed {block_name} generate data: {parts!r}")
        if (end - start) * increment < 0:
            raise ValueError(f"Malformed {block_name} generate data: {parts!r}")

        return list(range(start, end + (1 if increment > 0 else -1), increment))

    if any(part == "" for part in normalized_parts):
        raise ValueError(f"Malformed {block_name} data: {parts!r}")

    return [int(value) for value in normalized_parts]


def _parse_boundary_spec(parts: List[str], parameters: Dict[str, str]) -> InpBoundarySpec:
    fields = _trim_trailing_empty_fields(parts)
    fields = _require_non_empty_fields(fields, "*Boundary", 2)

    target = fields[0]
    second_field = fields[1]

    try:
        first_dof = int(second_field)
    except ValueError:
        if len(fields) != 2:
            raise ValueError(f"Malformed *Boundary data: {parts!r}")
        return InpBoundarySpec(
            target=target,
            parameters=dict(parameters),
            boundary_type=second_field.upper(),
        )

    last_dof = first_dof
    value: Optional[float] = None

    if len(fields) >= 3:
        try:
            last_dof = int(fields[2])
            if len(fields) >= 4:
                value = float(fields[3])
        except ValueError:
            last_dof = first_dof
            value = float(fields[2])
            if len(fields) >= 4:
                raise ValueError(f"Malformed *Boundary data: {parts!r}")

    if len(fields) > 4:
        raise ValueError(f"Malformed *Boundary data: {parts!r}")

    return InpBoundarySpec(
        target=target,
        parameters=dict(parameters),
        first_dof=first_dof,
        last_dof=last_dof,
        value=value,
    )


def _parse_cload_spec(parts: List[str], parameters: Dict[str, str]) -> InpCloadSpec:
    fields = _require_non_empty_fields(
        _trim_trailing_empty_fields(parts),
        "*Cload",
        3,
    )
    if len(fields) != 3:
        raise ValueError(f"Malformed *Cload data: {parts!r}")

    return InpCloadSpec(
        target=fields[0],
        parameters=dict(parameters),
        dof=int(fields[1]),
        magnitude=float(fields[2]),
    )


def _parse_dload_spec(parts: List[str], parameters: Dict[str, str]) -> InpDloadSpec:
    fields = _require_non_empty_fields(
        _trim_trailing_empty_fields(parts),
        "*Dload",
        3,
    )

    return InpDloadSpec(
        target=fields[0],
        parameters=dict(parameters),
        load_type=fields[1].upper(),
        magnitude=float(fields[2]),
        components=tuple(float(value) for value in fields[3:]),
    )


def read_abaqus_inp_model(inp_path: str) -> AbaqusInpModel:
    """Read a low-level Abaqus INP model description."""

    model = AbaqusInpModel()

    current_block: Optional[str] = None
    current_element_block: Optional[InpElementBlock] = None
    current_nset: Optional[str] = None
    current_elset: Optional[str] = None
    current_material: Optional[InpMaterial] = None
    current_step_name: Optional[str] = None
    current_step: Optional[InpStaticStep] = None
    in_step = False
    current_section: Optional[InpSection] = None
    current_unhandled_step_spec: Optional[InpUnhandledStepSpec] = None
    current_block_params: Dict[str, str] = {}
    current_scope = "part"
    current_nset_generate = False
    current_elset_generate = False

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                keyword, params = _parse_inp_keyword(line)
                current_block = None
                current_unhandled_step_spec = None
                current_block_params = dict(params)

                if keyword == "PART":
                    current_scope = "part"
                    model.part_name = params.get("name")
                elif keyword == "END PART":
                    current_scope = "assembly"
                    current_nset = None
                    current_elset = None
                elif keyword == "NODE":
                    current_block = "node"
                elif keyword == "ELEMENT":
                    current_element_block = InpElementBlock(
                        element_type=params.get("type", ""),
                        elset=params.get("elset"),
                    )
                    model.element_blocks.append(current_element_block)
                    current_block = "element"
                elif keyword == "NSET":
                    current_nset = params.get("nset")
                    current_nset_generate = "generate" in params or any(
                        part.strip().lower() == "generate"
                        for part in line.strip()[1:].split(",")[1:]
                    )
                    current_block = "nset"
                elif keyword == "ELSET":
                    current_elset = params.get("elset")
                    current_elset_generate = "generate" in params or any(
                        part.strip().lower() == "generate"
                        for part in line.strip()[1:].split(",")[1:]
                    )
                    current_block = "elset"
                elif keyword == "SOLID SECTION":
                    current_section = InpSection(
                        section_type=keyword,
                        elset=params.get("elset"),
                        material_name=params.get("material"),
                        parameters=dict(params),
                    )
                    model.sections.append(current_section)
                    current_block = "section"
                elif keyword == "INSTANCE":
                    model.instance_name = params.get("name")
                elif keyword == "ASSEMBLY":
                    current_scope = "assembly"
                elif keyword == "MATERIAL":
                    material_name = params.get("name")
                    if material_name is not None:
                        current_material = model.materials.setdefault(
                            material_name,
                            InpMaterial(name=material_name),
                        )
                elif keyword == "ELASTIC":
                    current_block = "elastic"
                elif keyword == "DENSITY":
                    current_block = "density"
                elif keyword == "STEP":
                    current_step_name = params.get("name")
                    in_step = True
                    current_step = None
                elif keyword == "STATIC":
                    if not in_step:
                        raise ValueError("*Static outside *Step is invalid")
                    if current_step is None:
                        current_step = InpStaticStep(name=current_step_name)
                        model.steps.append(current_step)
                    current_block = "static"
                elif keyword == "BOUNDARY":
                    if in_step and current_step is None:
                        raise ValueError(
                            "*Boundary encountered before supported step procedure"
                        )
                    if current_step is not None:
                        current_block = "boundary"
                elif keyword == "CLOAD":
                    if in_step and current_step is None:
                        raise ValueError(
                            "*Cload encountered before supported step procedure"
                        )
                    if current_step is not None:
                        current_block = "cload"
                elif keyword == "DLOAD":
                    if in_step and current_step is None:
                        raise ValueError(
                            "*Dload encountered before supported step procedure"
                        )
                    if current_step is not None:
                        current_block = "dload"
                elif keyword == "END STEP":
                    current_step_name = None
                    current_step = None
                    in_step = False
                elif keyword in {"SURFACE", "COUPLING", "KINEMATIC"} and current_step is not None:
                    current_unhandled_step_spec = InpUnhandledStepSpec(
                        keyword=keyword,
                        parameters=dict(params),
                    )
                    current_step.unhandled_specs.append(current_unhandled_step_spec)
                    current_block = "unhandled_step"

                continue

            parts = _split_inp_data_line(line)
            if not parts:
                continue

            if current_block == "node":
                fields = _require_non_empty_fields(parts, "*Node", 2)
                node_id = int(fields[0])
                coordinates = tuple(float(value) for value in fields[1:])
                model.nodes[node_id] = InpNode(id=node_id, coordinates=coordinates)
            elif current_block == "element" and current_element_block is not None:
                fields = _require_non_empty_fields(parts, "*Element", 2)
                element_id = int(fields[0])
                node_ids = tuple(int(value) for value in fields[1:])
                current_element_block.elements.append(
                    InpElement(id=element_id, node_ids=node_ids)
                )
            elif current_block == "nset":
                _extend_named_id_set(
                    model.nsets,
                    current_scope,
                    current_nset,
                    _parse_set_values(parts, "*Nset", current_nset_generate),
                )
            elif current_block == "elset":
                _extend_named_id_set(
                    model.elsets,
                    current_scope,
                    current_elset,
                    _parse_set_values(parts, "*Elset", current_elset_generate),
                )
            elif current_block == "elastic" and current_material is not None:
                fields = [part for part in parts if part != ""]
                current_material.elastic = tuple(float(value) for value in fields)
            elif current_block == "density" and current_material is not None:
                fields = _require_non_empty_fields(
                    _trim_trailing_empty_fields(parts),
                    "*Density",
                    1,
                )
                current_material.density = float(fields[0])
            elif current_block == "section" and current_section is not None:
                section_parts = _trim_trailing_empty_fields(parts)
                current_section.data.append(tuple(section_parts))
            elif current_block == "static" and current_step is not None:
                fields = [part for part in parts if part != ""]
                current_step.static_parameters = tuple(float(value) for value in fields)
                current_block = None
            elif current_block == "boundary" and current_step is not None:
                current_step.boundary_specs.append(
                    _parse_boundary_spec(parts, current_block_params)
                )
            elif current_block == "cload" and current_step is not None:
                current_step.cload_specs.append(
                    _parse_cload_spec(parts, current_block_params)
                )
            elif current_block == "dload" and current_step is not None:
                current_step.dload_specs.append(
                    _parse_dload_spec(parts, current_block_params)
                )
            elif current_block == "unhandled_step" and current_unhandled_step_spec is not None:
                current_unhandled_step_spec.data_lines.append(
                    tuple(_trim_trailing_empty_fields(parts))
                )

    return model


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

    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)
        if material_id not in materials_dict:
            raise KeyError(f"material_id={material_id} 不在材料表中（material_path={material_path}）")

    nodes: List[Node2D] = []
    elements: List[Element2D] = []

    node_lookup: Dict[int, Node2D] = {}

    in_node_block = False
    in_elem_block = False
    elem_abaqus_type: Optional[str] = None  # "CPS3" or "CPE3"

    def _parse_keyword(line: str) -> str:
        return line.strip()

    def _parse_csv_like_numbers(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        parts = [p for p in parts if p != ""]
        return parts

    def _infer_plane_type_from_elem_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS3"):
            return "stress"
        if etu.startswith("CPE3"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "" or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = _parse_keyword(line).upper()

                in_node_block = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                    if "TYPE=" in kw:
                        parts = [p.strip() for p in kw.split(",")]
                        etype = None
                        for p in parts:
                            if p.startswith("TYPE="):
                                etype = p.split("=", 1)[1].strip()
                                break
                        if etype in ("CPS3", "CPE3"):
                            in_elem_block = True
                            elem_abaqus_type = etype
                    continue

                if not in_node_block:
                    pass
                if not kw.startswith("*ELEMENT"):
                    in_elem_block = False
                    elem_abaqus_type = None

                continue  

            # 数据行
            if in_node_block:
                parts = _parse_csv_like_numbers(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node = Node2D(id=nid, x=x, y=y)
                nodes.append(node)
                node_lookup[nid] = node

            elif in_elem_block and elem_abaqus_type is not None:
                parts = _parse_csv_like_numbers(line)
                if len(parts) < 4:
                    continue
                eid = int(parts[0])
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])

                if plane_type is None:
                    pt = _infer_plane_type_from_elem_type(elem_abaqus_type)
                else:
                    pt = str(plane_type).lower()
                    if pt.startswith("stress"):
                        pt = "stress"
                    elif pt.startswith("strain"):
                        pt = "strain"
                    else:
                        raise ValueError("plane_type 必须是 'stress' 或 'strain'")

                props: Dict[str, any] = {
                    "material_id": int(material_id),
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                if materials_dict:
                    mat_row = materials_dict[int(material_id)]
                    E_val = _get_float_from_material(mat_row, ["E", "young", "youngs_modulus"])
                    nu_val = _get_float_from_material(mat_row, ["nu", "poisson", "poisson_ratio"])
                    rho_val = _get_float_from_material(mat_row, ["rho", "rou", "density"])

                    if E_val is None or nu_val is None:
                        raise KeyError(
                            f"材料 {material_id} 缺少 E/nu 信息，row={mat_row}"
                        )
                    props["E"] = E_val
                    props["nu"] = nu_val
                    if rho_val is not None:
                        props["rho"] = rho_val  

                elem = Element2D(
                    id=eid,
                    node_ids=[n1, n2, n3],
                    type="Tri3Plane",
                    props=props,
                )
                elements.append(elem)

    if not nodes:
        raise ValueError(f"未在 {inp_path} 中解析到 *Node 数据")
    if not elements:
        raise ValueError(f"未在 {inp_path} 中解析到 CPS3/CPE3 的 *Element 数据")

    return PlaneMesh2D(nodes=nodes, elements=elements)


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
    materials: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials = read_materials_as_dict(material_path)
        if material_id not in materials:
            raise KeyError(f"material_id={material_id} not found in {material_path}")

    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS4"):
            return "stress"
        if etu.startswith("CPE4"):
            return "strain"
        return "stress"

    def signed_area_quad(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        return 0.5 * (
            x1 * y2 - x2 * y1 +
            x2 * y3 - x3 * y2 +
            x3 * y4 - x4 * y3 +
            x4 * y1 - x1 * y4
        )

    def is_parallelogram(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        d1 = (x1 + x3 - x2 - x4, y1 + y3 - y2 - y4)  # diag midpoints: p1+p3 == p2+p4
        return (d1[0] * d1[0] + d1[1] * d1[1]) <= tol

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("CPS4", "CPE4", "CPS4R", "CPE4R"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                n = Node2D(id=nid, x=x, y=y)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 5:
                    continue
                eid = int(parts[0])
                n1, n2, n3, n4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError("plane_type must be 'stress' or 'strain'")

                props: Dict[str, any] = {
                    "material_id": int(material_id),
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                if materials:
                    row = materials[int(material_id)]
                    E = _get_float_from_material(row, ["E", "young", "youngs_modulus"])
                    nu = _get_float_from_material(row, ["nu", "poisson", "poisson_ratio"])
                    rho = _get_float_from_material(row, ["rho", "rou", "density"])
                    if E is None or nu is None:
                        raise KeyError(f"Material {material_id} missing E/nu: {row}")
                    props["E"] = E
                    props["nu"] = nu
                    if rho is not None:
                        props["rho"] = rho

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Quad4Plane",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS4/CPE4 *Element data found in {inp_path}")

    if enforce_parallelogram or fix_orientation:
        for e in elements:
            n1, n2, n3, n4 = e.node_ids
            try:
                p1 = (node_lookup[n1].x, node_lookup[n1].y)
                p2 = (node_lookup[n2].x, node_lookup[n2].y)
                p3 = (node_lookup[n3].x, node_lookup[n3].y)
                p4 = (node_lookup[n4].x, node_lookup[n4].y)
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")

            if enforce_parallelogram and not is_parallelogram(p1, p2, p3, p4):
                raise ValueError(f"Element {e.id} is not a parallelogram by tolerance {tol}")

            if fix_orientation:
                A = signed_area_quad(p1, p2, p3, p4)
                if A < 0.0:
                    e.node_ids = [n1, n4, n3, n2]

    return PlaneMesh2D(nodes=nodes, elements=elements)


def read_quad8_2d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
    default_thickness: float = 1.0,
    plane_type: Optional[str] = None,
    fix_orientation: bool = True,
) -> PlaneMesh2D:
    """Read Quad8 plane mesh (CPS8/CPE8) from Abaqus INP file."""
    materials: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials = read_materials_as_dict(material_path)
        if material_id not in materials:
            raise KeyError(f"material_id={material_id} not found in {material_path}")

    nodes: List[Node2D] = []
    elements: List[Element2D] = []
    node_lookup: Dict[int, Node2D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    def infer_plane_type(et: str) -> str:
        etu = et.upper()
        if etu.startswith("CPS8"):
            return "stress"
        if etu.startswith("CPE8"):
            return "strain"
        return "stress"

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("CPS8", "CPE8", "CPS8R", "CPE8R"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 3:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                n = Node2D(id=nid, x=x, y=y)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 9:
                    continue
                eid = int(parts[0])
                nids = [int(p) for p in parts[1:9]]

                pt = infer_plane_type(elem_abaqus_type) if plane_type is None else str(plane_type).lower()
                if pt.startswith("stress"):
                    pt = "stress"
                elif pt.startswith("strain"):
                    pt = "strain"
                else:
                    raise ValueError("plane_type must be 'stress' or 'strain'")

                props: Dict[str, any] = {
                    "material_id": int(material_id),
                    "thickness": float(default_thickness),
                    "plane_type": pt,
                }

                if materials:
                    row = materials[int(material_id)]
                    E = _get_float_from_material(row, ["E", "young", "youngs_modulus"])
                    nu = _get_float_from_material(row, ["nu", "poisson", "poisson_ratio"])
                    rho = _get_float_from_material(row, ["rho", "rou", "density"])
                    if E is None or nu is None:
                        raise KeyError(f"Material {material_id} missing E/nu: {row}")
                    props["E"] = E
                    props["nu"] = nu
                    if rho is not None:
                        props["rho"] = rho

                elements.append(
                    Element2D(
                        id=eid,
                        node_ids=nids,
                        type="Quad8Plane",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No CPS8/CPE8 *Element data found in {inp_path}")

    if fix_orientation:
        for e in elements:
            try:
                n1, n2, n3, n4 = (node_lookup[e.node_ids[i]] for i in range(4))
            except KeyError as ex:
                raise KeyError(f"Element {e.id} references missing node {ex.args[0]}")
            area = 0.5 * (
                n1.x * n2.y - n2.x * n1.y
                + n2.x * n3.y - n3.x * n2.y
                + n3.x * n4.y - n4.x * n3.y
                + n4.x * n1.y - n1.x * n4.y
            )
            if area < 0.0:
                if len(e.node_ids) != 8:
                    raise ValueError(f"Element {e.id} expected 8 nodes for orientation fix, got {len(e.node_ids)}")
                n1_id, n2_id, n3_id, n4_id, n5_id, n6_id, n7_id, n8_id = e.node_ids
                e.node_ids = [n1_id, n4_id, n3_id, n2_id, n8_id, n7_id, n6_id, n5_id]

    return PlaneMesh2D(nodes=nodes, elements=elements)


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
    materials: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials = read_materials_as_dict(material_path)
        if material_id not in materials:
            raise KeyError(f"material_id={material_id} not found in {material_path}")

    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None
    pending_elem_parts: List[str] = []

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                if in_elem and pending_elem_parts:
                    raise ValueError(
                        f"Incomplete C3D10 connectivity record in {inp_path}: {pending_elem_parts}"
                    )
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet10 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    pending_elem_parts = []
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("C3D10", "C3D10T"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                pending_elem_parts.extend(split_nums(line))
                while len(pending_elem_parts) >= 11:
                    eid = int(pending_elem_parts[0])
                    nids = [int(p) for p in pending_elem_parts[1:11]]
                    pending_elem_parts = pending_elem_parts[11:]

                    props: Dict[str, any] = {
                        "material_id": int(material_id),
                    }

                    if materials:
                        row = materials[int(material_id)]
                        E = _get_float_from_material(row, ["E", "young", "youngs_modulus"])
                        nu = _get_float_from_material(row, ["nu", "poisson", "poisson_ratio"])
                        rho = _get_float_from_material(row, ["rho", "rou", "density"])
                        if E is None or nu is None:
                            raise KeyError(f"Material {material_id} missing E/nu: {row}")
                        props["E"] = E
                        props["nu"] = nu
                        if rho is not None:
                            props["rho"] = rho

                    elements.append(
                        Element3D(
                            id=eid,
                            node_ids=nids,
                            type="Tet10",
                            props=props,
                        )
                    )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if pending_elem_parts:
        raise ValueError(
            f"Incomplete C3D10 connectivity record at end of file {inp_path}: {pending_elem_parts}"
        )
    if not elements:
        raise ValueError(f"No C3D10 *Element data found in {inp_path}")

    from .stiffness import _tet10_gauss_points, _tet10_shape_funcs_grads

    for e in elements:
        coords = [node_lookup[nid] for nid in e.node_ids]
        x = np.array([n.x for n in coords], dtype=float)
        y = np.array([n.y for n in coords], dtype=float)
        z = np.array([n.z for n in coords], dtype=float)

        for xi, eta, zeta, _ in _tet10_gauss_points():
            _, dN_dxi, dN_deta, dN_dzeta = _tet10_shape_funcs_grads(xi, eta, zeta)
            J = np.array([
                [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
                [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
                [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
            ], dtype=float)
            if np.linalg.det(J) <= 0.0:
                raise ValueError(
                    f"Element {e.id} has zero or negative Jacobian determinant. Check node ordering."
                )

    return TetMesh3D(nodes=nodes, elements=elements)


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
    materials: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials = read_materials_as_dict(material_path)
        if material_id not in materials:
            raise KeyError(f"material_id={material_id} not found in {material_path}")

    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                if kw.startswith("*ASSEMBLY"):
                    # This reader consumes the part-level Tet4 mesh only.
                    # Assembly sections may contain reference points that reuse node ids.
                    break
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et in ("C3D4", "C3D4T"):
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 5:
                    continue
                eid = int(parts[0])
                n1, n2, n3, n4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

                props: Dict[str, any] = {
                    "material_id": int(material_id),
                }

                if materials:
                    row = materials[int(material_id)]
                    E = _get_float_from_material(row, ["E", "young", "youngs_modulus"])
                    nu = _get_float_from_material(row, ["nu", "poisson", "poisson_ratio"])
                    rho = _get_float_from_material(row, ["rho", "rou", "density"])
                    if E is None or nu is None:
                        raise KeyError(f"Material {material_id} missing E/nu: {row}")
                    props["E"] = E
                    props["nu"] = nu
                    if rho is not None:
                        props["rho"] = rho

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=[n1, n2, n3, n4],
                        type="Tet4",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D4/C3D4T *Element data found in {inp_path}")

    # Check volume (Jacobian determinant) for each element
    for e in elements:
        n1, n2, n3, n4 = (node_lookup[nid] for nid in e.node_ids)
        # Volume = det(J)/6 where J columns are (x2-x1, x3-x1, x4-x1)
        v1 = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z])
        v2 = np.array([n3.x - n1.x, n3.y - n1.y, n3.z - n1.z])
        v3 = np.array([n4.x - n1.x, n4.y - n1.y, n4.z - n1.z])
        vol = np.dot(v1, np.cross(v2, v3)) / 6.0
        if vol <= 0.0:
            raise ValueError(
                f"Element {e.id} has zero or negative volume "
                f"(nodes: {e.node_ids}). Check node ordering."
            )

    return TetMesh3D(nodes=nodes, elements=elements)


def read_hex8_3d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read a Hex8 3D mesh from Abaqus .inp file (C3D8 elements)."""
    materials: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials = read_materials_as_dict(material_path)
        if material_id not in materials:
            raise KeyError(f"material_id={material_id} not found in {material_path}")

    nodes: List[Node3D] = []
    elements: List[Element3D] = []
    node_lookup: Dict[int, Node3D] = {}

    in_node = False
    in_elem = False
    elem_abaqus_type: Optional[str] = None

    def split_nums(line: str) -> List[str]:
        parts = [p.strip() for p in line.strip().split(",")]
        return [p for p in parts if p]

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                kw = line.strip().upper()
                in_node = kw.startswith("*NODE")
                if kw.startswith("*ELEMENT"):
                    in_elem = False
                    elem_abaqus_type = None
                    parts = [p.strip() for p in kw.split(",")]
                    et = None
                    for p in parts:
                        if p.startswith("TYPE="):
                            et = p.split("=", 1)[1].strip()
                            break
                    if et == "C3D8":
                        in_elem = True
                        elem_abaqus_type = et
                else:
                    in_elem = False
                    elem_abaqus_type = None
                continue

            if in_node:
                parts = split_nums(line)
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                n = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(n)
                node_lookup[nid] = n
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 9:
                    continue
                eid = int(parts[0])
                nids = [int(p) for p in parts[1:9]]

                props: Dict[str, any] = {
                    "material_id": int(material_id),
                }

                if materials:
                    row = materials[int(material_id)]
                    E = _get_float_from_material(row, ["E", "young", "youngs_modulus"])
                    nu = _get_float_from_material(row, ["nu", "poisson", "poisson_ratio"])
                    rho = _get_float_from_material(row, ["rho", "rou", "density"])
                    if E is None or nu is None:
                        raise KeyError(f"Material {material_id} missing E/nu: {row}")
                    props["E"] = E
                    props["nu"] = nu
                    if rho is not None:
                        props["rho"] = rho

                elements.append(
                    Element3D(
                        id=eid,
                        node_ids=nids,
                        type="Hex8",
                        props=props,
                    )
                )

    if not nodes:
        raise ValueError(f"No *Node data found in {inp_path}")
    if not elements:
        raise ValueError(f"No C3D8 *Element data found in {inp_path}")

    return HexMesh3D(nodes=nodes, elements=elements)


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

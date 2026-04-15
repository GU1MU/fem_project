from __future__ import  annotations

import csv
from typing import Dict, List, Optional, Tuple
from .mesh import Node2D, Element2D, TrussMesh2D, BeamMesh2D, PlaneMesh2D, Node3D, Element3D, HexMesh3D


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

def read_hex8_3d_abaqus(
    inp_path: str,
    material_id: int,
    material_path: Optional[str] = None,
) -> HexMesh3D:
    """Read Hex8 3D mesh (C3D8) from Abaqus INP file."""
    materials_dict: Dict[int, Dict[str, str]] = {}
    if material_path is not None:
        materials_dict = read_materials_as_dict(material_path)
        if material_id not in materials_dict:
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

                    if et in ("C3D8"):
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

                node = Node3D(id=nid, x=x, y=y, z=z)
                nodes.append(node)
                node_lookup[nid] = node
                continue

            if in_elem and elem_abaqus_type is not None:
                parts = split_nums(line)
                if len(parts) < 9:
                    continue

                eid = int(parts[0])
                nids = [int(p) for p in parts[1:9]]

                props: Dict[str, object] = {
                    "material_id": int(material_id),
                }

                if materials_dict:
                    row = materials_dict[int(material_id)]
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
        raise ValueError(f"No C3D8 Element data found in {inp_path}")

    return HexMesh3D(nodes=nodes, elements=elements)
from __future__ import  annotations

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
from .boundary import BoundaryCondition2D, BoundaryCondition3D
from .mesh import Node2D, Element2D, TrussMesh2D, BeamMesh2D, PlaneMesh2D, Node3D, Element3D, HexMesh3D, TetMesh3D


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


from .abaqus.parser import read_abaqus_inp_model as _read_abaqus_inp_model_impl
from .abaqus.convert import (
    build_boundary_from_inp_model as _build_boundary_from_inp_model_impl,
    build_mesh_from_inp_model as _build_mesh_from_inp_model_impl,
    read_abaqus_inp_as_model_data as _read_abaqus_inp_as_model_data_impl,
)


def _classify_inp_element_family(
    element_type: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    normalized = element_type.strip().upper()

    if normalized == "":
        return None

    if normalized.startswith("CPS3"):
        return ("plane_tri3", "Tri3Plane", "stress")
    if normalized.startswith("CPE3"):
        return ("plane_tri3", "Tri3Plane", "strain")
    if normalized.startswith("CPS4"):
        return ("plane_quad4", "Quad4Plane", "stress")
    if normalized.startswith("CPE4"):
        return ("plane_quad4", "Quad4Plane", "strain")
    if normalized.startswith("CPS8"):
        return ("plane_quad8", "Quad8Plane", "stress")
    if normalized.startswith("CPE8"):
        return ("plane_quad8", "Quad8Plane", "strain")
    if normalized.startswith("C3D4"):
        return ("solid_tet4", "Tet4", None)
    if normalized.startswith("C3D8"):
        return ("solid_hex8", "Hex8", None)

    return ("unsupported", normalized, None)


def _build_element_id_to_elset_names(model: AbaqusInpModel) -> Dict[int, List[str]]:
    element_id_to_elsets: Dict[int, List[str]] = {}

    for scope_sets in model.elsets.values():
        for elset_name, element_ids in scope_sets.items():
            for element_id in element_ids:
                element_id_to_elsets.setdefault(element_id, [])
                if elset_name not in element_id_to_elsets[element_id]:
                    element_id_to_elsets[element_id].append(elset_name)

    for block in model.element_blocks:
        if block.elset is None:
            continue
        for element in block.elements:
            element_id_to_elsets.setdefault(element.id, [])
            if block.elset not in element_id_to_elsets[element.id]:
                element_id_to_elsets[element.id].append(block.elset)

    return element_id_to_elsets


def _resolve_inp_section_material_props(
    model: AbaqusInpModel,
) -> Dict[int, Dict[str, object]]:
    section_props_by_element_id: Dict[int, Dict[str, object]] = {}
    element_id_to_elsets = _build_element_id_to_elset_names(model)

    for section in model.sections:
        if section.elset is None:
            raise ValueError("Section is missing required elset for mesh conversion")
        if section.material_name is None:
            raise ValueError(
                f"Section '{section.elset}' is missing required material for mesh conversion"
            )

        material = model.materials.get(section.material_name)
        if material is None:
            raise KeyError(
                f"Section '{section.elset}' references unknown material '{section.material_name}'"
            )
        if material.elastic is None or len(material.elastic) < 2:
            raise KeyError(
                f"Material '{section.material_name}' is missing required elastic data"
            )

        props: Dict[str, object] = {
            "section_type": section.section_type,
            "material_name": section.material_name,
            "E": float(material.elastic[0]),
            "nu": float(material.elastic[1]),
        }
        if material.density is not None:
            props["rho"] = float(material.density)
        if section.data and section.data[0]:
            first_value = section.data[0][0]
            if first_value != "":
                props["thickness"] = float(first_value)

        matched_element_ids = [
            element_id
            for element_id, elset_names in element_id_to_elsets.items()
            if section.elset in elset_names
        ]
        if not matched_element_ids:
            raise ValueError(
                f"Section '{section.elset}' does not resolve to any parsed elements"
            )

        for element_id in matched_element_ids:
            if element_id in section_props_by_element_id:
                raise ValueError(
                    f"Element {element_id} is assigned by multiple sections during mesh conversion"
                )
            section_props_by_element_id[element_id] = dict(props)

    return section_props_by_element_id


def _build_plane_nodes_from_inp_model(
    model: AbaqusInpModel,
) -> Tuple[List[Node2D], Dict[int, Node2D]]:
    nodes: List[Node2D] = []
    node_lookup: Dict[int, Node2D] = {}

    for node_id, node in sorted(model.nodes.items()):
        if len(node.coordinates) < 2:
            raise ValueError(f"Node {node_id} must have at least 2 coordinates")

        runtime_node = Node2D(
            id=node_id,
            x=float(node.coordinates[0]),
            y=float(node.coordinates[1]),
        )
        nodes.append(runtime_node)
        node_lookup[node_id] = runtime_node

    return nodes, node_lookup


def _validate_plane_element_connectivity(
    family_name: str,
    elements: List[Element2D],
    node_lookup: Dict[int, Node2D],
) -> None:
    expected_counts = {
        "plane_tri3": 3,
        "plane_quad4": 4,
        "plane_quad8": 8,
    }
    expected_count = expected_counts.get(family_name)

    for element in elements:
        if expected_count is not None and len(element.node_ids) != expected_count:
            raise ValueError(
                f"Element {element.id} must have exactly {expected_count} node IDs for {family_name}"
            )
        for node_id in element.node_ids:
            if node_id not in node_lookup:
                raise ValueError(f"Element {element.id} references missing node {node_id}")


def _signed_area_for_quad_nodes(node_lookup: Dict[int, Node2D], node_ids: List[int]) -> float:
    if len(node_ids) < 4:
        raise ValueError(
            f"Element connectivity must provide at least 4 nodes for quad orientation validation: {node_ids}"
        )

    n1, n2, n3, n4 = (node_lookup[node_ids[index]] for index in range(4))
    return 0.5 * (
        n1.x * n2.y - n2.x * n1.y
        + n2.x * n3.y - n3.x * n2.y
        + n3.x * n4.y - n4.x * n3.y
        + n4.x * n1.y - n1.x * n4.y
    )


def _fix_and_validate_plane_element_orientation(
    family_name: str,
    elements: List[Element2D],
    node_lookup: Dict[int, Node2D],
) -> None:
    if family_name not in {"plane_quad4", "plane_quad8"}:
        return

    for element in elements:
        area = _signed_area_for_quad_nodes(node_lookup, element.node_ids)
        if area < 0.0:
            if family_name == "plane_quad4":
                n1_id, n2_id, n3_id, n4_id = element.node_ids
                element.node_ids = [n1_id, n4_id, n3_id, n2_id]
            elif family_name == "plane_quad8":
                if len(element.node_ids) != 8:
                    raise ValueError(
                        f"Element {element.id} expected 8 nodes for orientation fix, got {len(element.node_ids)}"
                    )
                n1_id, n2_id, n3_id, n4_id, n5_id, n6_id, n7_id, n8_id = element.node_ids
                element.node_ids = [n1_id, n4_id, n3_id, n2_id, n8_id, n7_id, n6_id, n5_id]
                area = _signed_area_for_quad_nodes(node_lookup, element.node_ids)

        if area == 0.0:
            raise ValueError(f"Element {element.id} has invalid quad orientation or zero area")


def _build_solid_nodes_from_inp_model(
    model: AbaqusInpModel,
) -> Tuple[List[Node3D], Dict[int, Node3D]]:
    nodes: List[Node3D] = []
    node_lookup: Dict[int, Node3D] = {}

    for node_id, node in sorted(model.nodes.items()):
        if len(node.coordinates) < 3:
            raise ValueError(f"Node {node_id} must have at least 3 coordinates")

        runtime_node = Node3D(
            id=node_id,
            x=float(node.coordinates[0]),
            y=float(node.coordinates[1]),
            z=float(node.coordinates[2]),
        )
        nodes.append(runtime_node)
        node_lookup[node_id] = runtime_node

    return nodes, node_lookup


def _validate_solid_element_connectivity(
    family_name: str,
    elements: List[Element3D],
    node_lookup: Dict[int, Node3D],
) -> None:
    expected_counts = {
        "solid_tet4": 4,
        "solid_hex8": 8,
    }
    expected_count = expected_counts.get(family_name)

    for element in elements:
        if expected_count is not None and len(element.node_ids) != expected_count:
            raise ValueError(
                f"Element {element.id} must have exactly {expected_count} node IDs for {family_name}"
            )
        for node_id in element.node_ids:
            if node_id not in node_lookup:
                raise ValueError(f"Element {element.id} references missing node {node_id}")


def build_mesh_from_inp_model(model: AbaqusInpModel) -> Any:
    """Convert a parsed Abaqus INP model into the current runtime mesh containers."""

    supported_families: List[str] = []
    unsupported_types: List[str] = []

    for block in model.element_blocks:
        family_info = _classify_inp_element_family(block.element_type)
        if family_info is None:
            continue
        family_name = family_info[0]
        if family_name == "unsupported":
            unsupported_types.append(block.element_type)
            continue
        if family_name not in supported_families:
            supported_families.append(family_name)

    if unsupported_types:
        raise ValueError(
            "Unsupported Abaqus element types for mesh conversion: "
            + ", ".join(sorted(unsupported_types))
        )

    if not supported_families:
        raise ValueError("No supported Abaqus elements found for mesh conversion")

    if len(supported_families) != 1:
        raise ValueError(
            "Cannot build mesh from incompatible mixed element families: "
            + ", ".join(supported_families)
        )

    selected_family = supported_families[0]
    section_props_by_element_id = _resolve_inp_section_material_props(model)
    if selected_family in {"plane_tri3", "plane_quad4", "plane_quad8"}:
        nodes, node_lookup = _build_plane_nodes_from_inp_model(model)
        if not nodes:
            raise ValueError("No nodes found for mesh conversion")

        elements: List[Element2D] = []
        for block in model.element_blocks:
            family_info = _classify_inp_element_family(block.element_type)
            if family_info is None or family_info[0] != selected_family:
                continue

            _, runtime_element_type, plane_type = family_info
            for inp_element in block.elements:
                if inp_element.id not in section_props_by_element_id:
                    raise ValueError(
                        f"Element {inp_element.id} is missing section/material assignment"
                    )

                props = dict(section_props_by_element_id[inp_element.id])
                if plane_type is not None:
                    props["plane_type"] = plane_type

                elements.append(
                    Element2D(
                        id=inp_element.id,
                        node_ids=list(inp_element.node_ids),
                        type=runtime_element_type,
                        props=props,
                    )
                )

        if not elements:
            raise ValueError("No elements found for mesh conversion")

        _validate_plane_element_connectivity(selected_family, elements, node_lookup)
        _fix_and_validate_plane_element_orientation(selected_family, elements, node_lookup)

        return PlaneMesh2D(nodes=nodes, elements=elements)

    if selected_family in {"solid_tet4", "solid_hex8"}:
        nodes, node_lookup = _build_solid_nodes_from_inp_model(model)
        if not nodes:
            raise ValueError("No nodes found for mesh conversion")

        elements_3d: List[Element3D] = []
        for block in model.element_blocks:
            family_info = _classify_inp_element_family(block.element_type)
            if family_info is None or family_info[0] != selected_family:
                continue

            _, runtime_element_type, _ = family_info
            for inp_element in block.elements:
                if inp_element.id not in section_props_by_element_id:
                    raise ValueError(
                        f"Element {inp_element.id} is missing section/material assignment"
                    )

                elements_3d.append(
                    Element3D(
                        id=inp_element.id,
                        node_ids=list(inp_element.node_ids),
                        type=runtime_element_type,
                        props=dict(section_props_by_element_id[inp_element.id]),
                    )
                )

        if not elements_3d:
            raise ValueError("No elements found for mesh conversion")

        _validate_solid_element_connectivity(selected_family, elements_3d, node_lookup)

        if selected_family == "solid_tet4":
            return TetMesh3D(nodes=nodes, elements=elements_3d)
        return HexMesh3D(nodes=nodes, elements=elements_3d)

    raise ValueError(
        "Unsupported Abaqus element family for mesh conversion: "
        + selected_family
    )


def _select_inp_step(
    model: AbaqusInpModel,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> InpStaticStep:
    if not model.steps:
        raise ValueError("No supported Abaqus *Step data found for boundary conversion")

    if step_name is not None:
        for step in model.steps:
            if step.name == step_name:
                return step
        raise KeyError(f"Abaqus step '{step_name}' not found")

    if step_index < 0 or step_index >= len(model.steps):
        raise IndexError(
            f"step_index {step_index} out of range for {len(model.steps)} parsed steps"
        )

    return model.steps[step_index]


def _deduplicate_preserving_order(values: List[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_inp_set_target(
    target: str,
    scoped_sets: Dict[str, Dict[str, List[int]]],
    *,
    part_name: Optional[str],
    instance_name: Optional[str],
) -> Optional[List[int]]:
    if target in scoped_sets.get("assembly", {}):
        return _deduplicate_preserving_order(scoped_sets["assembly"][target])

    if target in scoped_sets.get("part", {}):
        return _deduplicate_preserving_order(scoped_sets["part"][target])

    if "." not in target:
        return None

    prefix, suffix = target.split(".", 1)
    if suffix == "":
        return None

    if instance_name is not None and prefix == instance_name and suffix in scoped_sets.get("part", {}):
        return _deduplicate_preserving_order(scoped_sets["part"][suffix])

    if part_name is not None and prefix == part_name and suffix in scoped_sets.get("part", {}):
        return _deduplicate_preserving_order(scoped_sets["part"][suffix])

    return None


def _resolve_inp_target_ids(
    model: AbaqusInpModel,
    target: str,
    *,
    scoped_sets: Dict[str, Dict[str, List[int]]],
    valid_ids: List[int],
    kind: str,
) -> List[int]:
    resolved = _resolve_inp_set_target(
        target,
        scoped_sets,
        part_name=model.part_name,
        instance_name=model.instance_name,
    )
    valid_id_set = set(valid_ids)

    if resolved is None:
        try:
            resolved = [int(target)]
        except ValueError as exc:
            raise KeyError(f"Unable to resolve Abaqus {kind} target '{target}'") from exc

    missing_ids = [value for value in resolved if value not in valid_id_set]
    if missing_ids:
        raise ValueError(
            f"Abaqus {kind} target '{target}' resolved to IDs not present in mesh: {missing_ids}"
        )

    return resolved


def _validate_abaqus_dof(dof: int, dofs_per_node: int, *, context: str) -> int:
    component = dof - 1
    if component < 0 or component >= dofs_per_node:
        raise ValueError(
            f"Unsupported Abaqus DOF {dof} for {context} on mesh with {dofs_per_node} DOFs per node"
        )
    return component


def _resolve_gravity_components(
    spec: InpDloadSpec,
    *,
    dimensions: int,
) -> Tuple[float, ...]:
    if len(spec.components) < dimensions:
        raise ValueError(
            f"Unsupported *Dload conversion for target '{spec.target}': "
            f"GRAV requires at least {dimensions} direction components"
        )

    extra_components = spec.components[dimensions:]
    if any(component != 0.0 for component in extra_components):
        raise ValueError(
            f"Unsupported *Dload conversion for target '{spec.target}': "
            "out-of-plane GRAV components are not supported"
        )

    return tuple(spec.magnitude * component for component in spec.components[:dimensions])


def build_boundary_from_inp_model(
    model: AbaqusInpModel,
    mesh: Any,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> Any:
    step = _select_inp_step(model, step_name=step_name, step_index=step_index)

    if isinstance(mesh, PlaneMesh2D):
        boundary: Any = BoundaryCondition2D()
        dimensions = 2
        valid_node_ids = [node.id for node in mesh.nodes]
        valid_element_ids = [element.id for element in mesh.elements]
        element_lookup = {element.id: element for element in mesh.elements}
    elif isinstance(mesh, (HexMesh3D, TetMesh3D)):
        boundary = BoundaryCondition3D()
        dimensions = 3
        valid_node_ids = [node.id for node in mesh.nodes]
        valid_element_ids = [element.id for element in mesh.elements]
        element_lookup = {element.id: element for element in mesh.elements}
    else:
        raise TypeError(
            "build_boundary_from_inp_model only supports PlaneMesh2D, TetMesh3D, or HexMesh3D"
        )

    for spec in step.boundary_specs:
        node_ids = _resolve_inp_target_ids(
            model,
            spec.target,
            scoped_sets=model.nsets,
            valid_ids=valid_node_ids,
            kind="node",
        )

        if spec.boundary_type is not None:
            if spec.boundary_type != "ENCASTRE":
                raise ValueError(
                    f"Unsupported *Boundary conversion type '{spec.boundary_type}' for target '{spec.target}'"
                )
            for node_id in node_ids:
                boundary.add_fixed_support(node_id, range(mesh.dofs_per_node), mesh)
            continue

        if spec.first_dof is None or spec.last_dof is None:
            raise ValueError(
                f"Unsupported *Boundary conversion for target '{spec.target}': missing DOF range"
            )
        if spec.first_dof > spec.last_dof:
            raise ValueError(
                f"Unsupported *Boundary conversion for target '{spec.target}': "
                f"first_dof > last_dof ({spec.first_dof} > {spec.last_dof})"
            )

        value = 0.0 if spec.value is None else float(spec.value)
        for abaqus_dof in range(spec.first_dof, spec.last_dof + 1):
            if abaqus_dof > mesh.dofs_per_node:
                if value != 0.0:
                    raise ValueError(
                        f"Unsupported Abaqus DOF {abaqus_dof} for *Boundary target "
                        f"'{spec.target}' on mesh with {mesh.dofs_per_node} DOFs per node"
                    )
                continue
            component = _validate_abaqus_dof(
                abaqus_dof,
                mesh.dofs_per_node,
                context=f"*Boundary target '{spec.target}'",
            )
            for node_id in node_ids:
                boundary.add_displacement(node_id, component, value, mesh)

    for spec in step.cload_specs:
        node_ids = _resolve_inp_target_ids(
            model,
            spec.target,
            scoped_sets=model.nsets,
            valid_ids=valid_node_ids,
            kind="node",
        )
        component = _validate_abaqus_dof(
            spec.dof,
            mesh.dofs_per_node,
            context=f"*Cload target '{spec.target}'",
        )
        for node_id in node_ids:
            dof_id = mesh.global_dof(node_id, component)
            boundary.nodal_forces[dof_id] = (
                boundary.nodal_forces.get(dof_id, 0.0) + float(spec.magnitude)
            )

    for spec in step.dload_specs:
        if spec.load_type != "GRAV":
            raise ValueError(
                f"Unsupported *Dload conversion for target '{spec.target}': {spec.load_type}"
            )

        element_ids = _resolve_inp_target_ids(
            model,
            spec.target,
            scoped_sets=model.elsets,
            valid_ids=valid_element_ids,
            kind="element",
        )
        gravity_components = _resolve_gravity_components(spec, dimensions=dimensions)

        for element_id in element_ids:
            element = element_lookup[element_id]
            density = element.props.get("rho")
            if density is None:
                raise KeyError(
                    f"Element {element_id} is missing density required for GRAV conversion"
                )

            if dimensions == 2:
                boundary.add_body_force_element(
                    element_id,
                    float(density) * gravity_components[0],
                    float(density) * gravity_components[1],
                )
            else:
                boundary.add_body_force_element(
                    element_id,
                    float(density) * gravity_components[0],
                    float(density) * gravity_components[1],
                    float(density) * gravity_components[2],
                )

    return boundary


def read_abaqus_inp_as_model_data(
    inp_path: str,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> InpModelData:
    model = read_abaqus_inp_model(inp_path)
    step = _select_inp_step(model, step_name=step_name, step_index=step_index)
    mesh = build_mesh_from_inp_model(model)
    boundary = build_boundary_from_inp_model(
        model,
        mesh,
        step_name=step.name if step_name is None else step_name,
        step_index=step_index,
    )
    return InpModelData(model=model, mesh=mesh, boundary=boundary, step=step)


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

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..boundary import BoundaryCondition2D, BoundaryCondition3D
from ..mesh import Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D
from .model import AbaqusInpModel, InpDloadSpec, InpModelData
from .parser import read_abaqus_inp_model
from .resolve import (
    _resolve_inp_section_material_props,
    _resolve_inp_target_ids,
    _select_inp_step,
    _validate_abaqus_dof,
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


def _build_mesh_from_inp_model(model: AbaqusInpModel) -> Any:
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
        "Unsupported Abaqus element family for mesh conversion: " + selected_family
    )


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


def _build_boundary_from_inp_model(
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
            "_build_boundary_from_inp_model only supports PlaneMesh2D, TetMesh3D, or HexMesh3D"
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


def _read_abaqus_inp_as_model_data(
    inp_path: str,
    *,
    step_name: Optional[str] = None,
    step_index: int = 0,
) -> InpModelData:
    model = read_abaqus_inp_model(inp_path)
    step = _select_inp_step(model, step_name=step_name, step_index=step_index)
    mesh = _build_mesh_from_inp_model(model)
    boundary = _build_boundary_from_inp_model(
        model,
        mesh,
        step_name=step.name if step_name is None else step_name,
        step_index=step_index,
    )
    return InpModelData(model=model, mesh=mesh, boundary=boundary, step=step)

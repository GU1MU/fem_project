from __future__ import annotations

from typing import Any

from ..boundary.condition import BoundaryCondition
from ..core.mesh import Element2D, Element3D, HexMesh3D, Node2D, Node3D, PlaneMesh2D, TetMesh3D
from ..core.model import (
    AnalysisStep,
    DisplacementConstraint,
    ElementFace,
    ElementSet,
    FEMModel,
    MaterialDefinition,
    NodalLoad,
    NodeSet,
    SectionAssignment,
    Surface,
)
from ..selection import faces as face_selection
from .deck import AbaqusBoundary, AbaqusDeck, AbaqusElement, AbaqusStep


def build_model(deck: AbaqusDeck) -> FEMModel:
    """Build a FEMModel from a parsed Abaqus input deck."""
    mesh = _build_mesh(deck)
    node_sets = {
        name: NodeSet(name, _unique_ids(ids))
        for name, ids in deck.node_sets.items()
    }
    element_sets = {
        name: ElementSet(name, _unique_ids(ids))
        for name, ids in deck.element_sets.items()
    }
    materials = {
        name: MaterialDefinition(name, dict(material.properties))
        for name, material in deck.materials.items()
    }
    sections = [
        SectionAssignment(section.element_set, section.material, section.section_type)
        for section in deck.sections
    ]

    _apply_sections(mesh, deck, materials, element_sets)
    surfaces = _build_surfaces(mesh, deck, element_sets)
    steps = [_build_step(step, mesh) for step in deck.steps]
    boundary = _build_boundary(mesh, node_sets, steps[0]) if steps else BoundaryCondition()

    return FEMModel(
        mesh=mesh,
        boundary=boundary,
        name=deck.name,
        node_sets=node_sets,
        element_sets=element_sets,
        surfaces=surfaces,
        materials=materials,
        sections=sections,
        steps=steps,
    )


def _build_mesh(deck: AbaqusDeck) -> Any:
    """Build a mesh from deck nodes and elements."""
    if not deck.nodes:
        raise ValueError("Abaqus deck has no nodes")
    if not deck.elements:
        raise ValueError("Abaqus deck has no elements")

    dimension = _mesh_dimension(deck.elements)
    if dimension == 2:
        nodes2d = [
            Node2D(node_id, coords[0], coords[1])
            for node_id, coords in sorted(deck.nodes.items())
        ]
        elements2d = [
            Element2D(
                element.id,
                list(element.node_ids),
                _element_type(element),
                _element_props(element),
            )
            for element in deck.elements
        ]
        return PlaneMesh2D(nodes2d, elements2d)

    nodes3d = [
        Node3D(node_id, coords[0], coords[1], coords[2])
        for node_id, coords in sorted(deck.nodes.items())
    ]
    elements3d = [
        Element3D(
            element.id,
            list(element.node_ids),
            _element_type(element),
            _element_props(element),
        )
        for element in deck.elements
    ]
    if all("tet" in elem.type.lower() for elem in elements3d):
        return TetMesh3D(nodes3d, elements3d)
    return HexMesh3D(nodes3d, elements3d)


def _apply_sections(
    mesh: Any,
    deck: AbaqusDeck,
    materials: dict[str, MaterialDefinition],
    element_sets: dict[str, ElementSet],
) -> None:
    """Copy section material properties onto mesh elements."""
    element_lookup = {elem.id: elem for elem in mesh.elements}
    for section in deck.sections:
        if section.material not in materials:
            raise KeyError(f"section material {section.material} is not defined")
        if section.element_ids:
            element_ids = section.element_ids
        else:
            element_ids = element_sets[section.element_set].element_ids
        for element_id in element_ids:
            elem = element_lookup[element_id]
            elem.props.update(materials[section.material].properties)
            elem.props["material"] = section.material


def _build_surfaces(
    mesh: Any,
    deck: AbaqusDeck,
    element_sets: dict[str, ElementSet],
) -> dict[str, Surface]:
    """Build named model surfaces from deck surface entries."""
    face_lookup = {
        (elem_id, local_index): node_ids
        for elem_id, local_index, node_ids in face_selection.all(mesh)
    }
    surfaces: dict[str, Surface] = {}

    for name, entries in deck.surfaces.items():
        model_faces: list[ElementFace] = []
        for entry in entries:
            local_index = _face_label_to_index(entry.face_label)
            for element_id in _resolve_element_target(entry.target, element_sets):
                node_ids = face_lookup.get((element_id, local_index))
                if node_ids is None:
                    raise ValueError(
                        f"element {element_id} does not have Abaqus face {entry.face_label}"
                    )
                model_faces.append(ElementFace(element_id, local_index, node_ids))
        surfaces[name] = Surface(name, model_faces)

    return surfaces


def _build_step(step: AbaqusStep, mesh: Any) -> AnalysisStep:
    """Convert raw Abaqus step data to core step data."""
    boundaries: list[DisplacementConstraint] = []
    for boundary in step.boundaries:
        for first, last, value in _constraint_ranges(boundary, mesh.dofs_per_node):
            boundaries.append(
                DisplacementConstraint(boundary.target, first, last, value)
            )

    cloads = [
        NodalLoad(load.target, load.component, load.value)
        for load in step.cloads
    ]
    return AnalysisStep(
        step.name,
        procedure=step.procedure,
        boundaries=boundaries,
        cloads=cloads,
        metadata=dict(step.metadata),
    )


def _build_boundary(
    mesh: Any,
    node_sets: dict[str, NodeSet],
    step: AnalysisStep,
) -> BoundaryCondition:
    """Build solver boundary data from one model step."""
    boundary = BoundaryCondition()
    for constraint in step.boundaries:
        for node_id in _resolve_node_target(constraint.target, node_sets):
            for component in range(constraint.first_component, constraint.last_component + 1):
                _validate_component(mesh, component)
                boundary.add_displacement(node_id, component - 1, constraint.value, mesh)

    for load in step.cloads:
        _validate_component(mesh, load.component)
        for node_id in _resolve_node_target(load.target, node_sets):
            boundary.add_nodal_force(node_id, load.component - 1, load.value, mesh)

    return boundary


def _mesh_dimension(elements: list[AbaqusElement]) -> int:
    """Infer mesh dimension from Abaqus element types."""
    dimensions = {_element_dimension(element.type) for element in elements}
    if len(dimensions) != 1:
        raise ValueError(f"mixed mesh dimensions are not supported: {dimensions}")
    return dimensions.pop()


def _element_dimension(element_type: str) -> int:
    """Return spatial dimension for an Abaqus element type."""
    etype = element_type.upper()
    if etype.startswith(("CPS", "CPE")):
        return 2
    if etype.startswith("C3D"):
        return 3
    raise ValueError(f"unsupported Abaqus element type: {element_type}")


def _element_type(element: AbaqusElement) -> str:
    """Map Abaqus element type to local element type."""
    etype = element.type.upper()
    if etype.startswith(("CPS3", "CPE3")):
        return "Tri3Plane"
    if etype.startswith(("CPS4", "CPE4")):
        return "Quad4Plane"
    if etype.startswith(("CPS8", "CPE8")):
        return "Quad8Plane"
    if etype.startswith("C3D4"):
        return "Tet4"
    if etype.startswith("C3D10"):
        return "Tet10"
    if etype.startswith("C3D8"):
        return "Hex8"
    raise ValueError(f"unsupported Abaqus element type: {element.type}")


def _element_props(element: AbaqusElement) -> dict[str, Any]:
    """Return base properties for one mesh element."""
    props: dict[str, Any] = {"abaqus_type": element.type}
    if element.element_set is not None:
        props["element_set"] = element.element_set
    if element.type.upper().startswith("CPS"):
        props["plane_type"] = "stress"
        props["thickness"] = 1.0
    elif element.type.upper().startswith("CPE"):
        props["plane_type"] = "strain"
        props["thickness"] = 1.0
    return props


def _constraint_ranges(
    boundary: AbaqusBoundary,
    dofs_per_node: int,
) -> list[tuple[int, int, float]]:
    """Return 1-based component ranges for a boundary line."""
    first = boundary.first_component
    if isinstance(first, str):
        label = first.upper()
        if label == "ENCASTRE":
            return [(1, dofs_per_node, 0.0)]
        if label == "XSYMM":
            return [(1, 1, 0.0)]
        if label == "YSYMM":
            return [(2, 2, 0.0)]
        if label == "ZSYMM":
            return [(3, 3, 0.0)]
        raise ValueError(f"unsupported Abaqus boundary label: {label}")

    last = boundary.last_component if boundary.last_component is not None else first
    return [(int(first), int(last), float(boundary.value))]


def _resolve_node_target(target: str | int, node_sets: dict[str, NodeSet]) -> tuple[int, ...]:
    """Resolve a node id or node set name."""
    if isinstance(target, int):
        return (target,)
    if target not in node_sets:
        raise KeyError(f"node set {target} is not defined")
    return node_sets[target].node_ids


def _resolve_element_target(
    target: str | int,
    element_sets: dict[str, ElementSet],
) -> tuple[int, ...]:
    """Resolve an element id or element set name."""
    if isinstance(target, int):
        return (target,)
    if target not in element_sets:
        raise KeyError(f"element set {target} is not defined")
    return element_sets[target].element_ids


def _face_label_to_index(face_label: str) -> int:
    """Convert Abaqus S1-style labels to 0-based local face index."""
    label = face_label.strip().upper()
    if not label.startswith("S"):
        raise ValueError(f"unsupported Abaqus face label: {face_label}")
    return int(label[1:]) - 1


def _validate_component(mesh: Any, component: int) -> None:
    """Validate a 1-based component against mesh DOFs."""
    if component < 1 or component > mesh.dofs_per_node:
        raise ValueError(
            f"component {component} is invalid for mesh with {mesh.dofs_per_node} DOFs per node"
        )


def _unique_ids(ids: Any) -> tuple[int, ...]:
    """Return ids without duplicates while preserving order."""
    result: list[int] = []
    seen: set[int] = set()
    for value in ids:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)

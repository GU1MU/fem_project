from __future__ import annotations

from typing import Dict, List, Optional

from .model import AbaqusInpModel, InpStaticStep


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

    if (
        instance_name is not None
        and prefix == instance_name
        and suffix in scoped_sets.get("part", {})
    ):
        return _deduplicate_preserving_order(scoped_sets["part"][suffix])

    if (
        part_name is not None
        and prefix == part_name
        and suffix in scoped_sets.get("part", {})
    ):
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

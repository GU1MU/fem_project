from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .model import (
    AbaqusInpModel,
    InpBoundarySpec,
    InpCloadSpec,
    InpDloadSpec,
    InpElement,
    InpElementBlock,
    InpMaterial,
    InpNode,
    InpSection,
    InpStaticStep,
    InpUnhandledStepSpec,
)


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
                if current_scope == "part":
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


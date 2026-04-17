# Abaqus INP Reader Expansion Design

## Background

The current Abaqus `.inp` readers in `src/fem/mesh_io.py` are specialized by element family, such as `read_quad4_2d_abaqus` and `read_tet4_3d_abaqus`. They focus on mesh extraction and limited material mapping, while boundary conditions, concentrated loads, and step data are still assembled manually in `examples/*.py`.

This works for the current single-element-family examples, but it does not scale to the next target capabilities:

- mixed element blocks in one `.inp`
- material definitions inside `.inp`
- section-to-material assignment
- boundary conditions and concentrated loads inside steps
- step and static analysis metadata

The repository architecture requires these additions to stay within the existing main flow:

`mesh_io -> mesh/dof_manager -> stiffness -> assemble -> boundary -> solve -> post`

That means the new work should extend input parsing and object construction in `mesh_io`, not move solving or assembly logic into the reader.

## Goals

- Parse Abaqus `.inp` files into a complete intermediate representation that preserves mixed element information, materials, sections, steps, and supported boundary and load data.
- Provide a conversion layer that can build current runtime objects from the parsed model:
  - existing mesh containers such as `PlaneMesh2D`, `TetMesh3D`, and `HexMesh3D`
  - existing boundary containers such as `BoundaryCondition2D` and `BoundaryCondition3D`
- Preserve mixed element information even when the current solver path cannot solve the mixed mesh directly.
- Keep current example entry points and current specialized reader APIs usable.
- Fail explicitly when parsed Abaqus semantics cannot be mapped safely onto the current solver data model.

## Non-Goals For The First Iteration

- Full Abaqus assembly semantics for multiple parts, multiple instances, and arbitrary assembly-level remapping.
- Automatic support for every Abaqus load or interaction keyword.
- Automatic solution of arbitrary mixed-element meshes in one assembled system.
- Refactoring the whole solver stack to support a new universal mesh container.
- Package installation changes. The project still runs with `PYTHONPATH=src`.

## Scope Of First Iteration

The first iteration will support these keywords at the parsing layer:

- `*Node`
- `*Element`
- `*Nset`
- `*Elset`
- `*Material`
- `*Elastic`
- `*Density`
- `*Solid Section`
- `*Step`
- `*Static`
- `*Boundary`
- `*Cload`
- `*Dload`

The first iteration will support these conversions into current runtime objects:

- mesh conversion for supported element families already handled by the solver path
- material property injection from parsed sections and materials into element `props`
- boundary conversion from `*Boundary` into prescribed displacements
- concentrated load conversion from `*Cload` into nodal forces
- step metadata exposure for downstream example scripts

The first iteration will parse but not fully materialize advanced features such as:

- `*Surface`
- `*Coupling`
- reference-node-driven coupling constraints
- other unsupported load or interaction keywords

Those items must be preserved in parsed metadata and surfaced clearly during conversion.

## Design Summary

The design uses a two-layer approach.

Layer 1 is a general Abaqus `.inp` parser that returns a neutral intermediate model object.

Layer 2 is a conversion layer that turns selected parts of that model into current project runtime objects:

- mesh containers from `src/fem/mesh.py`
- boundary condition containers from `src/fem/boundary.py`
- step metadata that examples can inspect directly

This follows the selected option C:

- keep a complete low-level spec object
- also provide convenience conversion functions

## Proposed Data Model

The new intermediate representation should live in `src/fem/mesh_io.py` for the first iteration, because the architecture explicitly places input parsing and object construction in `mesh_io`.

### `AbaqusInpModel`

Top-level parsed representation of one `.inp` file.

Recommended fields:

- `nodes`
  - mapping of node id to parsed node coordinates
- `element_blocks`
  - ordered list of parsed element blocks, one per `*Element` section
- `nsets`
  - mapping of set name to node ids
- `elsets`
  - mapping of set name to element ids
- `materials`
  - mapping of material name to parsed material object
- `sections`
  - ordered list of parsed section assignments
- `steps`
  - ordered list of parsed step objects
- `part_name`
  - optional single-part identifier for current examples
- `instance_name`
  - optional single-instance identifier for current examples
- `raw_keywords`
  - optional lightweight record of unsupported or pass-through keyword blocks

### `InpNode`

Represents one parsed node.

Recommended fields:

- `id`
- `coords`

`coords` should preserve dimensionality from the file, but downstream conversion may normalize to 2D or 3D.

### `InpElement`

Represents one parsed element record.

Recommended fields:

- `id`
- `node_ids`
- `element_type`
- `elset`

### `InpElementBlock`

Represents one `*Element, type=...` block.

Recommended fields:

- `element_type`
- `elset`
- `elements`

This is the key structure for preserving mixed element data. Even if the downstream solver only selects one compatible subset, the parser still keeps all blocks.

### `InpMaterial`

Represents one Abaqus material definition.

Recommended fields:

- `name`
- `elastic`
  - optional structure holding `E` and `nu`
- `density`
  - optional float
- `raw_properties`
  - optional mapping for future extension

The first iteration only needs `*Elastic` and `*Density`.

### `InpSection`

Represents section-to-material assignment.

Recommended fields:

- `section_type`
- `elset`
- `material_name`
- `parameters`
- `data`

The first iteration only needs `*Solid Section`, but the structure should not prevent later `*Shell Section` support.

### `InpStaticStep`

Represents one supported analysis step.

Recommended fields:

- `name`
- `procedure`
  - `static` for first iteration
- `static_parameters`
  - tuple or structure from the `*Static` data line
- `boundary_specs`
- `cload_specs`
- `dload_specs`
- `unhandled_specs`

### `InpBoundarySpec`

Represents one Abaqus boundary entry.

Recommended fields:

- `target`
  - set name or node id source token
- `first_dof`
- `last_dof`
- `value`
- `raw_tokens`

This object should preserve syntax before conversion into solver DOFs.

### `InpCloadSpec`

Represents one concentrated load entry.

Recommended fields:

- `target`
- `dof`
- `value`
- `raw_tokens`

### `InpDloadSpec`

Represents one distributed load entry.

Recommended fields:

- `target`
- `load_type`
- `value_tokens`
- `raw_tokens`

The first iteration should parse and preserve this object even if only a subset becomes convertible.

### `InpModelData`

Convenience wrapper returned by the high-level API.

Recommended fields:

- `model`
- `mesh`
- `boundary`
- `step`
- `metadata`

This gives users a direct path from `.inp` file to current runtime objects while keeping the full parsed model available.

## Parser API

### Low-Level Parsing Entry Point

Add a new parser function in `src/fem/mesh_io.py`:

```python
def read_abaqus_inp_model(inp_path: str) -> AbaqusInpModel:
    ...
```

Responsibilities:

- parse supported Abaqus keywords into the intermediate model
- preserve mixed element blocks and their original grouping
- preserve unsupported but recognized higher-level semantics as raw metadata
- avoid solving, assembly, or post-processing behavior

This function should succeed whenever the file can be parsed structurally, even if later conversion may fail.

### Conversion Entry Points

Add these conversion helpers in `src/fem/mesh_io.py`:

```python
def build_mesh_from_inp_model(
    model: AbaqusInpModel,
    *,
    element_block_names: Optional[List[str]] = None,
    element_types: Optional[List[str]] = None,
    step_name: Optional[str] = None,
):
    ...


def build_boundary_from_inp_model(
    model: AbaqusInpModel,
    mesh,
    *,
    step_name: Optional[str] = None,
):
    ...
```

Responsibilities:

- select a compatible subset of parsed elements
- resolve section-to-material mapping into element `props`
- convert supported step constraints and loads into current boundary objects
- reject unsupported mixed families or unsupported semantics with explicit errors

### High-Level Convenience Entry Point

Add:

```python
def read_abaqus_inp_as_model_data(
    inp_path: str,
    *,
    step_name: Optional[str] = None,
    element_block_names: Optional[List[str]] = None,
    element_types: Optional[List[str]] = None,
) -> InpModelData:
    ...
```

Responsibilities:

- call `read_abaqus_inp_model`
- build current runtime mesh and boundary objects
- return selected step and conversion metadata

## Mesh Conversion Rules

The parser must preserve mixed element blocks in the low-level model. The conversion layer must be stricter.

### Allowed Conversion In First Iteration

`build_mesh_from_inp_model(...)` may create a runtime mesh only when the selected elements belong to one solver-compatible family already supported by the repository.

Examples of compatible results:

- 2D plane family into `PlaneMesh2D`
  - `CPS3`
  - `CPE3`
  - `CPS4`
  - `CPE4`
  - `CPS4R`
  - `CPE4R`
  - `CPS8`
  - `CPE8`
- 3D tetrahedral family into `TetMesh3D`
  - `C3D4`
  - `C3D4T`
  - `C3D10`
  - `C3D10T`
- 3D hexahedral family into `HexMesh3D`
  - `C3D8`

### Mixed Element Policy

Mixed element information is fully supported at parse time.

At conversion time:

- if selected blocks belong to the same runtime family and the current solver path can handle them through existing mesh containers, conversion may proceed
- if selected blocks cross incompatible runtime families or imply incompatible assembly/stiffness dispatch behavior, conversion must fail explicitly

Examples:

- `CPS3 + CPS4 + CPS8` may be preserved in the parsed model, but conversion to a single current solver path should only proceed if downstream assembly and stiffness dispatch are explicitly supported
- `C3D4 + C3D10 + C3D8` may be preserved in the parsed model, but current convenience conversion should reject direct mixed solving unless explicitly implemented

The first iteration should prefer correctness and clarity over automatic coercion.

## Material And Section Mapping

Material mapping should follow the Abaqus path:

`element -> elset -> section -> material`

Conversion responsibilities:

- resolve each converted element to its section
- resolve the section to a material
- inject supported material properties into the element `props`

For the first iteration:

- `*Elastic` must map to `E` and `nu` where applicable
- `*Density` must map to `rho`
- `*Solid Section` must attach material identity to the matching elements

If a converted element has no resolvable material where the current stiffness or load path requires one, conversion must fail with a clear error.

## Boundary And Load Conversion

Boundary conversion should be implemented only in the conversion layer, not in the parser.

### `*Boundary`

Convert supported `*Boundary` entries into `BoundaryCondition2D` or `BoundaryCondition3D.prescribed_displacements`.

Rules:

- resolve the target set or node
- expand Abaqus DOF ranges into current mesh DOF components
- write explicit prescribed values to the boundary container

### `ENCASTRE`

If represented by the parsed data, `ENCASTRE` should be expanded during conversion.

First-iteration mapping:

- for current 2D meshes, expand to all available mesh DOFs
- for current 3D solid meshes, expand to all available translational DOFs

If the target mesh has a DOF layout that cannot be inferred safely, conversion must fail explicitly.

### `*Cload`

Convert supported concentrated load entries into `BoundaryCondition2D.nodal_forces` or `BoundaryCondition3D.nodal_forces`.

Rules:

- resolve the target set or node
- map Abaqus DOF numbering to current mesh component indices
- accumulate into the current boundary container

### `*Dload`

The first iteration should parse all encountered `*Dload` rows into `InpDloadSpec`, but only convert the subset that can be represented safely by current `boundary.py`.

Unsupported cases must not disappear silently. They must be reported through conversion metadata or explicit exceptions.

## Step Handling

The first iteration only needs `*Step -> *Static`.

Recommended step-selection policy:

- if the file has exactly one step, use it by default
- if the file has multiple steps, require `step_name` unless a later explicit default strategy is added

The returned step object should expose:

- step name
- static procedure parameters
- step-local boundary entries
- step-local loads

This keeps example scripts able to inspect or override step metadata without re-parsing the file.

## Unsupported And Advanced Keywords

The parser should distinguish between three states:

- supported and fully converted
- supported and parsed but not converted
- unsupported but detected

### Parsed But Not Converted In First Iteration

- `*Surface`
- `*Coupling`
- reference-node coupling semantics
- other solver features not represented by current boundary containers

These must be retained in the low-level model or metadata.

### Failure Policy

`read_abaqus_inp_model(...)`

- should fail only on structural parsing errors or malformed supported keyword content
- should not fail merely because the file contains advanced unsupported semantics

`build_mesh_from_inp_model(...)` and `build_boundary_from_inp_model(...)`

- must fail if conversion would otherwise drop physically meaningful information required for the requested step or selected mesh
- must not silently ignore unsupported constraints or loads that affect correctness

## Error Handling Requirements

The conversion layer must raise explicit, context-rich errors for:

- missing referenced node sets or element sets
- missing materials referenced by sections
- elements with unsupported or unmapped types
- incompatible mixed element selections
- step selection ambiguity when multiple steps exist
- loads or constraints that cannot be mapped to current runtime objects
- references to nodes or elements not present in the selected mesh

Error messages should identify:

- keyword type
- object name or id
- step name where relevant
- the unsupported or missing mapping

## Backward Compatibility Strategy

Existing specialized readers must remain available:

- `read_tri3_2d_abaqus`
- `read_quad4_2d_abaqus`
- `read_quad8_2d_abaqus`
- `read_tet4_3d_abaqus`
- `read_tet10_3d_abaqus`
- `read_hex8_3d_abaqus`

Compatibility plan:

- keep current function names and call patterns
- progressively reimplement them on top of `read_abaqus_inp_model(...)` plus conversion helpers
- preserve current example usage patterns unless a later explicit migration is approved

This approach keeps the current examples stable while allowing the new parser to become the internal source of truth.

## Implementation Phases

### Phase 1: General Parser

Modify `src/fem/mesh_io.py` to add:

- intermediate dataclasses
- `read_abaqus_inp_model(...)`
- supported keyword parsing logic

Expected result:

- complete low-level parsed representation of mesh, sets, materials, sections, and steps

### Phase 2: Conversion Layer

Modify `src/fem/mesh_io.py` to add:

- `build_mesh_from_inp_model(...)`
- `build_boundary_from_inp_model(...)`
- `read_abaqus_inp_as_model_data(...)`

Expected result:

- supported `.inp` files can produce current mesh and boundary objects directly

### Phase 3: Compatibility Adoption

Modify specialized readers and examples incrementally to reuse the new parser where safe.

Expected result:

- old entry points remain usable
- new functionality is available without forcing a full example rewrite

## Verification Strategy

The repository currently does not contain an active test suite, so verification should still follow a test-first mindset where feasible and focus on the highest-risk parsing and conversion paths.

### Parser Verification

Add tests or equivalent verification coverage for:

- node parsing
- element block grouping
- node set and element set parsing
- material parsing
- section-to-material mapping capture
- step parsing
- boundary, `cload`, and `dload` capture

### Conversion Verification

Add tests or equivalent verification coverage for:

- material property injection into element `props`
- prescribed displacement conversion
- concentrated load conversion
- step selection behavior
- explicit rejection of unsupported mixed-family conversion

### Compatibility Verification

Check that current single-family examples remain functionally compatible with the retained reader APIs.

All run instructions and examples must continue to respect the repository rule that imports require `PYTHONPATH=src`.

## Review Focus

Because this work is AI-assisted and touches core input parsing plus boundary-condition mapping, it requires additional review before merge.

Review must at minimum cover:

- correctness of set, section, and material resolution
- correctness of DOF mapping from Abaqus conventions into current mesh DOF conventions
- architecture consistency with `docs/project_guides/architecture.md`
- regression risk for current examples
- clarity of failure paths for unsupported semantics

If implementation later reveals a recurring parser or mapping mistake pattern, it is worth asking whether that pattern should be added to `docs/project_guides/review.md` as a future review rule.

## Open Follow-Up After First Iteration

The first iteration should leave room for later support of:

- `*Shell Section`
- `*Surface`
- `*Coupling`
- richer `*Dload` mapping
- multiple-part and multiple-instance assembly semantics
- true mixed-element assembly and stiffness dispatch

These are explicit follow-up items, not hidden requirements for the first iteration.

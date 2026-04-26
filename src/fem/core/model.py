from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NodeSet:
    """Named node id set."""
    name: str
    node_ids: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))


@dataclass(frozen=True)
class ElementSet:
    """Named element id set."""
    name: str
    element_ids: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_ids",
            tuple(int(element_id) for element_id in self.element_ids),
        )


@dataclass(frozen=True)
class ElementFace:
    """Element face identified by element id and local face index."""
    elem_id: int
    local_index: int
    node_ids: Sequence[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_id", int(self.elem_id))
        object.__setattr__(self, "local_index", int(self.local_index))
        object.__setattr__(self, "node_ids", tuple(int(node_id) for node_id in self.node_ids))


@dataclass(frozen=True)
class Surface:
    """Named collection of element faces."""
    name: str
    faces: Sequence[ElementFace]

    def __post_init__(self) -> None:
        object.__setattr__(self, "faces", tuple(self.faces))


@dataclass(frozen=True)
class MaterialDefinition:
    """Named material properties."""
    name: str
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionAssignment:
    """Assign a material to an element set."""
    element_set: str
    material: str
    section_type: str = "solid"
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "properties", dict(self.properties))


@dataclass(frozen=True)
class DisplacementConstraint:
    """Abaqus-style displacement constraint using 1-based components."""
    target: str | int
    first_component: int
    last_component: int
    value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_component", int(self.first_component))
        object.__setattr__(self, "last_component", int(self.last_component))
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class NodalLoad:
    """Abaqus-style nodal load using a 1-based component."""
    target: str | int
    component: int
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", int(self.component))
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class SurfaceLoad:
    """Surface load attached to a named surface."""
    surface: str
    vector: Sequence[float] = ()
    magnitude: float | None = None
    load_type: str = "traction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        if self.magnitude is not None:
            object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "load_type", str(self.load_type).lower())


@dataclass(frozen=True)
class OutputRequest:
    """Output request attached to an analysis step."""
    kind: str
    target: str
    variables: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind).lower())
        object.__setattr__(self, "target", str(self.target).lower())
        object.__setattr__(self, "variables", tuple(str(value) for value in self.variables))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class AnalysisStep:
    """Analysis step with loads and output metadata."""
    name: str
    procedure: str = "static"
    boundaries: Sequence[DisplacementConstraint] = ()
    cloads: Sequence[NodalLoad] = ()
    surface_loads: Sequence[SurfaceLoad] = ()
    outputs: Sequence[OutputRequest] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "cloads", tuple(self.cloads))
        object.__setattr__(self, "surface_loads", tuple(self.surface_loads))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class FEMModel:
    """Finite element model data independent of input format."""
    mesh: Any
    boundary: Any | None = None
    name: str | None = None
    node_sets: dict[str, NodeSet] = field(default_factory=dict)
    element_sets: dict[str, ElementSet] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    materials: dict[str, MaterialDefinition] = field(default_factory=dict)
    sections: list[SectionAssignment] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        name: str | None = None,
        boundary: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FEMModel":
        """Create a model from a mesh."""
        return cls(mesh=mesh, boundary=boundary, name=name, metadata=dict(metadata or {}))

    def add_node_set(
        self,
        name: str | NodeSet,
        node_ids: Sequence[int] | None = None,
    ) -> NodeSet:
        """Add a named node set."""
        node_set = (
            name
            if isinstance(name, NodeSet)
            else NodeSet(str(name), _required_ids(node_ids))
        )
        self.node_sets[node_set.name] = node_set
        return node_set

    def add_element_set(
        self,
        name: str | ElementSet,
        element_ids: Sequence[int] | None = None,
    ) -> ElementSet:
        """Add a named element set."""
        element_set = (
            name
            if isinstance(name, ElementSet)
            else ElementSet(str(name), _required_ids(element_ids))
        )
        self.element_sets[element_set.name] = element_set
        return element_set

    def add_surface(
        self,
        name: str | Surface,
        faces: Sequence[ElementFace] | None = None,
    ) -> Surface:
        """Add a named surface."""
        surface = (
            name
            if isinstance(name, Surface)
            else Surface(str(name), _required_faces(faces))
        )
        self.surfaces[surface.name] = surface
        return surface

    def add_material(
        self,
        name: str | MaterialDefinition,
        **properties: Any,
    ) -> MaterialDefinition:
        """Add named material properties."""
        if isinstance(name, MaterialDefinition):
            if properties:
                raise ValueError("material properties cannot be passed with a MaterialDefinition")
            material = name
        else:
            material = MaterialDefinition(str(name), dict(properties))
        self.materials[material.name] = material
        self._apply_sections_for_material(material.name)
        return material

    def assign_section(
        self,
        element_set: str | ElementSet,
        material: str | MaterialDefinition,
        section_type: str = "solid",
        **properties: Any,
    ) -> SectionAssignment:
        """Assign a material and section properties to an element set."""
        element_set_name = (
            element_set.name
            if isinstance(element_set, ElementSet)
            else str(element_set)
        )
        material_name = (
            material.name
            if isinstance(material, MaterialDefinition)
            else str(material)
        )
        section = SectionAssignment(
            element_set_name,
            material_name,
            section_type,
            dict(properties),
        )
        self.sections.append(section)
        self._apply_section(section)
        return section

    def add_step(
        self,
        name: str | AnalysisStep,
        procedure: str = "static",
        **metadata: Any,
    ) -> AnalysisStep:
        """Add an analysis step."""
        step = (
            name
            if isinstance(name, AnalysisStep)
            else AnalysisStep(str(name), procedure, metadata=metadata)
        )
        self.steps.append(step)
        return step

    def add_displacement(
        self,
        step: str | int | AnalysisStep | None,
        target: str | int,
        first_component: int,
        last_component: int | None = None,
        value: float = 0.0,
    ) -> DisplacementConstraint:
        """Add a displacement constraint to a step."""
        selected_step = self._require_step(step)
        constraint = DisplacementConstraint(
            target,
            first_component,
            last_component if last_component is not None else first_component,
            value,
        )
        self._replace_step(
            selected_step,
            boundaries=tuple(selected_step.boundaries) + (constraint,),
        )
        return constraint

    def add_nodal_load(
        self,
        step: str | int | AnalysisStep | None,
        target: str | int,
        component: int,
        value: float,
    ) -> NodalLoad:
        """Add a nodal load to a step."""
        selected_step = self._require_step(step)
        load = NodalLoad(target, component, value)
        self._replace_step(selected_step, cloads=tuple(selected_step.cloads) + (load,))
        return load

    def add_surface_traction(
        self,
        step: str | int | AnalysisStep | None,
        surface: str | Surface,
        vector: Sequence[float],
    ) -> SurfaceLoad:
        """Add a surface traction to a step."""
        selected_step = self._require_step(step)
        surface_name = surface.name if isinstance(surface, Surface) else str(surface)
        load = SurfaceLoad(surface_name, vector, load_type="traction")
        self._replace_step(
            selected_step,
            surface_loads=tuple(selected_step.surface_loads) + (load,),
        )
        return load

    def add_surface_pressure(
        self,
        step: str | int | AnalysisStep | None,
        surface: str | Surface,
        magnitude: float,
    ) -> SurfaceLoad:
        """Add an inward pressure load to a step."""
        selected_step = self._require_step(step)
        surface_name = surface.name if isinstance(surface, Surface) else str(surface)
        load = SurfaceLoad(surface_name, magnitude=magnitude, load_type="pressure")
        self._replace_step(
            selected_step,
            surface_loads=tuple(selected_step.surface_loads) + (load,),
        )
        return load

    def add_output_request(
        self,
        step: str | int | AnalysisStep | None,
        kind: str,
        target: str,
        variables: Sequence[str] = (),
        **metadata: Any,
    ) -> OutputRequest:
        """Add an output request to a step."""
        selected_step = self._require_step(step)
        output = OutputRequest(kind, target, variables, metadata)
        self._replace_step(selected_step, outputs=tuple(selected_step.outputs) + (output,))
        return output

    def get_step(self, step: str | int | AnalysisStep | None = None) -> AnalysisStep | None:
        """Return an analysis step by name or index."""
        if step is None:
            for candidate in self.steps:
                if candidate.name.lower() != "initial":
                    return candidate
            return self.steps[0] if self.steps else None
        if isinstance(step, AnalysisStep):
            return step
        if isinstance(step, int):
            return self.steps[step]
        for candidate in self.steps:
            if candidate.name == step:
                return candidate
        raise KeyError(f"analysis step {step} is not defined")

    def boundary_for_step(self, step: str | int | AnalysisStep | None = None) -> Any:
        """Build solver boundary data for one analysis step."""
        selected_step = self.get_step(step)
        if selected_step is None:
            if self.boundary is None:
                from ..boundary.condition import BoundaryCondition

                return BoundaryCondition()
            return self.boundary

        from ..boundary.condition import BoundaryCondition

        boundary = BoundaryCondition()
        for constraint in self._step_boundaries(selected_step):
            for node_id in self._resolve_node_target(constraint.target):
                for component in range(
                    constraint.first_component,
                    constraint.last_component + 1,
                ):
                    self._validate_component(component)
                    boundary.add_displacement(
                        node_id,
                        component - 1,
                        constraint.value,
                        self.mesh,
                    )

        for load in selected_step.cloads:
            self._validate_component(load.component)
            for node_id in self._resolve_node_target(load.target):
                boundary.add_nodal_force(
                    node_id,
                    load.component - 1,
                    load.value,
                    self.mesh,
                )

        for surface_load in selected_step.surface_loads:
            if surface_load.surface not in self.surfaces:
                raise KeyError(f"surface {surface_load.surface} is not defined")
            for face in self.surfaces[surface_load.surface].faces:
                if surface_load.load_type == "pressure":
                    vector = self._pressure_vector(face, surface_load)
                elif surface_load.load_type == "traction":
                    vector = surface_load.vector
                else:
                    raise ValueError(f"unsupported surface load type: {surface_load.load_type}")
                boundary.add_surface_traction(face.elem_id, face.local_index, *vector)

        return boundary

    def assemble_stiffness(self) -> Any:
        """Assemble sparse global stiffness for this model."""
        from ..assemble import assemble_global_stiffness_sparse

        return assemble_global_stiffness_sparse(self.mesh)

    def load_vector(self, step: str | int | AnalysisStep | None = None) -> Any:
        """Build the global load vector for one step."""
        from ..boundary.loads import build_load_vector

        return build_load_vector(self.mesh, self.boundary_for_step(step))

    def solve(self, step: str | int | AnalysisStep | None = None) -> Any:
        """Solve one linear static step."""
        return self.run(step).U

    def run(
        self,
        step: str | int | AnalysisStep | None = None,
        output_dir: Any = "results",
        name: str | None = None,
    ) -> Any:
        """Solve one linear static step and return a model result."""
        from ..boundary.constraints import apply_dirichlet
        from ..boundary.loads import build_load_vector
        from ..solvers import linear
        from .result import ModelResult

        selected_step = self.get_step(step)
        boundary = self.boundary_for_step(step)
        K = self.assemble_stiffness()
        F = build_load_vector(self.mesh, boundary)
        K_mod, F_mod = apply_dirichlet(K, F, boundary)
        U = linear.solve(K_mod, F_mod)
        reactions = K @ U - F
        return ModelResult(
            self,
            selected_step,
            U,
            reactions,
            boundary,
            output_dir=output_dir,
            name=name,
        )

    def run_all(
        self,
        steps: Any = None,
        output_dir: Any = "results",
        name: str | None = None,
    ) -> Any:
        """Solve multiple non-initial steps and return a result collection."""
        from .result import ModelResults

        selected_steps = self._run_all_steps(steps)
        multi_step = len(selected_steps) > 1
        results = [
            self.run(
                step,
                output_dir=output_dir,
                name=self._result_name(step, name, multi_step),
            )
            for step in selected_steps
        ]
        return ModelResults(self, tuple(results))

    def _resolve_node_target(self, target: str | int) -> tuple[int, ...]:
        """Resolve a node id or named node set."""
        if isinstance(target, int):
            return (target,)
        if target not in self.node_sets:
            raise KeyError(f"node set {target} is not defined")
        return self.node_sets[target].node_ids

    def _run_all_steps(self, steps: Any) -> tuple[AnalysisStep | None, ...]:
        """Resolve run_all step selectors."""
        if steps is None:
            runnable = tuple(step for step in self.steps if step.name.lower() != "initial")
            if runnable:
                return runnable
            if self.steps:
                return (self.steps[0],)
            return (None,)
        if isinstance(steps, (str, int, AnalysisStep)):
            return (self.get_step(steps),)
        return tuple(self.get_step(step) for step in steps)

    def _result_name(
        self,
        step: AnalysisStep | None,
        name: str | None,
        multi_step: bool,
    ) -> str | None:
        """Return a non-conflicting result name for run_all."""
        if not multi_step:
            return name
        base = name or self.name or "result"
        step_name = step.name if step is not None else "step"
        return f"{base}_{step_name}"

    def _require_step(self, step: str | int | AnalysisStep | None) -> AnalysisStep:
        """Return an existing step or create a default step."""
        selected_step = self.get_step(step)
        if selected_step is None:
            return self.add_step("Step-1")
        return selected_step

    def _replace_step(self, step: AnalysisStep, **changes: Any) -> AnalysisStep:
        """Update a step in the model."""
        for candidate in self.steps:
            if candidate is step:
                for name, value in changes.items():
                    setattr(step, name, value)
                return step
        raise KeyError(f"analysis step {step.name} is not in this model")

    def _apply_sections_for_material(self, material: str) -> None:
        """Apply all sections that reference a material."""
        for section in self.sections:
            if section.material == material:
                self._apply_section(section)

    def _apply_section(self, section: SectionAssignment) -> None:
        """Copy material and section properties onto element props."""
        if section.element_set not in self.element_sets:
            raise KeyError(f"element set {section.element_set} is not defined")
        if section.material not in self.materials:
            raise KeyError(f"material {section.material} is not defined")

        element_lookup = {elem.id: elem for elem in self.mesh.elements}
        props = dict(self.materials[section.material].properties)
        props.update(section.properties)
        props["material"] = section.material
        for element_id in self.element_sets[section.element_set].element_ids:
            element_lookup[element_id].props.update(props)

    def _step_boundaries(self, step: AnalysisStep) -> tuple[DisplacementConstraint, ...]:
        """Return initial boundaries inherited by the selected step."""
        initial = next(
            (candidate for candidate in self.steps if candidate.name.lower() == "initial"),
            None,
        )
        if initial is None or initial is step:
            return tuple(step.boundaries)
        return tuple(initial.boundaries) + tuple(step.boundaries)

    def _pressure_vector(self, face: ElementFace, surface_load: SurfaceLoad) -> tuple[float, ...]:
        """Return an inward pressure vector for one surface face."""
        if surface_load.magnitude is None:
            raise ValueError("pressure surface load requires a magnitude")
        import numpy as np

        node_lookup = {node.id: node for node in self.mesh.nodes}
        coords = []
        for node_id in face.node_ids:
            node = node_lookup[node_id]
            coords.append([float(node.x), float(node.y), float(getattr(node, "z", 0.0))])
        if len(coords) < 3:
            raise ValueError(f"surface face {face} must contain at least 3 nodes for pressure")

        p0 = np.array(coords[0], dtype=float)
        p1 = np.array(coords[1], dtype=float)
        p2 = np.array(coords[2], dtype=float)
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise ValueError(f"surface face {face} has zero normal")
        return tuple(float(value) for value in -surface_load.magnitude * normal / norm)

    def _validate_component(self, component: int) -> None:
        """Validate a 1-based component against mesh DOFs."""
        if component < 1 or component > self.mesh.dofs_per_node:
            raise ValueError(
                f"component {component} is invalid for mesh with "
                f"{self.mesh.dofs_per_node} DOFs per node"
            )


def _required_ids(ids: Sequence[int] | None) -> Sequence[int]:
    """Return required id data for set builders."""
    if ids is None:
        raise ValueError("ids must be provided")
    return ids


def _required_faces(faces: Sequence[ElementFace] | None) -> Sequence[ElementFace]:
    """Return required face data for surface builders."""
    if faces is None:
        raise ValueError("faces must be provided")
    return faces

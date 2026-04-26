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


@dataclass(frozen=True)
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

    def _resolve_node_target(self, target: str | int) -> tuple[int, ...]:
        """Resolve a node id or named node set."""
        if isinstance(target, int):
            return (target,)
        if target not in self.node_sets:
            raise KeyError(f"node set {target} is not defined")
        return self.node_sets[target].node_ids

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

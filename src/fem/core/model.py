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
    vector: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))


@dataclass(frozen=True)
class AnalysisStep:
    """Analysis step with loads and output metadata."""
    name: str
    procedure: str = "static"
    boundaries: Sequence[DisplacementConstraint] = ()
    cloads: Sequence[NodalLoad] = ()
    surface_loads: Sequence[SurfaceLoad] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "cloads", tuple(self.cloads))
        object.__setattr__(self, "surface_loads", tuple(self.surface_loads))
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
        for constraint in selected_step.boundaries:
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
                boundary.add_surface_traction(
                    face.elem_id,
                    face.local_index,
                    *surface_load.vector,
                )

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
        from ..boundary.constraints import apply_dirichlet
        from ..boundary.loads import build_load_vector
        from ..solvers import linear

        boundary = self.boundary_for_step(step)
        K = self.assemble_stiffness()
        F = build_load_vector(self.mesh, boundary)
        K_mod, F_mod = apply_dirichlet(K, F, boundary)
        return linear.solve(K_mod, F_mod)

    def _resolve_node_target(self, target: str | int) -> tuple[int, ...]:
        """Resolve a node id or named node set."""
        if isinstance(target, int):
            return (target,)
        if target not in self.node_sets:
            raise KeyError(f"node set {target} is not defined")
        return self.node_sets[target].node_ids

    def _validate_component(self, component: int) -> None:
        """Validate a 1-based component against mesh DOFs."""
        if component < 1 or component > self.mesh.dofs_per_node:
            raise ValueError(
                f"component {component} is invalid for mesh with "
                f"{self.mesh.dofs_per_node} DOFs per node"
            )

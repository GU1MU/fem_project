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
    name: str | None = None
    node_sets: dict[str, NodeSet] = field(default_factory=dict)
    element_sets: dict[str, ElementSet] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    materials: dict[str, MaterialDefinition] = field(default_factory=dict)
    sections: list[SectionAssignment] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

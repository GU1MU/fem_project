from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class InpNode:
    id: int
    coordinates: Tuple[float, ...]


@dataclass(frozen=True)
class InpElement:
    id: int
    node_ids: Tuple[int, ...]


@dataclass
class InpElementBlock:
    element_type: str
    elset: Optional[str] = None
    elements: List[InpElement] = field(default_factory=list)


@dataclass
class InpMaterial:
    name: str
    elastic: Optional[Tuple[float, ...]] = None
    density: Optional[float] = None


@dataclass(frozen=True)
class InpSection:
    section_type: str
    elset: Optional[str]
    material_name: Optional[str]
    parameters: Dict[str, str]
    data: List[Tuple[str, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class InpBoundarySpec:
    target: str
    boundary_type: Optional[str] = None
    first_dof: Optional[int] = None
    last_dof: Optional[int] = None
    value: Optional[float] = None
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InpCloadSpec:
    target: str
    dof: int
    magnitude: float
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InpDloadSpec:
    target: str
    load_type: str
    magnitude: float
    components: Tuple[float, ...] = ()
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class InpUnhandledStepSpec:
    keyword: str
    parameters: Dict[str, str]
    data_lines: List[Tuple[str, ...]] = field(default_factory=list)


@dataclass
class InpStaticStep:
    name: Optional[str]
    static_parameters: Tuple[float, ...] = ()
    boundary_specs: List[InpBoundarySpec] = field(default_factory=list)
    cload_specs: List[InpCloadSpec] = field(default_factory=list)
    dload_specs: List[InpDloadSpec] = field(default_factory=list)
    unhandled_specs: List[InpUnhandledStepSpec] = field(default_factory=list)


@dataclass
class AbaqusInpModel:
    part_name: Optional[str] = None
    instance_name: Optional[str] = None
    nodes: Dict[int, InpNode] = field(default_factory=dict)
    element_blocks: List[InpElementBlock] = field(default_factory=list)
    nsets: Dict[str, Dict[str, List[int]]] = field(
        default_factory=lambda: {"part": {}, "assembly": {}}
    )
    elsets: Dict[str, Dict[str, List[int]]] = field(
        default_factory=lambda: {"part": {}, "assembly": {}}
    )
    materials: Dict[str, InpMaterial] = field(default_factory=dict)
    sections: List[InpSection] = field(default_factory=list)
    steps: List[InpStaticStep] = field(default_factory=list)


@dataclass(frozen=True)
class InpModelData:
    model: AbaqusInpModel
    mesh: Any
    boundary: Any
    step: InpStaticStep

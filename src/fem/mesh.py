from dataclasses import dataclass,field
from typing import List,Dict,Protocol,Sequence,runtime_checkable,Any

@dataclass
class Node2D:
    """2D node with id and coordinates."""
    id: int
    x: float
    y: float

@dataclass
class Element2D:
    """2D element with node list, type, and properties."""
    id: int
    node_ids: List[int]
    type: str = "Truss2D"
    props: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Mesh2DProtocol(Protocol):
    """Protocol for 2D FEM meshes."""
    dofs_per_node: int
    num_dofs: int

    # Basic containers
    nodes: List
    elements: List

    # DOF interface (required)
    def global_dof(self, node_id: int, component: int) -> int: ...
    def node_dofs(self, node_id: int) -> Sequence[int]: ...
    def element_dofs(self, elem) -> Sequence[int]: ...

from .dof_manager import DofManager2D
@dataclass
class TrussMesh2D:
    """Truss2D mesh container (ux, uy)."""
    
    nodes: List[Node2D]
    elements: List[Element2D]
    dofs_per_node: int = 2  # Truss: ux, uy

    dof_manager: DofManager2D = field(init=False)

    def __post_init__(self):
        # 初始化 DOF 管理器
        self.dof_manager = DofManager2D.from_nodes(self.nodes, self.dofs_per_node)

    @property
    def num_dofs(self) -> int:
        return self.dof_manager.num_dofs

    @property
    def num_nodes(self) -> int:
        return self.dof_manager.num_nodes

    def global_dof(self, node_id: int, component: int) -> int:
        return self.dof_manager.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        return self.dof_manager.node_dofs(node_id)

    def element_dofs(self, elem: Element2D):
        return self.dof_manager.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        return self.dof_manager.generate_global_dof_sequence()
    
@dataclass
class BeamMesh2D:
    """Beam2D mesh container (ux, uy, rz)."""
    nodes: List[Node2D]
    elements: List[Element2D]
    dofs_per_node: int = 3  # ux, uy, rz

    dof_manager: DofManager2D = field(init=False)

    def __post_init__(self):
        self.dof_manager = DofManager2D.from_nodes(self.nodes, self.dofs_per_node)

    @property
    def num_dofs(self) -> int:
        return self.dof_manager.num_dofs

    @property
    def num_nodes(self) -> int:
        return self.dof_manager.num_nodes

    def global_dof(self, node_id: int, component: int) -> int:
        return self.dof_manager.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        return self.dof_manager.node_dofs(node_id)

    def element_dofs(self, elem: Element2D):
        return self.dof_manager.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        return self.dof_manager.generate_global_dof_sequence()
    
@dataclass
class PlaneMesh2D:
    """Plane mesh container (ux, uy)."""

    nodes: List[Node2D]
    elements: List[Element2D]
    dofs_per_node: int = 2  # ux, uy

    dof_manager: DofManager2D = field(init=False)

    def __post_init__(self):
        self.dof_manager = DofManager2D.from_nodes(self.nodes, self.dofs_per_node)

    @property
    def num_dofs(self) -> int:
        return self.dof_manager.num_dofs

    @property
    def num_nodes(self) -> int:
        return self.dof_manager.num_nodes

    def global_dof(self, node_id: int, component: int) -> int:
        return self.dof_manager.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        return self.dof_manager.node_dofs(node_id)

    def element_dofs(self, elem: Element2D):
        """
        Tri3:  node_ids=[n1,n2,n3]
               -> [ux1,uy1, ux2,uy2, ux3,uy3]  
        Quad4: node_ids=[n1,n2,n3,n4]
               -> [ux1,uy1, ux2,uy2, ux3,uy3, ux4,uy4] 
        Quad8: node_ids=[n1..n8]
               -> [ux1,uy1, ux2,uy2, ux3,uy3, ux4,uy4, ux5,uy5, ux6,uy6, ux7,uy7, ux8,uy8]
        """
        return self.dof_manager.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        return self.dof_manager.generate_global_dof_sequence()

@dataclass
class Node3D:
    """3D node with id and coordinates."""
    id: int
    x: float
    y: float
    z: float

@dataclass
class Element3D:
    """3D element with node list, type, and properties."""
    id: int
    node_ids: List[int]
    type: str = "Hex8"
    props: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Mesh3DProtocol(Protocol):
    """Protocol for 3D FEM meshes."""
    dofs_per_node: int
    num_dofs: int

    # Basic containers
    nodes: List
    elements: List

    # DOF interface (required)
    def global_dof(self, node_id: int, component: int) -> int: ...
    def node_dofs(self, node_id: int) -> Sequence[int]: ...
    def element_dofs(self, elem) -> Sequence[int]: ...

from .dof_manager import DofManager3D
@dataclass
class HexMesh3D:
    """Hexahedral 3D mesh container (ux, uy, uz)."""
    
    nodes: List[Node3D]
    elements: List[Element3D]
    dofs_per_node: int = 3  # ux, uy, uz

    dof_manager: DofManager3D = field(init=False)

    def __post_init__(self):
        self.dof_manager = DofManager3D.from_nodes(self.nodes, self.dofs_per_node)

    @property
    def num_dofs(self) -> int:
        return self.dof_manager.num_dofs

    @property
    def num_nodes(self) -> int:
        return self.dof_manager.num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements."""
        return len(self.elements)

    def global_dof(self, node_id: int, component: int) -> int:
        return self.dof_manager.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        return self.dof_manager.node_dofs(node_id)

    def element_dofs(self, elem: Element3D):
        return self.dof_manager.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        return self.dof_manager.generate_global_dof_sequence()


@dataclass
class TetMesh3D:
    """Tetrahedral 3D mesh container (ux, uy, uz)."""

    nodes: List[Node3D]
    elements: List[Element3D]
    dofs_per_node: int = 3  # ux, uy, uz

    dof_manager: DofManager3D = field(init=False)

    def __post_init__(self):
        self.dof_manager = DofManager3D.from_nodes(self.nodes, self.dofs_per_node)

    @property
    def num_dofs(self) -> int:
        return self.dof_manager.num_dofs

    @property
    def num_nodes(self) -> int:
        return self.dof_manager.num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements."""
        return len(self.elements)

    def global_dof(self, node_id: int, component: int) -> int:
        return self.dof_manager.global_dof(node_id, component)

    def node_dofs(self, node_id: int):
        return self.dof_manager.node_dofs(node_id)

    def element_dofs(self, elem: Element3D):
        return self.dof_manager.element_dofs(elem.node_ids)

    def generate_global_dof_sequence(self):
        return self.dof_manager.generate_global_dof_sequence()
    

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Node2D

@dataclass
class DofManager2D:
    """Generic 2D DOF manager."""
    dofs_per_node: int
    node_ids: List[int]
    node_id_to_index: Dict[int, int]

    @classmethod
    def from_nodes(cls, nodes: List[Node2D], dofs_per_node: int):
        node_ids = sorted(n.id for n in nodes)
        node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        return cls(dofs_per_node, node_ids, node_id_to_index)

    @property
    def num_nodes(self):
        """Number of nodes."""
        return len(self.node_ids)

    @property
    def num_dofs(self):
        """Total number of DOFs."""
        return self.num_nodes * self.dofs_per_node

    def global_dof(self, node_id: int, component: int) -> int:
        """Return global DOF index for a node component (0-based)."""
        idx = self.node_id_to_index[node_id]
        return idx * self.dofs_per_node + component

    def node_dofs(self, node_id: int) -> List[int]:
        """Return global DOF indices for a node."""
        base = self.node_id_to_index[node_id] * self.dofs_per_node
        return [base + i for i in range(self.dofs_per_node)]

    def element_dofs(self, node_ids: List[int]) -> List[int]:
        """Return global DOF indices for element nodes."""
        dofs = []
        for nid in node_ids:
            dofs.extend(self.node_dofs(nid))
        return dofs

    def generate_global_dof_sequence(self) -> List[Tuple[int, int, int]]:
        """Generate (node_id, component, dof_id) tuples."""
        seq = []
        for nid in self.node_ids:
            for comp in range(self.dofs_per_node):
                seq.append((nid, comp, self.global_dof(nid, comp)))
        return seq



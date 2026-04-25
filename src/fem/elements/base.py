from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ElementKernel(Protocol):
    """Element type adapter for stiffness, loads, and stress."""
    type_names: tuple[str, ...]

    def stiffness(
        self,
        mesh: Any,
        elem: Any,
        node_lookup: dict[int, Any] | None = None,
    ) -> np.ndarray:
        """Return element stiffness matrix."""
        ...


def build_node_lookup(mesh: Any) -> dict[int, Any]:
    """Return node lookup keyed by node id."""
    return {node.id: node for node in mesh.nodes}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelResult:
    """Result data for one solved model step."""
    model: Any
    step: Any
    U: np.ndarray
    reactions: np.ndarray
    boundary: Any | None = None
    output_dir: Path | str = Path("results")
    name: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.name is None:
            self.name = _default_result_name(self.model, self.step)

    @property
    def mesh(self) -> Any:
        """Return the solved model mesh."""
        return self.model.mesh

    @property
    def displacement_path(self) -> Path:
        """Default nodal displacement CSV path."""
        return self.output_dir / f"{self.name}_nodal_displacements.csv"

    @property
    def element_stress_path(self) -> Path:
        """Default element stress CSV path."""
        return self.output_dir / f"{self.name}_element_stress.csv"

    @property
    def nodal_stress_path(self) -> Path:
        """Default nodal stress CSV path."""
        return self.output_dir / f"{self.name}_nodal_stress.csv"

    @property
    def reaction_path(self) -> Path:
        """Default reaction CSV path."""
        return self.output_dir / f"{self.name}_reactions.csv"

    @property
    def vtk_path(self) -> Path:
        """Default VTK output path."""
        return self.output_dir / f"{self.name}.vtk"

    def displacement(self, node_id: int, component: int | None = None):
        """Return nodal displacement tuple or one component."""
        values = tuple(float(self.U[dof]) for dof in self.mesh.node_dofs(node_id))
        if component is None:
            return values
        return values[int(component)]

    def reaction(self, node_id: int, component: int | None = None):
        """Return nodal reaction tuple or one component."""
        values = tuple(float(self.reactions[dof]) for dof in self.mesh.node_dofs(node_id))
        if component is None:
            return values
        return values[int(component)]

    def export_displacement(
        self,
        path: str | Path | None = None,
        component_names: list[str] | None = None,
    ) -> None:
        """Export nodal displacements to CSV."""
        from ..post import displacement

        path = self._prepare_path(path or self.displacement_path)
        displacement.export.nodal(self.mesh, self.U, path, component_names)

    def export_element_stress(
        self,
        path: str | Path | None = None,
        element_type: str | None = None,
        gauss_order: int | None = None,
    ) -> None:
        """Export element stresses to CSV."""
        from ..post import stress

        path = self._prepare_path(path or self.element_stress_path)
        stress.export.element(self.mesh, self.U, path, element_type, gauss_order)

    def export_nodal_stress(
        self,
        path: str | Path | None = None,
        element_type: str | None = None,
        gauss_order: int | None = None,
    ) -> None:
        """Export nodal stresses to CSV."""
        from ..post import stress

        path = self._prepare_path(path or self.nodal_stress_path)
        stress.export.nodal(self.mesh, self.U, path, element_type, gauss_order)

    def export_reactions(
        self,
        path: str | Path | None = None,
        component_names: list[str] | None = None,
    ) -> None:
        """Export nodal reactions to CSV."""
        from ..post import displacement

        if component_names is None:
            component_names = _reaction_component_names(self.mesh.dofs_per_node)
        path = self._prepare_path(path or self.reaction_path)
        displacement.export.nodal(self.mesh, self.reactions, path, component_names)

    def export_vtk(
        self,
        path: str | Path | None = None,
        displacement_csv_path: str | Path | None = None,
        element_stress_csv_path: str | Path | None = None,
        nodal_stress_csv_path: str | Path | None = None,
    ) -> None:
        """Export VTK through the existing CSV-based post pipeline."""
        from ..post import vtk

        vtk_path = self._prepare_path(path or self.vtk_path)
        disp_csv = displacement_csv_path or self.displacement_path
        elem_csv = element_stress_csv_path or self.element_stress_path
        nodal_csv = nodal_stress_csv_path or self.nodal_stress_path

        self.export_displacement(disp_csv)
        self.export_element_stress(elem_csv)
        self.export_nodal_stress(nodal_csv)
        vtk.export.from_csv(self.mesh, disp_csv, elem_csv, vtk_path, nodal_csv)

    def _prepare_path(self, path: str | Path) -> Path:
        """Create parent directory and return normalized path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _default_result_name(model: Any, step: Any) -> str:
    """Return a stable default result name."""
    if getattr(model, "name", None):
        return str(model.name)
    if step is not None and getattr(step, "name", None):
        return str(step.name)
    return "result"


def _reaction_component_names(dofs_per_node: int) -> list[str]:
    """Return default reaction component names."""
    if dofs_per_node == 2:
        return ["rx", "ry"]
    if dofs_per_node == 3:
        return ["rx", "ry", "rz"]
    return [f"r{component}" for component in range(dofs_per_node)]

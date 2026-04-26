from __future__ import annotations

from typing import Any, Sequence

from . import dispatch, element as element_export, nodal as nodal_export


def element(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export element stresses to CSV. Element type is inferred for single-type meshes."""
    type_key = dispatch.resolve_type_key(mesh, element_type)
    element_export.by_type(type_key, mesh, U, path, gauss_order)


def nodal(
    mesh: Any,
    U: Sequence[float],
    path: str,
    element_type: str | None = None,
    gauss_order: int | None = None,
) -> None:
    """Export nodal stresses to CSV. Element type is inferred for single-type meshes."""
    type_key = dispatch.resolve_type_key(mesh, element_type)
    nodal_export.by_type(type_key, mesh, U, path, gauss_order)

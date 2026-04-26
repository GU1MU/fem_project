from __future__ import annotations

from typing import Any, Iterable

from ..core.model import ElementSet


def all(mesh: Any) -> list[int]:
    """Return all element ids."""
    return [elem.id for elem in mesh.elements]


def by_type(mesh: Any, element_type: str) -> list[int]:
    """Return element ids whose type contains the requested name."""
    type_key = str(element_type).lower()
    return [elem.id for elem in mesh.elements if type_key in str(elem.type).lower()]


def by_ids(mesh: Any, element_ids: Iterable[int]) -> list[int]:
    """Return existing element ids from a requested id collection."""
    requested = {int(element_id) for element_id in element_ids}
    return [elem.id for elem in mesh.elements if elem.id in requested]


def set_all(mesh: Any, name: str) -> ElementSet:
    """Return a named element set containing all elements."""
    return ElementSet(name, all(mesh))


def set_by_type(mesh: Any, name: str, element_type: str) -> ElementSet:
    """Return a named element set selected by element type."""
    return ElementSet(name, by_type(mesh, element_type))


def set_by_ids(mesh: Any, name: str, element_ids: Iterable[int]) -> ElementSet:
    """Return a named element set selected by ids."""
    return ElementSet(name, by_ids(mesh, element_ids))

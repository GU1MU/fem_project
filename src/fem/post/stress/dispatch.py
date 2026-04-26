from __future__ import annotations

from typing import Any


def resolve_type_key(mesh: Any, element_type: str | None) -> str:
    """Resolve a normalized stress exporter key."""
    if element_type is not None:
        type_key = type_key_from_name(element_type)
        if type_key is None:
            raise ValueError(f"Unsupported stress element type: {element_type!r}")
        return type_key

    type_keys = {type_key_from_name(elem.type) for elem in mesh.elements}
    type_keys.discard(None)
    if not type_keys:
        raise ValueError("Cannot infer stress element type from mesh")
    if len(type_keys) > 1:
        raise ValueError("Mixed element meshes require an explicit element_type")
    return type_keys.pop()


def type_key_from_name(element_type: Any) -> str | None:
    """Normalize mesh element type names to stress exporter keys."""
    etype = str(element_type).lower()
    if "truss" in etype:
        return "truss2d"
    if "tri3" in etype:
        return "tri3"
    if "quad4" in etype:
        return "quad4"
    if "quad8" in etype:
        return "quad8"
    if "hex8" in etype:
        return "hex8"
    if "tet10" in etype:
        return "tet10"
    if "tet4" in etype:
        return "tet4"
    return None

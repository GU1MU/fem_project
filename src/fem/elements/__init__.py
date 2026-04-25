from __future__ import annotations

from .base import ElementKernel
from .hex8 import Hex8Kernel
from .quad4 import Quad4PlaneKernel


_KERNELS: dict[str, ElementKernel] = {}


def register_element_kernel(kernel: ElementKernel) -> None:
    """Register an element kernel for all declared type names."""
    for name in kernel.type_names:
        _KERNELS[name.lower()] = kernel


def get_element_kernel(element_type: str) -> ElementKernel:
    """Return the registered element kernel for an element type."""
    key = str(element_type).lower()
    if key in _KERNELS:
        return _KERNELS[key]

    for name, kernel in _KERNELS.items():
        if name in key:
            return kernel

    raise NotImplementedError(f"Unsupported element type: {element_type}")


register_element_kernel(Quad4PlaneKernel())
register_element_kernel(Hex8Kernel())

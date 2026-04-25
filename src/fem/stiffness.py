from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from .mesh import BeamMesh2D, TrussMesh2D, PlaneMesh2D, Element2D, Node2D, Mesh2DProtocol, HexMesh3D, Element3D, Node3D, Mesh3DProtocol
from .elements import get_element_kernel
from .elements.hex8 import hex8_shape_funcs_grads
from .elements.quad4 import quad4_shape_grad_xi_eta
from .elements.quad8 import quad8_gauss_points, quad8_shape_funcs_grads
from .elements.tet import (
    tet4_gauss_points,
    tet4_shape_funcs_grads,
    tet10_gauss_points,
    tet10_shape_funcs_grads,
)
from .materials import compute_3d_elastic_matrix, compute_plane_strain_matrix, compute_plane_stress_matrix

def _build_node_lookup(mesh: Mesh2DProtocol) -> Dict[int, Node2D]:
    """Build Node2D lookup by node_id."""
    return {node.id: node for node in mesh.nodes}

def _build_node_lookup_3d(mesh: Mesh3DProtocol) -> Dict[int, Node3D]:
    """Build Node3D lookup by node_id."""
    return {node.id: node for node in mesh.nodes}


def compute_truss2d_element_stiffness(
    mesh: TrussMesh2D,
    elem: Element2D,
    node_lookup: Dict[int, Node2D] = None,
) -> np.ndarray:
    """Compute Truss2D element stiffness matrix."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup)


def compute_beam2d_element_stiffness(
    mesh: BeamMesh2D,
    elem: Element2D,
    node_lookup: Dict[int, Node2D] = None,
) -> np.ndarray:
    """Compute Beam2D element stiffness matrix."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup)


def _compute_D_plane_stress(E: float, nu: float) -> np.ndarray:
    """Plane stress constitutive matrix D (3x3)."""
    return compute_plane_stress_matrix(E, nu)


def _compute_D_plane_strain(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix D (3x3)."""
    return compute_plane_strain_matrix(E, nu)


def compute_tri3_plane_element_stiffness(
    mesh: PlaneMesh2D,
    elem: Element2D,
    node_lookup: Dict[int, Node2D] = None,
) -> np.ndarray:
    """Compute Tri3 plane element stiffness matrix."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup)


def _quad4_shape_grad_xi_eta(xi: float, eta: float) -> np.ndarray:
    """Return dN/dxi and dN/deta for bilinear Quad4."""
    return quad4_shape_grad_xi_eta(xi, eta)


def compute_quad4_plane_element_stiffness(
    mesh,
    elem: Element2D,
    node_lookup: Optional[Dict[int, Node2D]] = None,
    gauss_order: int = 2,
) -> np.ndarray:
    """Compute isoparametric Quad4 plane element stiffness."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup, gauss_order)


def _compute_D_3d(E: float, nu: float) -> np.ndarray:
    """3D constitutive matrix D (6x6)."""
    return compute_3d_elastic_matrix(E, nu)

def _hex8_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N, dN/dxi, dN/deta, dN/dzeta for Hex8."""
    return hex8_shape_funcs_grads(xi, eta, zeta)

def compute_hex8_element_stiffness(
    mesh: HexMesh3D,
    elem: Element3D,
    node_lookup: Optional[Dict[int, Node3D]] = None,
    gauss_order: int = 2,
) -> np.ndarray:
    """Compute isoparametric Hex8 element stiffness matrix."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup, gauss_order)


def _tet4_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N and natural gradients for Tet4."""
    return tet4_shape_funcs_grads(xi, eta, zeta)


def _tet4_gauss_points():
    """Return Gauss integration points for Tet4."""
    return tet4_gauss_points()


def compute_tet4_element_stiffness(
    mesh,
    elem,
    node_lookup: Optional[Dict[int, Node3D]] = None,
) -> np.ndarray:
    """Compute isoparametric Tet4 element stiffness matrix (12x12)."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup)


def _tet10_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N and natural gradients for Tet10."""
    return tet10_shape_funcs_grads(xi, eta, zeta)


def _tet10_gauss_points():
    """Return Gauss integration points for Tet10."""
    return tet10_gauss_points()


def compute_tet10_element_stiffness(
    mesh,
    elem,
    node_lookup: Optional[Dict[int, Node3D]] = None,
) -> np.ndarray:
    """Compute isoparametric Tet10 element stiffness matrix (30x30)."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup)


def _quad8_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, dN/deta for serendipity Quad8."""
    return quad8_shape_funcs_grads(xi, eta)


def _quad8_gauss_points(gauss_order: int):
    """Return Gauss points (xi, eta, weight) for Quad8."""
    return quad8_gauss_points(gauss_order)


def compute_quad8_plane_element_stiffness(
    mesh,
    elem: Element2D,
    node_lookup: Optional[Dict[int, Node2D]] = None,
    gauss_order: int = 3,
) -> np.ndarray:
    """Compute isoparametric Quad8 plane element stiffness."""
    return get_element_kernel(elem.type).stiffness(mesh, elem, node_lookup, gauss_order)


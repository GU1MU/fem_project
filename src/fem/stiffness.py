from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from .mesh import BeamMesh2D, TrussMesh2D, PlaneMesh2D, Element2D, Node2D, Mesh2DProtocol, HexMesh3D, Element3D, Node3D, Mesh3DProtocol

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
    try:
        A = float(elem.props["area"])
        E = float(elem.props["E"])
    except KeyError as e:
        raise KeyError(f"元素 {elem.id} 缺少属性 {e.args[0]}，props={elem.props}")

    if len(elem.node_ids) != 2:
        raise ValueError(f"Truss2D 单元必须是 2 节点，elem {elem.id} 有 node_ids={elem.node_ids}")

    n_i, n_j = elem.node_ids

    if node_lookup is None:
        node_lookup = _build_node_lookup(mesh)

    try:
        ni = node_lookup[n_i]
        nj = node_lookup[n_j]
    except KeyError as e:
        raise KeyError(f"在 mesh.nodes 中找不到 id={e.args[0]} 的节点")

    xi, yi = ni.x, ni.y
    xj, yj = nj.x, nj.y

    dx = xj - xi
    dy = yj - yi
    L = (dx**2 + dy**2) ** 0.5
    if L == 0.0:
        raise ValueError(f"单元 {elem.id} 长度为 0，请检查节点坐标")

    c = dx / L
    s = dy / L
    k = E * A / L

    Ke = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ], dtype=float)

    return Ke


def compute_beam2d_element_stiffness(
    mesh: BeamMesh2D,
    elem: Element2D,
    node_lookup: Dict[int, Node2D] = None,
) -> np.ndarray:
    """Compute Beam2D element stiffness matrix."""
    try:
        E = float(elem.props["E"])
        A = float(elem.props["area"])
        I = float(elem.props["Izz"])
    except KeyError as e:
        raise KeyError(
            f"元素 {elem.id} 的 props 缺少 {e.args[0]}，当前 props={elem.props}"
        )

    if len(elem.node_ids) != 2:
        raise ValueError(
            f"Beam2D 单元必须是 2 节点，elem {elem.id} node_ids={elem.node_ids}"
        )

    ni_id, nj_id = elem.node_ids

    if node_lookup is None:
        node_lookup = _build_node_lookup(mesh)

    try:
        ni = node_lookup[ni_id]
        nj = node_lookup[nj_id]
    except KeyError as e:
        raise KeyError(f"在 mesh.nodes 中找不到 id={e.args[0]} 的节点")

    xi, yi = ni.x, ni.y
    xj, yj = nj.x, nj.y

    dx = xj - xi
    dy = yj - yi
    L = (dx**2 + dy**2) ** 0.5
    if L == 0.0:
        raise ValueError(f"Beam 单元 {elem.id} 长度为 0，请检查节点坐标")

    c = dx / L
    s = dy / L

    EA_L = E * A / L
    EI_L3 = E * I / (L**3)
    EI_L2 = E * I / (L**2)
    EI_L = E * I / L

    # 局部自由度顺序：[u1, v1, θ1, u2, v2, θ2]
    k_local = np.array([
        [ EA_L,        0.0,         0.0,   -EA_L,        0.0,         0.0 ],
        [ 0.0,   12*EI_L3,   6*EI_L2,     0.0,  -12*EI_L3,   6*EI_L2 ],
        [ 0.0,    6*EI_L2,    4*EI_L,     0.0,   -6*EI_L2,    2*EI_L  ],
        [-EA_L,       0.0,         0.0,    EA_L,       0.0,         0.0 ],
        [ 0.0,  -12*EI_L3,  -6*EI_L2,     0.0,   12*EI_L3,  -6*EI_L2 ],
        [ 0.0,    6*EI_L2,    2*EI_L,     0.0,   -6*EI_L2,    4*EI_L  ],
    ], dtype=float)

    T = np.array([
        [ c, -s, 0.0,  0.0,  0.0, 0.0],
        [ s,  c, 0.0,  0.0,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0,  0.0, 0.0],
        [0.0, 0.0, 0.0,  c,  -s, 0.0],
        [0.0, 0.0, 0.0,  s,   c, 0.0],
        [0.0, 0.0, 0.0, 0.0,  0.0, 1.0],
    ], dtype=float)

    Ke_global = T.T @ k_local @ T
    return Ke_global


def _compute_D_plane_stress(E: float, nu: float) -> np.ndarray:
    """Plane stress constitutive matrix D (3x3)."""
    coef = E / (1.0 - nu ** 2)
    D = coef * np.array([
        [1.0,    nu,           0.0],
        [nu,     1.0,          0.0],
        [0.0,    0.0, (1.0 - nu) / 2.0],
    ], dtype=float)
    return D


def _compute_D_plane_strain(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix D (3x3)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    D = np.array([
        [lam + 2.0 * mu, lam,            0.0],
        [lam,            lam + 2.0 * mu, 0.0],
        [0.0,            0.0,            mu],
    ], dtype=float)
    return D


def compute_tri3_plane_element_stiffness(
    mesh: PlaneMesh2D,
    elem: Element2D,
    node_lookup: Dict[int, Node2D] = None,
) -> np.ndarray:
    """Compute Tri3 plane element stiffness matrix."""

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
        t = float(elem.props["thickness"])
    except KeyError as e:
        raise KeyError(
            f"元素 {elem.id} 的 props 缺少 {e.args[0]}，当前 props={elem.props}"
        )

    plane_type = elem.props.get("plane_type", "stress")
    plane_type = str(plane_type).lower()

    if plane_type.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
    elif plane_type.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
    else:
        raise ValueError(
            f"元素 {elem.id} 的 plane_type={elem.props.get('plane_type')} "
            f"无法识别，应为 'stress' 或 'strain'"
        )

    if len(elem.node_ids) != 3:
        raise ValueError(
            f"Tri3 单元必须有 3 个节点，elem {elem.id} node_ids={elem.node_ids}"
        )

    n1_id, n2_id, n3_id = elem.node_ids

    if node_lookup is None:
        node_lookup = _build_node_lookup(mesh)

    try:
        n1 = node_lookup[n1_id]
        n2 = node_lookup[n2_id]
        n3 = node_lookup[n3_id]
    except KeyError as e:
        raise KeyError(f"在 mesh.nodes 中找不到 id={e.args[0]} 的节点")

    x1, y1 = n1.x, n1.y
    x2, y2 = n2.x, n2.y
    x3, y3 = n3.x, n3.y

    detJ = (
        x2 * y3 - x3 * y2
        - x1 * y3 + x3 * y1
        + x1 * y2 - x2 * y1
    )
    A = 0.5 * detJ

    if A <= 0.0:
        raise ValueError(
            f"元素 {elem.id} 的面积 A={A} <= 0，请检查节点顺序或是否退化"
        )

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    coef = 1.0 / (2.0 * A)
    B = coef * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3],
    ], dtype=float)

    Ke = t * A * (B.T @ D @ B)

    return Ke


def _quad4_shape_grad_xi_eta(xi: float, eta: float) -> np.ndarray:
    """Return dN/dxi and dN/deta for bilinear Quad4."""
    dN_dxi = np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    ) * 0.25
    dN_deta = np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    ) * 0.25
    return np.vstack([dN_dxi, dN_deta])  # (2,4)


def compute_quad4_plane_element_stiffness(
    mesh,
    elem: Element2D,
    node_lookup: Optional[Dict[int, Node2D]] = None,
    gauss_order: int = 2,
) -> np.ndarray:
    """Compute isoparametric Quad4 plane element stiffness."""
    if len(elem.node_ids) != 4:
        raise ValueError(f"Quad4 needs 4 nodes, elem {elem.id} node_ids={elem.node_ids}")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
        t = float(elem.props.get("thickness", 1.0))
    except KeyError as e:
        raise KeyError(f"elem {elem.id} missing '{e.args[0]}' in props={elem.props}")

    pt = str(elem.props.get("plane_type", "stress")).lower()
    if pt.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
    elif pt.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
    else:
        raise ValueError(f"elem {elem.id} invalid plane_type={elem.props.get('plane_type')}")

    if node_lookup is None:
        node_lookup = _build_node_lookup(mesh.nodes)

    nids = elem.node_ids
    n1, n2, n3, n4 = (node_lookup[i] for i in nids)
    x = np.array([n1.x, n2.x, n3.x, n4.x], dtype=float)
    y = np.array([n1.y, n2.y, n3.y, n4.y], dtype=float)

    if gauss_order == 1:
        gps = [(0.0, 0.0, 4.0)]
    elif gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        gps = [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    else:
        raise ValueError("gauss_order must be 1 or 2")

    Ke = np.zeros((8, 8), dtype=float)

    for xi, eta, w in gps:
        dN = _quad4_shape_grad_xi_eta(xi, eta)  # (2,4)

        J = np.array(
            [[np.dot(dN[0], x), np.dot(dN[0], y)],
             [np.dot(dN[1], x), np.dot(dN[1], y)]],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError(f"elem {elem.id} singular Jacobian")

        invJ = np.linalg.inv(J)
        dN_xy = invJ @ dN  # (2,4)

        B = np.zeros((3, 8), dtype=float)
        for a_i in range(4):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        Ke += (B.T @ D @ B) * (t * detJ * w)

    return Ke


def _compute_D_3d(E: float, nu: float) -> np.ndarray:
    """3D constitutive matrix D (6x6)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    D = np.array([
        [lam + 2.0 * mu, lam,            lam,            0.0, 0.0, 0.0],
        [lam,            lam + 2.0 * mu, lam,            0.0, 0.0, 0.0],
        [lam,            lam,            lam + 2.0 * mu, 0.0, 0.0, 0.0],
        [0.0,            0.0,            0.0,            mu,  0.0, 0.0],
        [0.0,            0.0,            0.0,            0.0, mu,  0.0],
        [0.0,            0.0,            0.0,            0.0, 0.0, mu],
    ], dtype=float)
    return D

def _hex8_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N, dN/dxi, dN/deta, dN/dzeta for Hex8."""
    N = np.zeros(8, dtype=float)
    dN_dxi = np.zeros(8, dtype=float)
    dN_deta = np.zeros(8, dtype=float)
    dN_dzeta = np.zeros(8, dtype=float)

    # Shape functions for Hex8
    N[0] = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0
    N[1] = (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0
    N[2] = (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0
    N[3] = (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0
    N[4] = (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0
    N[5] = (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0
    N[6] = (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0
    N[7] = (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0

    # Derivatives
    dN_dxi[0] = -(1.0 - eta) * (1.0 - zeta) / 8.0
    dN_dxi[1] = (1.0 - eta) * (1.0 - zeta) / 8.0
    dN_dxi[2] = (1.0 + eta) * (1.0 - zeta) / 8.0
    dN_dxi[3] = -(1.0 + eta) * (1.0 - zeta) / 8.0
    dN_dxi[4] = -(1.0 - eta) * (1.0 + zeta) / 8.0
    dN_dxi[5] = (1.0 - eta) * (1.0 + zeta) / 8.0
    dN_dxi[6] = (1.0 + eta) * (1.0 + zeta) / 8.0
    dN_dxi[7] = -(1.0 + eta) * (1.0 + zeta) / 8.0

    dN_deta[0] = -(1.0 - xi) * (1.0 - zeta) / 8.0
    dN_deta[1] = -(1.0 + xi) * (1.0 - zeta) / 8.0
    dN_deta[2] = (1.0 + xi) * (1.0 - zeta) / 8.0
    dN_deta[3] = (1.0 - xi) * (1.0 - zeta) / 8.0
    dN_deta[4] = -(1.0 - xi) * (1.0 + zeta) / 8.0
    dN_deta[5] = -(1.0 + xi) * (1.0 + zeta) / 8.0
    dN_deta[6] = (1.0 + xi) * (1.0 + zeta) / 8.0
    dN_deta[7] = (1.0 - xi) * (1.0 + zeta) / 8.0

    dN_dzeta[0] = -(1.0 - xi) * (1.0 - eta) / 8.0
    dN_dzeta[1] = -(1.0 + xi) * (1.0 - eta) / 8.0
    dN_dzeta[2] = -(1.0 + xi) * (1.0 + eta) / 8.0
    dN_dzeta[3] = -(1.0 - xi) * (1.0 + eta) / 8.0
    dN_dzeta[4] = (1.0 - xi) * (1.0 - eta) / 8.0
    dN_dzeta[5] = (1.0 + xi) * (1.0 - eta) / 8.0
    dN_dzeta[6] = (1.0 + xi) * (1.0 + eta) / 8.0
    dN_dzeta[7] = (1.0 - xi) * (1.0 + eta) / 8.0

    return N, dN_dxi, dN_deta, dN_dzeta

def compute_hex8_element_stiffness(
    mesh: HexMesh3D,
    elem: Element3D,
    node_lookup: Optional[Dict[int, Node3D]] = None,
    gauss_order: int = 2,
) -> np.ndarray:
    """Compute isoparametric Hex8 element stiffness matrix."""
    if len(elem.node_ids) != 8:
        raise ValueError(f"Hex8 element must have 8 nodes, got {len(elem.node_ids)}")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"Element {elem.id} missing property {e.args[0]}, props={elem.props}")

    D = _compute_D_3d(E, nu)

    if node_lookup is None:
        node_lookup = _build_node_lookup_3d(mesh)

    nids = elem.node_ids
    nodes = [node_lookup[nid] for nid in nids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    z = np.array([n.z for n in nodes], dtype=float)

    # Gauss points for Hex8 (2x2x2 = 8 points)
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        gps = [
            (-a, -a, -a, 1.0),
            (a, -a, -a, 1.0),
            (a, a, -a, 1.0),
            (-a, a, -a, 1.0),
            (-a, -a, a, 1.0),
            (a, -a, a, 1.0),
            (a, a, a, 1.0),
            (-a, a, a, 1.0),
        ]
    else:
        raise ValueError(f"Unsupported gauss_order {gauss_order}, only 2 supported")

    Ke = np.zeros((24, 24), dtype=float)  # 8 nodes * 3 DOFs

    for xi, eta, zeta, w in gps:
        N, dN_dxi, dN_deta, dN_dzeta = _hex8_shape_funcs_grads(xi, eta, zeta)

        # Jacobian matrix
        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)

        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Element {elem.id} has negative or zero Jacobian determinant")

        invJ = np.linalg.inv(J)

        # Derivatives in physical coordinates
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
        dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta

        # Strain-displacement matrix B (6 strains, 24 DOFs)
        B = np.zeros((6, 24), dtype=float)
        for i in range(8):
            idx = 3 * i
            B[0, idx] = dN_dx[i]      # ε_xx
            B[1, idx + 1] = dN_dy[i]  # ε_yy
            B[2, idx + 2] = dN_dz[i]  # ε_zz
            B[3, idx] = dN_dy[i]      # γ_xy
            B[3, idx + 1] = dN_dx[i]
            B[4, idx + 1] = dN_dz[i]  # γ_yz
            B[4, idx + 2] = dN_dy[i]
            B[5, idx] = dN_dz[i]      # γ_zx
            B[5, idx + 2] = dN_dx[i]

        Ke += (B.T @ D @ B) * detJ * w

    return Ke


def _tet4_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N, dN/dxi, dN/deta, dN/dzeta for linear Tet4.

    Natural coordinates (volume coordinates):
        N1 = 1 - xi - eta - zeta
        N2 = xi
        N3 = eta
        N4 = zeta
    """
    N = np.array([
        1.0 - xi - eta - zeta,
        xi,
        eta,
        zeta,
    ], dtype=float)

    dN_dxi = np.array([-1.0, 1.0, 0.0, 0.0], dtype=float)
    dN_deta = np.array([-1.0, 0.0, 1.0, 0.0], dtype=float)
    dN_dzeta = np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)

    return N, dN_dxi, dN_deta, dN_dzeta


def _tet4_gauss_points():
    """Return Gauss integration points for Tet4 (single point at centroid).

    For linear Tet4, the stiffness matrix can be exactly integrated
    with a single point at (1/4, 1/4, 1/4) with weight 1/6.
    """
    return [(0.25, 0.25, 0.25, 1.0 / 6.0)]


def compute_tet4_element_stiffness(
    mesh,
    elem,
    node_lookup: Optional[Dict[int, Node3D]] = None,
) -> np.ndarray:
    """Compute isoparametric Tet4 element stiffness matrix (12x12)."""
    if len(elem.node_ids) != 4:
        raise ValueError(f"Tet4 element must have 4 nodes, got {len(elem.node_ids)}")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"Element {elem.id} missing property {e.args[0]}, props={elem.props}")

    D = _compute_D_3d(E, nu)

    if node_lookup is None:
        node_lookup = _build_node_lookup_3d(mesh)

    nids = elem.node_ids
    nodes = [node_lookup[nid] for nid in nids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    z = np.array([n.z for n in nodes], dtype=float)

    gps = _tet4_gauss_points()
    Ke = np.zeros((12, 12), dtype=float)  # 4 nodes * 3 DOFs

    for xi, eta, zeta, w in gps:
        N, dN_dxi, dN_deta, dN_dzeta = _tet4_shape_funcs_grads(xi, eta, zeta)

        # Jacobian matrix J = d(x,y,z)/d(xi,eta,zeta)
        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)

        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Element {elem.id} has negative or zero Jacobian determinant")

        invJ = np.linalg.inv(J)

        # Derivatives in physical coordinates: dN/dx = invJ * dN/d(xi,eta,zeta)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
        dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta

        # Strain-displacement matrix B (6 strains x 12 DOFs)
        B = np.zeros((6, 12), dtype=float)
        for i in range(4):
            idx = 3 * i
            B[0, idx] = dN_dx[i]       # eps_xx
            B[1, idx + 1] = dN_dy[i]   # eps_yy
            B[2, idx + 2] = dN_dz[i]   # eps_zz
            B[3, idx] = dN_dy[i]       # gamma_xy
            B[3, idx + 1] = dN_dx[i]
            B[4, idx + 1] = dN_dz[i]   # gamma_yz
            B[4, idx + 2] = dN_dy[i]
            B[5, idx] = dN_dz[i]       # gamma_zx
            B[5, idx + 2] = dN_dx[i]

        Ke += (B.T @ D @ B) * detJ * w

    return Ke


def _tet10_shape_funcs_grads(xi: float, eta: float, zeta: float):
    """Return N, dN/dxi, dN/deta, dN/dzeta for quadratic Tet10.

    Node ordering (Abaqus convention):
        Corner nodes:  1=(0,0,0), 2=(1,0,0), 3=(0,1,0), 4=(0,0,1)
        Edge midnodes: 5=edge(1,2), 6=edge(2,3), 7=edge(3,1),
                       8=edge(1,4), 9=edge(2,4), 10=edge(3,4)

    Shape functions (natural coords L1=1-xi-eta-zeta, L2=xi, L3=eta, L4=zeta):
        N_corner_i = (2*L_i - 1) * L_i
        N_edge_5  = 4*L1*L2   (edge 1-2)
        N_edge_6  = 4*L2*L3   (edge 2-3)
        N_edge_7  = 4*L1*L3   (edge 3-1)
        N_edge_8  = 4*L1*L4   (edge 1-4)
        N_edge_9  = 4*L2*L4   (edge 2-4)
        N_edge_10 = 4*L3*L4   (edge 3-4)
    """
    L1 = 1.0 - xi - eta - zeta
    L2 = xi
    L3 = eta
    L4 = zeta

    N = np.zeros(10, dtype=float)
    dN_dxi = np.zeros(10, dtype=float)
    dN_deta = np.zeros(10, dtype=float)
    dN_dzeta = np.zeros(10, dtype=float)

    # Corner nodes
    N[0] = (2.0 * L1 - 1.0) * L1
    N[1] = (2.0 * L2 - 1.0) * L2
    N[2] = (2.0 * L3 - 1.0) * L3
    N[3] = (2.0 * L4 - 1.0) * L4

    # Edge midnodes
    N[4] = 4.0 * L1 * L2   # edge 1-2
    N[5] = 4.0 * L2 * L3   # edge 2-3
    N[6] = 4.0 * L1 * L3   # edge 3-1
    N[7] = 4.0 * L1 * L4   # edge 1-4
    N[8] = 4.0 * L2 * L4   # edge 2-4
    N[9] = 4.0 * L3 * L4   # edge 3-4

    # Derivatives of volume coordinates
    dL1_dxi = -1.0; dL1_deta = -1.0; dL1_dzeta = -1.0
    dL2_dxi = 1.0;  dL2_deta = 0.0;  dL2_dzeta = 0.0
    dL3_dxi = 0.0;  dL3_deta = 1.0;  dL3_dzeta = 0.0
    dL4_dxi = 0.0;  dL4_deta = 0.0;  dL4_dzeta = 1.0

    # Corner node derivatives: N_i = (2*L_i - 1)*L_i, dN_i/dL = 4*L_i - 1
    dN_dxi[0]   = (4.0 * L1 - 1.0) * dL1_dxi
    dN_deta[0]  = (4.0 * L1 - 1.0) * dL1_deta
    dN_dzeta[0] = (4.0 * L1 - 1.0) * dL1_dzeta

    dN_dxi[1]   = (4.0 * L2 - 1.0) * dL2_dxi
    dN_deta[1]  = (4.0 * L2 - 1.0) * dL2_deta
    dN_dzeta[1] = (4.0 * L2 - 1.0) * dL2_dzeta

    dN_dxi[2]   = (4.0 * L3 - 1.0) * dL3_dxi
    dN_deta[2]  = (4.0 * L3 - 1.0) * dL3_deta
    dN_dzeta[2] = (4.0 * L3 - 1.0) * dL3_dzeta

    dN_dxi[3]   = (4.0 * L4 - 1.0) * dL4_dxi
    dN_deta[3]  = (4.0 * L4 - 1.0) * dL4_deta
    dN_dzeta[3] = (4.0 * L4 - 1.0) * dL4_dzeta

    # Edge midnode derivatives: N_5 = 4*L1*L2, dN_5/dL1=4*L2, dN_5/dL2=4*L1
    # Edge 5: 4*L1*L2
    dN_dxi[4]   = 4.0 * (L2 * dL1_dxi + L1 * dL2_dxi)
    dN_deta[4]  = 4.0 * (L2 * dL1_deta + L1 * dL2_deta)
    dN_dzeta[4] = 4.0 * (L2 * dL1_dzeta + L1 * dL2_dzeta)

    # Edge 6: 4*L2*L3
    dN_dxi[5]   = 4.0 * (L3 * dL2_dxi + L2 * dL3_dxi)
    dN_deta[5]  = 4.0 * (L3 * dL2_deta + L2 * dL3_deta)
    dN_dzeta[5] = 4.0 * (L3 * dL2_dzeta + L2 * dL3_dzeta)

    # Edge 7: 4*L1*L3
    dN_dxi[6]   = 4.0 * (L3 * dL1_dxi + L1 * dL3_dxi)
    dN_deta[6]  = 4.0 * (L3 * dL1_deta + L1 * dL3_deta)
    dN_dzeta[6] = 4.0 * (L3 * dL1_dzeta + L1 * dL3_dzeta)

    # Edge 8: 4*L1*L4
    dN_dxi[7]   = 4.0 * (L4 * dL1_dxi + L1 * dL4_dxi)
    dN_deta[7]  = 4.0 * (L4 * dL1_deta + L1 * dL4_deta)
    dN_dzeta[7] = 4.0 * (L4 * dL1_dzeta + L1 * dL4_dzeta)

    # Edge 9: 4*L2*L4
    dN_dxi[8]   = 4.0 * (L4 * dL2_dxi + L2 * dL4_dxi)
    dN_deta[8]  = 4.0 * (L4 * dL2_deta + L2 * dL4_deta)
    dN_dzeta[8] = 4.0 * (L4 * dL2_dzeta + L2 * dL4_dzeta)

    # Edge 10: 4*L3*L4
    dN_dxi[9]   = 4.0 * (L4 * dL3_dxi + L3 * dL4_dxi)
    dN_deta[9]  = 4.0 * (L4 * dL3_deta + L3 * dL4_deta)
    dN_dzeta[9] = 4.0 * (L4 * dL3_dzeta + L3 * dL4_dzeta)

    return N, dN_dxi, dN_deta, dN_dzeta


def _tet10_gauss_points():
    """Return Gauss integration points for Tet10 (4-point rule).

    Uses the 4-point tetrahedral rule with points at:
        (a, a, a)       w = 1/24
        (b, a, a)       w = 1/24
        (a, b, a)       w = 1/24
        (a, a, b)       w = 1/24
    where a = (1-n)/4, b = (1+3n)/4, n = 0.58541020 (Hammer point)
    """
    n = 0.58541020
    a = (1.0 - n) / 4.0
    b = (1.0 + 3.0 * n) / 4.0
    w = 1.0 / 24.0
    return [
        (a, a, a, w),
        (b, a, a, w),
        (a, b, a, w),
        (a, a, b, w),
    ]


def compute_tet10_element_stiffness(
    mesh,
    elem,
    node_lookup: Optional[Dict[int, Node3D]] = None,
) -> np.ndarray:
    """Compute isoparametric Tet10 element stiffness matrix (30x30).

    10-node quadratic tetrahedron with 3 DOFs per node (ux, uy, uz).
    """
    if len(elem.node_ids) != 10:
        raise ValueError(f"Tet10 element must have 10 nodes, got {len(elem.node_ids)}")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"Element {elem.id} missing property {e.args[0]}, props={elem.props}")

    D = _compute_D_3d(E, nu)

    if node_lookup is None:
        node_lookup = _build_node_lookup_3d(mesh)

    nids = elem.node_ids
    nodes = [node_lookup[nid] for nid in nids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    z = np.array([n.z for n in nodes], dtype=float)

    gps = _tet10_gauss_points()
    Ke = np.zeros((30, 30), dtype=float)  # 10 nodes * 3 DOFs

    for xi, eta, zeta, w in gps:
        N, dN_dxi, dN_deta, dN_dzeta = _tet10_shape_funcs_grads(xi, eta, zeta)

        # Jacobian matrix J = d(x,y,z)/d(xi,eta,zeta)
        J = np.array([
            [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
            [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
            [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
        ], dtype=float)

        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"Element {elem.id} has negative or zero Jacobian determinant")

        invJ = np.linalg.inv(J)

        # Derivatives in physical coordinates
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta + invJ[0, 2] * dN_dzeta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta + invJ[1, 2] * dN_dzeta
        dN_dz = invJ[2, 0] * dN_dxi + invJ[2, 1] * dN_deta + invJ[2, 2] * dN_dzeta

        # Strain-displacement matrix B (6 strains x 30 DOFs)
        B = np.zeros((6, 30), dtype=float)
        for i in range(10):
            idx = 3 * i
            B[0, idx] = dN_dx[i]       # eps_xx
            B[1, idx + 1] = dN_dy[i]   # eps_yy
            B[2, idx + 2] = dN_dz[i]   # eps_zz
            B[3, idx] = dN_dy[i]       # gamma_xy
            B[3, idx + 1] = dN_dx[i]
            B[4, idx + 1] = dN_dz[i]   # gamma_yz
            B[4, idx + 2] = dN_dy[i]
            B[5, idx] = dN_dz[i]       # gamma_zx
            B[5, idx + 2] = dN_dx[i]

        Ke += (B.T @ D @ B) * detJ * w

    return Ke


def _quad8_shape_funcs_grads(xi: float, eta: float):
    """Return N, dN/dxi, dN/deta for serendipity Quad8."""
    N = np.zeros(8, dtype=float)
    dN_dxi = np.zeros(8, dtype=float)
    dN_deta = np.zeros(8, dtype=float)

    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)

    dN_dxi[0] = 0.25 * (-(1.0 - eta) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_dxi[1] = 0.25 * ((1.0 - eta) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (1.0))
    dN_dxi[2] = 0.25 * ((1.0 + eta) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_dxi[3] = 0.25 * (-(1.0 + eta) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (-1.0))
    dN_dxi[4] = -xi * (1.0 - eta)
    dN_dxi[5] = 0.5 * (1.0 - eta * eta)
    dN_dxi[6] = -xi * (1.0 + eta)
    dN_dxi[7] = -0.5 * (1.0 - eta * eta)

    dN_deta[0] = 0.25 * (-(1.0 - xi) * (-xi - eta - 1.0) + (1.0 - xi) * (1.0 - eta) * (-1.0))
    dN_deta[1] = 0.25 * (-(1.0 + xi) * (xi - eta - 1.0) + (1.0 + xi) * (1.0 - eta) * (-1.0))
    dN_deta[2] = 0.25 * ((1.0 + xi) * (xi + eta - 1.0) + (1.0 + xi) * (1.0 + eta) * (1.0))
    dN_deta[3] = 0.25 * ((1.0 - xi) * (-xi + eta - 1.0) + (1.0 - xi) * (1.0 + eta) * (1.0))
    dN_deta[4] = -0.5 * (1.0 - xi * xi)
    dN_deta[5] = -(1.0 + xi) * eta
    dN_deta[6] = 0.5 * (1.0 - xi * xi)
    dN_deta[7] = -(1.0 - xi) * eta

    return N, dN_dxi, dN_deta


def _quad8_gauss_points(gauss_order: int):
    """Return Gauss points (xi, eta, weight) for Quad8."""
    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    if gauss_order == 3:
        r = np.sqrt(3.0 / 5.0)
        one_d = [(-r, 5.0 / 9.0), (0.0, 8.0 / 9.0), (r, 5.0 / 9.0)]
        pts = []
        for xi, wx in one_d:
            for eta, wy in one_d:
                pts.append((xi, eta, wx * wy))
        return pts
    raise ValueError("gauss_order must be 2 or 3 for Quad8")


def compute_quad8_plane_element_stiffness(
    mesh,
    elem: Element2D,
    node_lookup: Optional[Dict[int, Node2D]] = None,
    gauss_order: int = 3,
) -> np.ndarray:
    """Compute isoparametric Quad8 plane element stiffness."""
    if len(elem.node_ids) != 8:
        raise ValueError(f"Quad8 needs 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
        t = float(elem.props.get("thickness", 1.0))
    except KeyError as e:
        raise KeyError(f"elem {elem.id} missing '{e.args[0]}' in props={elem.props}")

    pt = str(elem.props.get("plane_type", "stress")).lower()
    if pt.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
    elif pt.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
    else:
        raise ValueError(f"elem {elem.id} invalid plane_type={elem.props.get('plane_type')}")

    if node_lookup is None:
        node_lookup = {n.id: n for n in mesh.nodes}

    nodes = [node_lookup[i] for i in elem.node_ids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)

    gps = _quad8_gauss_points(gauss_order)
    Ke = np.zeros((16, 16), dtype=float)

    for xi, eta, w in gps:
        _, dN_dxi, dN_deta = _quad8_shape_funcs_grads(xi, eta)

        J = np.array(
            [[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
             [np.dot(dN_deta, x), np.dot(dN_deta, y)]],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError(f"elem {elem.id} singular Jacobian")

        invJ = np.linalg.inv(J)
        dN_xy = invJ @ np.vstack([dN_dxi, dN_deta])  # (2,8)

        B = np.zeros((3, 16), dtype=float)
        for a_i in range(8):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        Ke += (B.T @ D @ B) * (t * detJ * w)

    return Ke


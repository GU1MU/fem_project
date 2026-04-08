from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from .mesh import BeamMesh2D, TrussMesh2D, PlaneMesh2D, Element2D, Node2D, Mesh2DProtocol

def _build_node_lookup(mesh: Mesh2DProtocol) -> Dict[int, Node2D]:
    """Build Node2D lookup by node_id."""
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


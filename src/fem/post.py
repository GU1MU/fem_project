import csv
from typing import Sequence, Optional, List, Dict
import numpy as np
from .mesh import Mesh2DProtocol, TrussMesh2D, PlaneMesh2D, Node2D, Mesh3DProtocol, HexMesh3D, Node3D


def export_nodal_displacements_csv(
    mesh: Mesh2DProtocol,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 2:
            component_names = ["ux", "uy"]
        elif dofs_per_node == 3:
            component_names = ["ux", "uy", "uz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y"] + component_names

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.dof_manager.node_ids:
            node: Node2D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y] + disp_vals)


def export_truss2d_element_stress_csv(
    mesh: TrussMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Truss2D element axial strain/stress and mises to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elem_id",
            "node_i",
            "node_j",
            "axial_strain",
            "axial_stress",
            "mises_stress",
        ])

        for elem in mesh.elements:
            if len(elem.node_ids) != 2:
                raise ValueError(
                    f"Truss2D elem must have 2 nodes, elem {elem.id} node_ids={elem.node_ids}"
                )
            ni_id, nj_id = elem.node_ids

            ni: Node2D = node_lookup[ni_id]
            nj: Node2D = node_lookup[nj_id]

            dx = nj.x - ni.x
            dy = nj.y - ni.y
            L = (dx**2 + dy**2) ** 0.5
            if L == 0.0:
                raise ValueError(f"elem {elem.id} has zero length")

            c = dx / L
            s = dy / L

            try:
                E = float(elem.props["E"])
            except KeyError:
                raise KeyError(f"elem {elem.id} missing E in props={elem.props}")

            dof_ix = mesh.global_dof(ni_id, 0)
            dof_iy = mesh.global_dof(ni_id, 1)
            dof_jx = mesh.global_dof(nj_id, 0)
            dof_jy = mesh.global_dof(nj_id, 1)

            uix = U[dof_ix]
            uiy = U[dof_iy]
            ujx = U[dof_jx]
            ujy = U[dof_jy]

            u_i_L = c * uix + s * uiy
            u_j_L = c * ujx + s * ujy

            axial_strain = (u_j_L - u_i_L) / L
            axial_stress = E * axial_strain
            mises_stress = abs(axial_stress)

            writer.writerow([
                elem.id,
                ni_id,
                nj_id,
                axial_strain,
                axial_stress,
                mises_stress,
            ])


def _compute_D_plane_stress(E: float, nu: float) -> np.ndarray:
    """Plane stress D matrix."""
    coef = E / (1.0 - nu ** 2)
    return coef * np.array([
        [1.0,    nu,           0.0],
        [nu,     1.0,          0.0],
        [0.0,    0.0, (1.0 - nu) / 2.0],
    ], dtype=float)


def _compute_D_plane_strain(E: float, nu: float) -> np.ndarray:
    """Plane strain D matrix."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array([
        [lam + 2.0 * mu, lam,            0.0],
        [lam,            lam + 2.0 * mu, 0.0],
        [0.0,            0.0,            mu],
    ], dtype=float)


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
    return np.vstack([dN_dxi, dN_deta])


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


def _lagrange_weights_1d(points, x):
    """Return Lagrange weights at x."""
    weights = []
    for i, xi in enumerate(points):
        w = 1.0
        for j, xj in enumerate(points):
            if i != j:
                w *= (x - xj) / (xi - xj)
        weights.append(w)
    return weights


def _extrapolate_from_gp(gp_vals, xi_pts, eta_pts, node_coords):
    """Extrapolate GP values to nodes using tensor Lagrange."""
    n_eta = len(eta_pts)
    node_vals = []
    for xi_n, eta_n in node_coords:
        wx = _lagrange_weights_1d(xi_pts, xi_n)
        wy = _lagrange_weights_1d(eta_pts, eta_n)
        val = np.zeros(gp_vals.shape[1], dtype=float)
        for i in range(len(xi_pts)):
            for j in range(len(eta_pts)):
                idx = i * n_eta + j
                val += gp_vals[idx] * (wx[i] * wy[j])
        node_vals.append(val)
    return np.array(node_vals)


def _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order):
    """Return Quad4 nodal stress by extrapolation."""
    if gauss_order != 2:
        raise ValueError("gauss_order must be 2 for Quad4 extrapolation")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"elem {elem.id} missing '{e.args[0]}' in props={elem.props}")

    pt = str(elem.props.get("plane_type", "stress")).lower()
    if pt.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
        plane_type = "stress"
    elif pt.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
        plane_type = "strain"
    else:
        raise ValueError(f"elem {elem.id} invalid plane_type={elem.props.get('plane_type')}")

    nodes = [node_lookup[i] for i in elem.node_ids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    dofs = mesh.element_dofs(elem)
    u_e = U[dofs]

    a = 1.0 / np.sqrt(3.0)
    xi_pts = [-a, a]
    eta_pts = [-a, a]
    gps = [(xi, eta) for xi in xi_pts for eta in eta_pts]

    gp_sigmas = []
    for xi, eta in gps:
        dN = _quad4_shape_grad_xi_eta(xi, eta)
        J = np.array(
            [[np.dot(dN[0], x), np.dot(dN[0], y)],
             [np.dot(dN[1], x), np.dot(dN[1], y)]],
            dtype=float,
        )
        detJ = float(np.linalg.det(J))
        if detJ == 0.0:
            raise ValueError(f"elem {elem.id} singular Jacobian")
        invJ = np.linalg.inv(J)
        dN_xy = invJ @ dN

        B = np.zeros((3, 8), dtype=float)
        for a_i in range(4):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        eps = B @ u_e
        sigma = D @ eps
        gp_sigmas.append(sigma)

    gp_vals = np.vstack(gp_sigmas)
    node_coords = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    node_vals = _extrapolate_from_gp(gp_vals, xi_pts, eta_pts, node_coords)
    return node_vals, plane_type, nu


def _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order):
    """Return Quad8 nodal stress by extrapolation."""
    if gauss_order not in (2, 3):
        raise ValueError("gauss_order must be 2 or 3 for Quad8 extrapolation")

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"elem {elem.id} missing '{e.args[0]}' in props={elem.props}")

    pt = str(elem.props.get("plane_type", "stress")).lower()
    if pt.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
        plane_type = "stress"
    elif pt.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
        plane_type = "strain"
    else:
        raise ValueError(f"elem {elem.id} invalid plane_type={elem.props.get('plane_type')}")

    nodes = [node_lookup[i] for i in elem.node_ids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    dofs = mesh.element_dofs(elem)
    u_e = U[dofs]

    if gauss_order == 2:
        a = 1.0 / np.sqrt(3.0)
        xi_pts = [-a, a]
        eta_pts = [-a, a]
    else:
        r = np.sqrt(3.0 / 5.0)
        xi_pts = [-r, 0.0, r]
        eta_pts = [-r, 0.0, r]

    gps = [(xi, eta) for xi in xi_pts for eta in eta_pts]

    gp_sigmas = []
    for xi, eta in gps:
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
        dN_xy = invJ @ np.vstack([dN_dxi, dN_deta])

        B = np.zeros((3, 16), dtype=float)
        for a_i in range(8):
            dN_dx = dN_xy[0, a_i]
            dN_dy = dN_xy[1, a_i]
            c = 2 * a_i
            B[0, c] = dN_dx
            B[1, c + 1] = dN_dy
            B[2, c] = dN_dy
            B[2, c + 1] = dN_dx

        eps = B @ u_e
        sigma = D @ eps
        gp_sigmas.append(sigma)

    gp_vals = np.vstack(gp_sigmas)
    node_coords = [
        (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
        (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
    ]
    node_vals = _extrapolate_from_gp(gp_vals, xi_pts, eta_pts, node_coords)
    return node_vals, plane_type, nu


def _tri3_element_stress(mesh, elem, U, node_lookup):
    """Return Tri3 constant stress and plane type."""
    if len(elem.node_ids) != 3:
        raise ValueError(
            f"Tri3 elem must have 3 nodes, elem {elem.id} node_ids={elem.node_ids}"
        )

    n1_id, n2_id, n3_id = elem.node_ids
    n1: Node2D = node_lookup[n1_id]
    n2: Node2D = node_lookup[n2_id]
    n3: Node2D = node_lookup[n3_id]

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
        raise ValueError(f"elem {elem.id} has non-positive area")

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

    dofs = mesh.element_dofs(elem)
    u_e = U[dofs]
    eps = B @ u_e

    try:
        E = float(elem.props["E"])
        nu = float(elem.props["nu"])
    except KeyError as e:
        raise KeyError(f"elem {elem.id} missing {e.args[0]} in props={elem.props}")

    plane_type = str(elem.props.get("plane_type", "stress")).lower()
    if plane_type.startswith("stress"):
        D = _compute_D_plane_stress(E, nu)
        plane_tag = "stress"
    elif plane_type.startswith("strain"):
        D = _compute_D_plane_strain(E, nu)
        plane_tag = "strain"
    else:
        raise ValueError(f"elem {elem.id} invalid plane_type")

    sigma = D @ eps
    return sigma, plane_tag, nu


def export_tri3_plane_element_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tri3 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            if not str(elem.type).lower().startswith("tri3"):
                continue

            sigma, plane_type, nu = _tri3_element_stress(mesh, elem, U, node_lookup)
            sig_x, sig_y, tau_xy = sigma.tolist()

            if plane_type == "stress":
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
            else:
                sig_z = nu * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )

            for i, nid in enumerate(elem.node_ids, start=1):
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_tri3_nodal_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Tri3 nodal stresses averaged from elements."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        if not str(elem.type).lower().startswith("tri3"):
            continue

        sigma, pt, nu = _tri3_element_stress(mesh, elem, U, node_lookup)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for nid in elem.node_ids:
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + sigma
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.dof_manager.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def export_quad4_plane_element_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            etype = str(elem.type).lower()
            if "quad4" not in etype:
                continue
            if len(elem.node_ids) != 4:
                raise ValueError(f"Quad4 elem must have 4 nodes, elem {elem.id} node_ids={elem.node_ids}")

            node_vals, plane_type, nu = _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)

            for i, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[i - 1].tolist()
                if plane_type == "stress":
                    mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
                else:
                    sig_z = nu * (sig_x + sig_y)
                    mises = np.sqrt(
                        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                        + 3.0 * tau_xy**2
                    )
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_quad8_plane_element_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 element-nodal stresses without averaging."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["elem_id", "node_id", "local_node", "sig_x", "sig_y", "tau_xy", "mises"])

        for elem in mesh.elements:
            etype = str(elem.type).lower()
            if "quad8" not in etype:
                continue
            if len(elem.node_ids) != 8:
                raise ValueError(f"Quad8 elem must have 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

            node_vals, plane_type, nu = _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)

            for i, nid in enumerate(elem.node_ids, start=1):
                sig_x, sig_y, tau_xy = node_vals[i - 1].tolist()
                if plane_type == "stress":
                    mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)
                else:
                    sig_z = nu * (sig_x + sig_y)
                    mises = np.sqrt(
                        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                        + 3.0 * tau_xy**2
                    )
                writer.writerow([elem.id, nid, i, sig_x, sig_y, tau_xy, mises])


def export_quad4_nodal_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Quad4 nodal stresses averaged from element extrapolation."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        etype = str(elem.type).lower()
        if "quad4" not in etype:
            continue
        if len(elem.node_ids) != 4:
            raise ValueError(f"Quad4 elem must have 4 nodes, elem {elem.id} node_ids={elem.node_ids}")

        node_vals, pt, nu = _quad4_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.dof_manager.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def export_quad8_nodal_stress_csv(
    mesh: PlaneMesh2D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 3,
) -> None:
    """Export Quad8 nodal stresses averaged from element extrapolation."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    plane_type = None
    nu_ref = 0.0

    for elem in mesh.elements:
        etype = str(elem.type).lower()
        if "quad8" not in etype:
            continue
        if len(elem.node_ids) != 8:
            raise ValueError(f"Quad8 elem must have 8 nodes, elem {elem.id} node_ids={elem.node_ids}")

        node_vals, pt, nu = _quad8_element_nodal_stress(mesh, elem, U, node_lookup, gauss_order)
        if plane_type is None:
            plane_type = pt
            nu_ref = nu

        for i, nid in enumerate(elem.node_ids):
            sums[nid] = sums.get(nid, np.zeros(3, dtype=float)) + node_vals[i]
            counts[nid] = counts.get(nid, 0) + 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "sig_x", "sig_y", "tau_xy", "mises"])

        for nid in mesh.dof_manager.node_ids:
            node = node_lookup[nid]
            if counts.get(nid, 0) == 0:
                sig_x = sig_y = tau_xy = 0.0
            else:
                avg = sums[nid] / counts[nid]
                sig_x, sig_y, tau_xy = avg.tolist()

            if plane_type == "strain":
                sig_z = nu_ref * (sig_x + sig_y)
                mises = np.sqrt(
                    0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
                    + 3.0 * tau_xy**2
                )
            else:
                mises = np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2)

            writer.writerow([nid, node.x, node.y, sig_x, sig_y, tau_xy, mises])


def _build_vtk_cells(mesh, node_id_to_pt_idx: Dict[int, int]):
    """Build VTK connectivity for supported element types."""
    cells = []
    cell_types = []
    elems_for_cell = []
    for elem in mesh.elements:
        etype = str(elem.type).lower()
        vtk_conn = None
        vtk_type = None

        if "truss" in etype or "beam" in etype:
            if len(elem.node_ids) != 2:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [2] + pt_ids
            vtk_type = 3

        elif "tri3" in etype:
            if len(elem.node_ids) != 3:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [3] + pt_ids
            vtk_type = 5

        elif "quad4" in etype:
            if len(elem.node_ids) != 4:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [4] + pt_ids
            vtk_type = 9

        elif "quad8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 23
        
        elif "hex8" in etype:
            if len(elem.node_ids) != 8:
                continue
            pt_ids = [node_id_to_pt_idx[nid] for nid in elem.node_ids]
            vtk_conn = [8] + pt_ids
            vtk_type = 12

        else:
            continue

        cells.append(vtk_conn)
        cell_types.append(vtk_type)
        elems_for_cell.append(elem)
    return cells, cell_types, elems_for_cell


def _polar_basis(x: float, y: float, center: Sequence[float]):
    """Return cos/sin of polar basis at (x, y)."""
    dx = x - float(center[0])
    dy = y - float(center[1])
    r = (dx * dx + dy * dy) ** 0.5
    if r == 0.0:
        return 1.0, 0.0
    return dx / r, dy / r


def _polar_displacement(c: float, s: float, ux: float, uy: float):
    """Return (ur, ut) from (ux, uy)."""
    ur = c * ux + s * uy
    ut = -s * ux + c * uy
    return ur, ut


def _polar_stress(c: float, s: float, sig_x: float, sig_y: float, tau_xy: float):
    """Return (sig_r, sig_t, tau_rt) from (sig_x, sig_y, tau_xy)."""
    sig_r = c * c * sig_x + s * s * sig_y + 2.0 * s * c * tau_xy
    sig_t = s * s * sig_x + c * c * sig_y - 2.0 * s * c * tau_xy
    tau_rt = -s * c * sig_x + s * c * sig_y + (c * c - s * s) * tau_xy
    return sig_r, sig_t, tau_rt


def convert_nodal_displacement_to_polar(
    mesh: Mesh2DProtocol,
    node_disp: Dict[int, Dict[str, float]],
    center: Sequence[float],
) -> Dict[int, Dict[str, float]]:
    """Convert nodal displacement dict into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")

    node_lookup = {node.id: node for node in mesh.nodes}
    polar_disp: Dict[int, Dict[str, float]] = {}

    for node in mesh.nodes:
        disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
        c, s = _polar_basis(node.x, node.y, center)
        ur, ut = _polar_displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[node.id] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    for nid, disp in node_disp.items():
        if nid in polar_disp:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        c, s = _polar_basis(node.x, node.y, center)
        ur, ut = _polar_displacement(c, s, float(disp.get("ux", 0.0)), float(disp.get("uy", 0.0)))
        polar_disp[nid] = {"ux": ur, "uy": ut, "rz": float(disp.get("rz", 0.0))}

    return polar_disp


def _convert_nodal_stress_fields_to_polar(
    mesh: Mesh2DProtocol,
    nodal_fields: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert nodal stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(nodal_fields) or polar_names.intersection(nodal_fields):
        return nodal_fields

    node_lookup = {node.id: node for node in mesh.nodes}
    new_fields = {name: vals for name, vals in nodal_fields.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for node in mesh.nodes:
        nid = node.id
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = _polar_basis(node.x, node.y, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    for nid in nodal_fields["sig_x"]:
        if nid in sig_r:
            continue
        node = node_lookup.get(nid)
        if node is None:
            continue
        sx = float(nodal_fields["sig_x"].get(nid, 0.0))
        sy = float(nodal_fields["sig_y"].get(nid, 0.0))
        txy = float(nodal_fields["tau_xy"].get(nid, 0.0))
        c, s = _polar_basis(node.x, node.y, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[nid] = sr
        sig_t[nid] = st
        tau_rt[nid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields


def _convert_element_stress_fields_to_polar(
    mesh: Mesh2DProtocol,
    field_data: Dict[str, Dict[int, float]],
    center: Sequence[float],
) -> Dict[str, Dict[int, float]]:
    """Convert element stress fields to polar components."""
    required = {"sig_x", "sig_y", "tau_xy"}
    polar_names = {"sig_r", "sig_t", "tau_rt"}
    if not required.issubset(field_data) or polar_names.intersection(field_data):
        return field_data

    node_lookup = {node.id: node for node in mesh.nodes}
    elem_lookup = {elem.id: elem for elem in mesh.elements}

    new_fields = {name: vals for name, vals in field_data.items() if name not in required}
    sig_r: Dict[int, float] = {}
    sig_t: Dict[int, float] = {}
    tau_rt: Dict[int, float] = {}

    for eid, elem in elem_lookup.items():
        sx = float(field_data["sig_x"].get(eid, 0.0))
        sy = float(field_data["sig_y"].get(eid, 0.0))
        txy = float(field_data["tau_xy"].get(eid, 0.0))
        xs = [node_lookup[nid].x for nid in elem.node_ids if nid in node_lookup]
        ys = [node_lookup[nid].y for nid in elem.node_ids if nid in node_lookup]
        if not xs or not ys:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        c, s = _polar_basis(cx, cy, center)
        sr, st, trt = _polar_stress(c, s, sx, sy, txy)
        sig_r[eid] = sr
        sig_t[eid] = st
        tau_rt[eid] = trt

    new_fields["sig_r"] = sig_r
    new_fields["sig_t"] = sig_t
    new_fields["tau_rt"] = tau_rt
    return new_fields


def _write_vtk(mesh, cells, cell_types, elems_for_cell, node_disp, field_data, vtk_path: str, nodal_fields: Optional[Dict[str, Dict[int, float]]] = None):
    """Write VTK file from node displacements and cell fields."""
    nodes: List[Node2D] = mesh.nodes
    num_points = len(nodes)
    num_cells = len(cells)

    cell_field_arrays: Dict[str, np.ndarray] = {}
    for field_name, field_dict in field_data.items():
        arr = np.zeros(num_cells, dtype=float)
        for cidx, elem in enumerate(elems_for_cell):
            eid = elem.id
            arr[cidx] = float(field_dict.get(eid, 0.0))
        cell_field_arrays[field_name] = arr

    with open(vtk_path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM results from CSV\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for node in nodes:
            f.write(f"{node.x} {node.y} 0.0\n")

        total_ints = sum(len(conn) for conn in cells)
        f.write(f"\nCELLS {num_cells} {total_ints}\n")
        for conn in cells:
            f.write(" ".join(str(v) for v in conn) + "\n")

        f.write(f"\nCELL_TYPES {num_cells}\n")
        for ct in cell_types:
            f.write(f"{ct}\n")

        f.write(f"\nPOINT_DATA {num_points}\n")
        f.write("VECTORS displacement float\n")
        for node in nodes:
            disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
            f.write(f"{disp['ux']} {disp['uy']} 0.0\n")

        has_any_rz = any(abs(d.get("rz", 0.0)) > 0.0 for d in node_disp.values())
        if getattr(mesh, "dofs_per_node", 0) >= 3 and has_any_rz:
            f.write("\nSCALARS rotz float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for node in nodes:
                f.write(f"{node_disp.get(node.id, {}).get('rz', 0.0)}\n")

        if nodal_fields:
            for field_name, field_dict in nodal_fields.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{float(field_dict.get(node.id, 0.0))}\n")

        if cell_field_arrays:
            f.write(f"\nCELL_DATA {num_cells}\n")
            for field_name, arr in cell_field_arrays.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in arr:
                    f.write(f"{val}\n")


def export_vtk_from_csv(
    mesh,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
    polar: bool = False,
    polar_center: Optional[Sequence[float]] = None,
) -> None:
    """Convert displacement + element stress CSV to VTK."""

    node_disp: Dict[int, Dict[str, float]] = {}

    with open(disp_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"node_id", "ux", "uy"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"Disp CSV requires columns {required_cols}, got {reader.fieldnames}")

        has_rz = "rz" in reader.fieldnames

        for row in reader:
            nid = int(row["node_id"])
            ux = float(row["ux"])
            uy = float(row["uy"])
            rz = float(row["rz"]) if has_rz and row.get("rz", "") != "" else 0.0
            node_disp[nid] = {"ux": ux, "uy": uy, "rz": rz}

    for node in mesh.nodes:
        if node.id not in node_disp:
            node_disp[node.id] = {"ux": 0.0, "uy": 0.0, "rz": 0.0}

    if polar:
        if polar_center is None:
            raise ValueError("export_vtk_from_csv: polar_center required when polar=True")
        node_disp = convert_nodal_displacement_to_polar(mesh, node_disp, polar_center)

    nodal_fields: Dict[str, Dict[int, float]] = {}
    if nodal_stress_csv_path is not None:
        with open(nodal_stress_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"Nodal stress CSV requires 'node_id', got {reader.fieldnames}")

            ignore_exact = {"node_id", "x", "y"}
            field_names = [name for name in (reader.fieldnames or []) if name not in ignore_exact]

            for name in field_names:
                nodal_fields[name] = {}

            for row in reader:
                nid = int(row["node_id"])
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    nodal_fields[name][nid] = val

    if polar and nodal_fields:
        nodal_fields = _convert_nodal_stress_fields_to_polar(mesh, nodal_fields, polar_center)

    field_data: Dict[str, Dict[int, float]] = {}
    if elem_csv_path is not None:
        with open(elem_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "elem_id" not in (reader.fieldnames or []):
                raise ValueError(f"Element stress CSV requires 'elem_id', got {reader.fieldnames}")

            ignore_prefixes = ("node", "nid")
            ignore_exact = {"elem_id", "local_node"}

            stress_field_names = [
                name for name in (reader.fieldnames or [])
                if name not in ignore_exact and not name.startswith(ignore_prefixes)
            ]

            for name in stress_field_names:
                field_data[name] = {}

            for row in reader:
                eid = int(row["elem_id"])
                for name in stress_field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    field_data[name][eid] = val

    if polar and field_data:
        field_data = _convert_element_stress_fields_to_polar(mesh, field_data, polar_center)

    node_id_to_pt_idx: Dict[int, int] = {node.id: i for i, node in enumerate(mesh.nodes)}
    cells, cell_types, elems_for_cell = _build_vtk_cells(mesh, node_id_to_pt_idx)
    if not cells:
        raise ValueError("export_vtk_from_csv: no supported elements")

    _write_vtk(mesh, cells, cell_types, elems_for_cell, node_disp, field_data, vtk_path, nodal_fields)


def extract_path_data(
    mesh: Mesh2DProtocol,
    start_id: int,
    end_id: int,
    points: int,
    target: str,
    path: str = "xydata.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
    normalized: bool = False,
) -> None:
    """Extract path data to CSV."""
    if points < 2:
        raise ValueError("points must be >= 2")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    if start_id not in node_lookup or end_id not in node_lookup:
        raise ValueError("start_id or end_id not in mesh nodes")

    start = np.array([node_lookup[start_id].x, node_lookup[start_id].y], dtype=float)
    end = np.array([node_lookup[end_id].x, node_lookup[end_id].y], dtype=float)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        raise ValueError("start_id and end_id define zero length path")
    direction = vec / length

    def _read_nodal_fields(csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV requires node_id column, got {reader.fieldnames}")

            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in {"node_id", "x", "y"}
            ]
            data: Dict[int, Dict[str, float]] = {}

            for row in reader:
                nid = int(row["node_id"])
                values: Dict[str, float] = {}
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        values[name] = float(val_str)
                    except ValueError:
                        values[name] = 0.0
                data[nid] = values
            return field_names, data

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(stress_csv_path)

    source_data = None
    if disp_csv_path is not None and target in disp_fields:
        source_data = disp_data
    elif stress_csv_path is not None and target in stress_fields:
        source_data = stress_data

    if source_data is None:
        raise ValueError(f"target {target} not found in provided CSV files")

    candidates = [
        node for node in mesh.nodes
        if node.id in source_data and target in source_data[node.id]
    ]
    if not candidates:
        raise ValueError("no nodes with target data available")

    selected_ids: List[int] = []
    for i in range(points):
        t = i / (points - 1)
        pos = start + t * vec
        best_id = None
        best_dist = None
        for node in candidates:
            dx = node.x - pos[0]
            dy = node.y - pos[1]
            dist2 = dx * dx + dy * dy
            if best_dist is None or dist2 < best_dist:
                best_dist = dist2
                best_id = node.id
        if best_id is None:
            raise ValueError("failed to select nodes along path")
        selected_ids.append(best_id)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "x", "y", target])

        for nid in selected_ids:
            node = node_lookup[nid]
            val = source_data[nid].get(target)
            if val is None:
                raise ValueError(f"node {nid} missing target {target}")

            proj = np.dot(np.array([node.x, node.y], dtype=float) - start, direction)
            dist = proj / length if normalized else proj
            writer.writerow([dist, node.x, node.y, val])


def extract_circle_data(
    center: Sequence[float],
    radius: float,
    points: int,
    target: str,
    csv_path: str,
    save_path: str,
) -> None:
    """Extract target data on a circle to CSV."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    if points < 2:
        raise ValueError("points must be >= 2")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("CSV requires x and y columns")
        if target not in reader.fieldnames:
            raise ValueError(f"target {target} not found in CSV header")

        rows = []
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                continue
            rows.append((x, y, row))

    if not rows:
        raise ValueError("no valid rows in CSV")

    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", target])

        for theta in angles:
            px = cx + radius * np.cos(theta)
            py = cy + radius * np.sin(theta)

            best_row = None
            best_dist = None
            for x, y, row in rows:
                dx = x - px
                dy = y - py
                dist2 = dx * dx + dy * dy
                if best_dist is None or dist2 < best_dist:
                    best_dist = dist2
                    best_row = row

            if best_row is None:
                continue
            writer.writerow([best_row.get("x", ""), best_row.get("y", ""), best_row.get(target, "")])


def extract_nodes_data(
    mesh: Mesh2DProtocol,
    node_ids: Sequence[int],
    targets: Sequence[str],
    path: str = "nodes_data.csv",
    stress_csv_path: Optional[str] = None,
    disp_csv_path: Optional[str] = None,
) -> None:
    """Extract nodal target data to CSV."""
    if not node_ids:
        raise ValueError("node_ids is empty")
    if not targets:
        raise ValueError("targets is empty")
    if stress_csv_path is None and disp_csv_path is None:
        raise ValueError("provide stress_csv_path or disp_csv_path")

    node_lookup = {node.id: node for node in mesh.nodes}
    for nid in node_ids:
        if nid not in node_lookup:
            raise ValueError(f"node_id {nid} not in mesh")

    def _read_nodal_fields(csv_path: str):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV requires node_id column, got {reader.fieldnames}")

            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in {"node_id", "x", "y"}
            ]
            data: Dict[int, Dict[str, float]] = {}

            for row in reader:
                nid = int(row["node_id"])
                values: Dict[str, float] = {}
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        values[name] = float(val_str)
                    except ValueError:
                        values[name] = 0.0
                data[nid] = values
            return field_names, data

    disp_fields: List[str] = []
    disp_data: Dict[int, Dict[str, float]] = {}
    if disp_csv_path is not None:
        disp_fields, disp_data = _read_nodal_fields(disp_csv_path)

    stress_fields: List[str] = []
    stress_data: Dict[int, Dict[str, float]] = {}
    if stress_csv_path is not None:
        stress_fields, stress_data = _read_nodal_fields(stress_csv_path)

    target_sources: Dict[str, Dict[int, Dict[str, float]]] = {}
    for target in targets:
        if disp_csv_path is not None and target in disp_fields:
            target_sources[target] = disp_data
        elif stress_csv_path is not None and target in stress_fields:
            target_sources[target] = stress_data
        else:
            raise ValueError(f"target {target} not found in provided CSV files")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y"] + list(targets))

        for nid in node_ids:
            node = node_lookup[nid]
            row = [nid, node.x, node.y]
            for target in targets:
                source = target_sources[target]
                if nid not in source or target not in source[nid]:
                    raise ValueError(f"node {nid} missing target {target}")
                row.append(source[nid][target])
            writer.writerow(row)


def convert_nodal_solution_into_polar_coord(
    csv_path: str,
    center: Sequence[float],
    out_path: str,
) -> None:
    """Convert nodal displacement or stress CSV into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        fields = list(reader.fieldnames)
        has_disp = "ux" in fields and "uy" in fields
        has_stress = "sig_x" in fields and "sig_y" in fields and "tau_xy" in fields

        if has_disp and has_stress:
            raise ValueError("CSV has both displacement and stress columns")
        if not has_disp and not has_stress:
            raise ValueError("CSV missing displacement or stress columns")

        if "x" not in fields or "y" not in fields:
            raise ValueError("CSV requires x and y columns")

        if has_disp:
            mapping = {"ux": "ur", "uy": "ut"}
        else:
            mapping = {"sig_x": "sig_r", "sig_y": "sig_t", "tau_xy": "tau_rt"}

        out_fields = [mapping.get(name, name) for name in fields]
        rows = list(reader)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_fields)

        for row in rows:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                raise ValueError("x or y is not numeric")

            dx = x - cx
            dy = y - cy
            r = (dx * dx + dy * dy) ** 0.5
            if r == 0.0:
                c = 1.0
                s = 0.0
            else:
                c = dx / r
                s = dy / r

            ux_val = uy_val = None
            sx_val = sy_val = txy_val = None
            if has_disp:
                try:
                    ux_val = float(row["ux"])
                    uy_val = float(row["uy"])
                except ValueError:
                    raise ValueError("ux or uy is not numeric")

                ur = c * ux_val + s * uy_val
                ut = -s * ux_val + c * uy_val

            if has_stress:
                try:
                    sx_val = float(row["sig_x"])
                    sy_val = float(row["sig_y"])
                    txy_val = float(row["tau_xy"])
                except ValueError:
                    raise ValueError("sig_x, sig_y, or tau_xy is not numeric")

                sig_r = c * c * sx_val + s * s * sy_val + 2.0 * s * c * txy_val
                sig_t = s * s * sx_val + c * c * sy_val - 2.0 * s * c * txy_val
                tau_rt = -s * c * sx_val + s * c * sy_val + (c * c - s * s) * txy_val

            out_row = []
            for name in fields:
                if has_disp and name == "ux":
                    out_row.append(ur)
                elif has_disp and name == "uy":
                    out_row.append(ut)
                elif has_stress and name == "sig_x":
                    out_row.append(sig_r)
                elif has_stress and name == "sig_y":
                    out_row.append(sig_t)
                elif has_stress and name == "tau_xy":
                    out_row.append(tau_rt)
                else:
                    out_row.append(row.get(name, ""))

            writer.writerow(out_row)


def export_nodal_displacements_csv_3d(
    mesh: Mesh3DProtocol,
    U: Sequence[float],
    path: str,
    component_names: Optional[List[str]] = None,
) -> None:
    """Export 3D nodal displacements to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    dofs_per_node = mesh.dofs_per_node

    if component_names is None:
        if dofs_per_node == 3:
            component_names = ["ux", "uy", "uz"]
        else:
            component_names = [f"u{c}" for c in range(dofs_per_node)]
    else:
        if len(component_names) != dofs_per_node:
            raise ValueError(
                f"component_names length {len(component_names)} != dofs_per_node={dofs_per_node}"
            )

    node_lookup = {node.id: node for node in mesh.nodes}
    header = ["node_id", "x", "y", "z"] + component_names

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.dof_manager.node_ids:
            node: Node3D = node_lookup[nid]
            dofs = mesh.node_dofs(nid)
            disp_vals = [U[dof] for dof in dofs]
            writer.writerow([nid, node.x, node.y, node.z] + disp_vals)


def export_hex8_element_stress_csv(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
) -> None:
    """Export Hex8 element stresses to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    header = ["elem_id", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for elem in mesh.elements:
            if elem.type.lower() != "hex8":
                continue

            # Compute element stresses at centroid (xi=0, eta=0, zeta=0)
            stresses = _compute_hex8_element_stress_at_point(mesh, elem, U, node_lookup, 0.0, 0.0, 0.0)
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = stresses
            mises = np.sqrt(0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2 +
                                   6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))

            writer.writerow([elem.id, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def _compute_hex8_element_stress_at_point(
    mesh: HexMesh3D,
    elem,
    U: np.ndarray,
    node_lookup: Dict[int, Node3D],
    xi: float,
    eta: float,
    zeta: float,
) -> tuple:
    """Compute stresses at a point in Hex8 element."""
    from .stiffness import _hex8_shape_funcs_grads, _compute_D_3d

    # Get material properties
    E = float(elem.props["E"])
    nu = float(elem.props["nu"])
    D = _compute_D_3d(E, nu)

    # Get node coordinates
    nids = elem.node_ids
    nodes = [node_lookup[nid] for nid in nids]
    x = np.array([n.x for n in nodes], dtype=float)
    y = np.array([n.y for n in nodes], dtype=float)
    z = np.array([n.z for n in nodes], dtype=float)

    # Shape function gradients
    N, dN_dxi, dN_deta, dN_dzeta = _hex8_shape_funcs_grads(xi, eta, zeta)

    # Jacobian matrix
    J = np.array([
        [np.sum(dN_dxi * x), np.sum(dN_dxi * y), np.sum(dN_dxi * z)],
        [np.sum(dN_deta * x), np.sum(dN_deta * y), np.sum(dN_deta * z)],
        [np.sum(dN_dzeta * x), np.sum(dN_dzeta * y), np.sum(dN_dzeta * z)],
    ], dtype=float)

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

    # Get element DOFs
    elem_dofs = mesh.element_dofs(elem)
    Ue = U[elem_dofs]

    # Strain vector
    epsilon = B @ Ue

    # Stress vector
    sigma = D @ epsilon

    return sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5]  # sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx


def export_hex8_nodal_stress_csv(
    mesh: HexMesh3D,
    U: Sequence[float],
    path: str,
    gauss_order: int = 2,
) -> None:
    """Export Hex8 nodal stresses (extrapolated from Gauss points) to CSV."""
    U = np.asarray(U, dtype=float).ravel()
    if U.shape[0] != mesh.num_dofs:
        raise ValueError(f"U length {U.shape[0]} != mesh.num_dofs={mesh.num_dofs}")

    node_lookup = {node.id: node for node in mesh.nodes}

    # Gauss points for extrapolation
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

    header = ["node_id", "x", "y", "z", "sig_x", "sig_y", "sig_z", "tau_xy", "tau_yz", "tau_zx", "mises"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for nid in mesh.dof_manager.node_ids:
            node = node_lookup[nid]

            # Find elements connected to this node
            connected_elems = [elem for elem in mesh.elements if nid in elem.node_ids]

            if not connected_elems:
                # Node not connected to any element
                writer.writerow([nid, node.x, node.y, node.z, 0, 0, 0, 0, 0, 0, 0])
                continue

            # Average stresses from connected elements
            stress_sum = np.zeros(6)
            count = 0

            for elem in connected_elems:
                # Compute stresses at Gauss points
                gp_stresses = []
                for xi, eta, zeta, w in gps:
                    stress = _compute_hex8_element_stress_at_point(mesh, elem, U, node_lookup, xi, eta, zeta)
                    gp_stresses.append(stress)

                # Simple average for now (could implement proper extrapolation)
                elem_stress = np.mean(gp_stresses, axis=0)
                stress_sum += elem_stress
                count += 1

            avg_stress = stress_sum / count
            sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx = avg_stress
            mises = np.sqrt(0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2 +
                                   6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))

            writer.writerow([nid, node.x, node.y, node.z, sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises])


def _write_vtk_3d(
    mesh,
    cells,
    cell_types,
    elems_for_cell,
    node_disp,
    field_data,
    vtk_path: str,
    nodal_fields: Optional[Dict[str, Dict[int, float]]] = None,
):
    """Write 3D VTK file from node displacements and cell fields."""
    nodes: List[Node3D] = mesh.nodes
    num_points = len(nodes)
    num_cells = len(cells)

    cell_field_arrays: Dict[str, np.ndarray] = {}
    for field_name, field_dict in field_data.items():
        arr = np.zeros(num_cells, dtype=float)
        for cidx, elem in enumerate(elems_for_cell):
            eid = elem.id
            arr[cidx] = float(field_dict.get(eid, 0.0))
        cell_field_arrays[field_name] = arr

    with open(vtk_path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEM 3D results from CSV\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for node in nodes:
            f.write(f"{node.x} {node.y} {node.z}\n")

        total_ints = sum(len(conn) for conn in cells)
        f.write(f"\nCELLS {num_cells} {total_ints}\n")
        for conn in cells:
            f.write(" ".join(str(v) for v in conn) + "\n")

        f.write(f"\nCELL_TYPES {num_cells}\n")
        for ct in cell_types:
            f.write(f"{ct}\n")

        f.write(f"\nPOINT_DATA {num_points}\n")
        f.write("VECTORS displacement float\n")
        for node in nodes:
            disp = node_disp.get(node.id, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
            f.write(f"{disp['ux']} {disp['uy']} {disp['uz']}\n")

        if nodal_fields:
            for field_name, field_dict in nodal_fields.items():
                f.write(f"\nSCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for node in nodes:
                    f.write(f"{float(field_dict.get(node.id, 0.0))}\n")

        f.write(f"\nCELL_DATA {num_cells}\n")
        for field_name, arr in cell_field_arrays.items():
            f.write(f"SCALARS {field_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{float(v)}\n")
            f.write("\n")
        
def export_vtk_from_csv_3d(
    mesh: HexMesh3D,
    disp_csv_path: str,
    elem_csv_path: Optional[str],
    vtk_path: str,
    nodal_stress_csv_path: Optional[str] = None,
) -> None:
    """Convert 3D displacement + Hex8 stress CSV files to VTK."""
    node_disp: Dict[int, Dict[str, float]] = {}

    with open(disp_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"node_id", "ux", "uy", "uz"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"3D disp CSV requires columns {required_cols}, got {reader.fieldnames}"
            )

        for row in reader:
            nid = int(row["node_id"])
            ux = float(row["ux"])
            uy = float(row["uy"])
            uz = float(row["uz"])
            node_disp[nid] = {"ux": ux, "uy": uy, "uz": uz}

    for node in mesh.nodes:
        if node.id not in node_disp:
            node_disp[node.id] = {"ux": 0.0, "uy": 0.0, "uz": 0.0}

    nodal_fields: Dict[str, Dict[int, float]] = {}
    if nodal_stress_csv_path is not None:
        with open(nodal_stress_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "node_id" not in (reader.fieldnames or []):
                raise ValueError(
                    f"3D nodal stress CSV requires 'node_id', got {reader.fieldnames}"
                )

            ignore_exact = {"node_id", "x", "y", "z"}
            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in ignore_exact
            ]

            for name in field_names:
                nodal_fields[name] = {}

            for row in reader:
                nid = int(row["node_id"])
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    nodal_fields[name][nid] = val

    field_data: Dict[str, Dict[int, float]] = {}
    if elem_csv_path is not None:
        with open(elem_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "elem_id" not in (reader.fieldnames or []):
                raise ValueError(
                    f"3D element stress CSV requires 'elem_id', got {reader.fieldnames}"
                )

            ignore_exact = {"elem_id"}
            field_names = [
                name for name in (reader.fieldnames or [])
                if name not in ignore_exact
            ]

            for name in field_names:
                field_data[name] = {}

            for row in reader:
                eid = int(row["elem_id"])
                for name in field_names:
                    val_str = row.get(name, "")
                    if val_str == "":
                        continue
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    field_data[name][eid] = val

    node_id_to_pt_idx: Dict[int, int] = {node.id: i for i, node in enumerate(mesh.nodes)}
    cells, cell_types, elems_for_cell = _build_vtk_cells(mesh, node_id_to_pt_idx)
    if not cells:
        raise ValueError("export_vtk_from_csv_3d: no supported elements")

    _write_vtk_3d(
        mesh,
        cells,
        cell_types,
        elems_for_cell,
        node_disp,
        field_data,
        vtk_path,
        nodal_fields,
    )
def select_node_ids_by_x(mesh, x_value: float, tol: float = 1e-8):
    """Return node ids whose x matches target within tol."""
    return [n.id for n in mesh.nodes if abs(n.x - x_value) <= tol]


def select_node_ids_by_y(mesh, y_value: float, tol: float = 1e-8):
    """Return node ids whose y matches target within tol."""
    return [n.id for n in mesh.nodes if abs(n.y - y_value) <= tol]

def select_node_ids_by_xz(mesh, x_value: float, z_value: float, tol: float = 1e-8):
    """Return node ids whose x and z match target within tol."""
    return [
        n.id for n in mesh.nodes
        if abs(n.x - x_value) <= tol and abs(n.z - z_value) <= tol
    ]


def select_node_ids_by_coord(mesh, x: float | None = None, y: float | None = None, tol: float = 1e-8):
    """Return node ids whose x and/or y matches given values within tol."""
    if x is None and y is None:
        raise ValueError("At least one of x or y must be provided")
    ids = []
    for n in mesh.nodes:
        okx = True if x is None else abs(n.x - x) <= tol
        oky = True if y is None else abs(n.y - y) <= tol
        if okx and oky:
            ids.append(n.id)
    return ids


def select_node_ids_in_box(
    mesh,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Return node ids inside a bounding box; None bounds mean open."""
    ids = []
    for n in mesh.nodes:
        if xmin is not None and n.x < xmin:
            continue
        if xmax is not None and n.x > xmax:
            continue
        if ymin is not None and n.y < ymin:
            continue
        if ymax is not None and n.y > ymax:
            continue
        ids.append(n.id)
    return ids


def select_boundary_node_ids(mesh, tol: float = 1e-8):
    """Return dict of boundary node ids on left/right/bottom/top by coordinate extent."""
    xs = [n.x for n in mesh.nodes]
    ys = [n.y for n in mesh.nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {
        "left": [n.id for n in mesh.nodes if abs(n.x - xmin) <= tol],
        "right": [n.id for n in mesh.nodes if abs(n.x - xmax) <= tol],
        "bottom": [n.id for n in mesh.nodes if abs(n.y - ymin) <= tol],
        "top": [n.id for n in mesh.nodes if abs(n.y - ymax) <= tol],
    }


def select_nearest_node_id(mesh, x: float, y: float):
    """Return the id of the node closest to (x, y)."""
    best_id = None
    best_dist2 = None
    for n in mesh.nodes:
        d2 = (n.x - x) ** 2 + (n.y - y) ** 2
        if best_dist2 is None or d2 < best_dist2:
            best_dist2 = d2
            best_id = n.id
    return best_id


def _element_edge_node_ids(elem):
    """Return list of edge node id lists for common 2D elements."""
    etype = str(elem.type).lower()
    nids = elem.node_ids

    if "tri3" in etype and len(nids) == 3:
        return [
            [nids[0], nids[1]],
            [nids[1], nids[2]],
            [nids[2], nids[0]],
        ]

    if "quad4" in etype and len(nids) == 4:
        return [
            [nids[0], nids[1]],
            [nids[1], nids[2]],
            [nids[2], nids[3]],
            [nids[3], nids[0]],
        ]

    if "quad8" in etype and len(nids) == 8:
        return [
            [nids[0], nids[4], nids[1]],
            [nids[1], nids[5], nids[2]],
            [nids[2], nids[6], nids[3]],
            [nids[3], nids[7], nids[0]],
        ]

    return []


def select_boundary_edges(mesh):
    """Return boundary edges as (elem_id, local_edge, node_ids)."""
    edge_count = {}
    edge_store = {}

    for elem in mesh.elements:
        edges = _element_edge_node_ids(elem)
        for local_edge, e_nids in enumerate(edges):
            key = tuple(sorted((e_nids[0], e_nids[-1])))
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_store.setdefault(key, []).append((elem.id, local_edge, e_nids))

    boundary = []
    for key, count in edge_count.items():
        if count == 1:
            boundary.extend(edge_store.get(key, []))

    return boundary


def select_all_edges(mesh):
    """Return all edges as (elem_id, local_edge, node_ids)."""
    edges = []
    for elem in mesh.elements:
        for local_edge, nids in enumerate(_element_edge_node_ids(elem)):
            edges.append((elem.id, local_edge, nids))
    return edges


def select_edges_by_x(mesh, x_value: float, tol: float = 1e-8, boundary_only: bool = True):
    """Return edges whose all nodes match x within tol."""
    edges = select_boundary_edges(mesh) if boundary_only else select_all_edges(mesh)
    node_lookup = {n.id: n for n in mesh.nodes}
    result = []
    for elem_id, local_edge, nids in edges:
        if all(abs(node_lookup[nid].x - x_value) <= tol for nid in nids):
            result.append((elem_id, local_edge, nids))
    return result


def select_edges_by_y(mesh, y_value: float, tol: float = 1e-8, boundary_only: bool = True):
    """Return edges whose all nodes match y within tol."""
    edges = select_boundary_edges(mesh) if boundary_only else select_all_edges(mesh)
    node_lookup = {n.id: n for n in mesh.nodes}
    result = []
    for elem_id, local_edge, nids in edges:
        if all(abs(node_lookup[nid].y - y_value) <= tol for nid in nids):
            result.append((elem_id, local_edge, nids))
    return result


def select_edges_by_coord(
    mesh,
    x: float | None = None,
    y: float | None = None,
    tol: float = 1e-8,
    boundary_only: bool = True,
):
    """Return edges whose all nodes match x/y within tol."""
    if x is None and y is None:
        raise ValueError("At least one of x or y must be provided")
    edges = select_boundary_edges(mesh) if boundary_only else select_all_edges(mesh)
    node_lookup = {n.id: n for n in mesh.nodes}
    result = []
    for elem_id, local_edge, nids in edges:
        okx = True if x is None else all(abs(node_lookup[nid].x - x) <= tol for nid in nids)
        oky = True if y is None else all(abs(node_lookup[nid].y - y) <= tol for nid in nids)
        if okx and oky:
            result.append((elem_id, local_edge, nids))
    return result


def select_node_ids_in_circle(mesh, x: float, y: float, r: float, tol: float = 1e-8):
    """Return node ids inside a circle."""
    if r < 0.0:
        raise ValueError("r must be non-negative")
    r_eff = r + tol
    r2 = r_eff * r_eff
    ids = []
    for n in mesh.nodes:
        dx = n.x - x
        dy = n.y - y
        if dx * dx + dy * dy <= r2:
            ids.append(n.id)
    return ids

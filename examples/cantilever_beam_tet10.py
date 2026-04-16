# Example: cantilever beam solved with Tet10 solid elements.

from pathlib import Path

from fem.mesh_io import read_tet10_3d_abaqus
from fem.stiffness import compute_tet10_element_stiffness
from fem.assemble import assemble_global_stiffness_sparse
from fem.helper import select_node_ids_by_x, select_node_ids_by_xz
from fem.boundary import BoundaryCondition3D, build_load_vector_3d, apply_dirichlet_bc_3d
from fem.solve import solve_linear_system_sparse
from fem.post import (
    export_nodal_displacements_csv_3d,
    export_tet10_element_stress_csv,
    export_tet10_nodal_stress_csv,
    export_vtk_from_csv_3d,
)


# Read mesh and material data from Abaqus input.
mesh = read_tet10_3d_abaqus(
    inp_path=r"examples\cantilever_beam_tet10.inp",
    material_path=r"examples\cantilever_beam_materials.csv",
    material_id=1,
)

# Geometry extents.
xs = [node.x for node in mesh.nodes]
zs = [node.z for node in mesh.nodes]
x_min, x_max = min(xs), max(xs)
z_max = max(zs)

# Assemble global stiffness matrix (sparse).
K = assemble_global_stiffness_sparse(
    num_dofs=mesh.num_dofs,
    num_elements=len(mesh.elements),
    get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
    compute_element_stiffness=lambda eid: compute_tet10_element_stiffness(
        mesh,
        mesh.elements[eid],
        node_lookup={node.id: node for node in mesh.nodes},
    ),
)

# Select fixed boundary (x=x_min) and loaded nodes (x=x_max, z=z_max).
nodes_sel_fixed = select_node_ids_by_x(mesh, x_value=x_min)
nodes_sel_loaded = select_node_ids_by_xz(mesh, x_value=x_max, z_value=z_max)

if not nodes_sel_fixed:
    raise ValueError("No fixed nodes found at x=x_min")
if not nodes_sel_loaded:
    raise ValueError("No loaded nodes found at x=x_max, z=z_max")

print("Fixed nodes at x=x_min:", nodes_sel_fixed)
print("Loaded nodes at x=x_max, z=z_max:", nodes_sel_loaded)

# Define boundary conditions:
# fully fixed at x=x_min, downward nodal load distributed on the free-end top edge.
bc = BoundaryCondition3D()

for node_id in nodes_sel_fixed:
    bc.add_fixed_support(node_id=node_id, components=[0, 1, 2], mesh=mesh)

load_per_node = -100.0 / len(nodes_sel_loaded)
for node_id in nodes_sel_loaded:
    bc.add_nodal_force(node_id=node_id, component=1, value=load_per_node, mesh=mesh)

# Build load vector and apply Dirichlet constraints.
F = build_load_vector_3d(mesh, bc)
K_mod, F_mod = apply_dirichlet_bc_3d(K, F, bc)

# Solve for nodal displacements.
U = solve_linear_system_sparse(K_mod, F_mod)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

disp_csv_path = results_dir / "cantilever_beam_tet10_nodal_displacements.csv"
elem_csv_path = results_dir / "cantilever_beam_tet10_element_stress.csv"
nodal_stress_csv_path = results_dir / "cantilever_beam_tet10_nodal_stress.csv"
vtk_path = results_dir / "cantilever_beam_tet10.vtk"

# Export nodal displacements and stresses.
export_nodal_displacements_csv_3d(
    mesh=mesh,
    U=U,
    path=str(disp_csv_path),
)

export_tet10_element_stress_csv(
    mesh=mesh,
    U=U,
    path=str(elem_csv_path),
)

export_tet10_nodal_stress_csv(
    mesh=mesh,
    U=U,
    path=str(nodal_stress_csv_path),
)

# Export VTK for visualization.
export_vtk_from_csv_3d(
    mesh=mesh,
    disp_csv_path=str(disp_csv_path),
    elem_csv_path=str(elem_csv_path),
    nodal_stress_csv_path=str(nodal_stress_csv_path),
    vtk_path=str(vtk_path),
)

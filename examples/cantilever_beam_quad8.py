# Example: cantilever beam solved with Quad8 plane elements.

from fem.mesh_io import read_quad8_2d_abaqus
from fem.stiffness import compute_quad8_plane_element_stiffness
from fem.assemble import assemble_global_stiffness_sparse
from fem.helper import select_node_ids_by_x, select_edges_by_x
from fem.boundary import BoundaryCondition2D, build_load_vector, apply_dirichlet_bc
from fem.solve import solve_linear_system_sparse
from fem.post import (
    export_nodal_displacements_csv,
    export_quad8_plane_element_stress_csv,
    export_quad8_nodal_stress_csv,
    export_vtk_from_csv,
)

# Read mesh and material data from Abaqus input.
mesh = read_quad8_2d_abaqus(
    inp_path=r"examples\cantilever_beam_quad8.inp",
    material_path=r"examples\cantilever_beam_materials.csv",
    material_id=1,
    default_thickness=1.0,
)

# Assemble global stiffness matrix (sparse).
K = assemble_global_stiffness_sparse(
    num_dofs=mesh.num_dofs,
    num_elements=len(mesh.elements),
    get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
    compute_element_stiffness=lambda eid: compute_quad8_plane_element_stiffness(
        mesh,
        mesh.elements[eid],
        node_lookup={node.id: node for node in mesh.nodes},
    ),
)

# Select fixed boundary (x=0) and loaded boundary edges (x=200).
nodes_sel_fixed = select_node_ids_by_x(mesh, x_value=0.0)
print("Fixed nodes at x=0:", nodes_sel_fixed)
edges_sel_loaded = select_edges_by_x(mesh, x_value=200.0, tol=1e-8, boundary_only=True)
print("Loaded edges at x=200:", edges_sel_loaded)

# Define boundary conditions: fully fixed at x=0 and downward traction at x=200.
bc = BoundaryCondition2D()

for node_id in nodes_sel_fixed:
    bc.add_fixed_support(node_id=node_id, components=[0, 1], mesh=mesh)
for elem_id, local_edge, nids in edges_sel_loaded:
    bc.add_surface_traction(elem_id=elem_id, local_edge=local_edge, tx=0.0, ty=-2.0)

# Build load vector and apply Dirichlet constraints.
F = build_load_vector(mesh, bc)
K_mod, F_mod = apply_dirichlet_bc(K, F, bc)

# Solve for nodal displacements.
U = solve_linear_system_sparse(K_mod, F_mod)

# Export nodal displacements and stresses.
export_nodal_displacements_csv(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_quad8_nodal_displacements.csv",
)

export_quad8_plane_element_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_quad8_element_stress.csv",
)

export_quad8_nodal_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_quad8_nodal_stress.csv",
)

# Export VTK for visualization.
export_vtk_from_csv(
    mesh=mesh,
    disp_csv_path=r"results\cantilever_beam_quad8_nodal_displacements.csv",
    elem_csv_path=r"results\cantilever_beam_quad8_element_stress.csv",
    nodal_stress_csv_path=r"results\cantilever_beam_quad8_nodal_stress.csv",
    vtk_path=r"results\cantilever_beam_quad8_results.vtk",
)

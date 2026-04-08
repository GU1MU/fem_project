# Example: plate with a hole solved with Quad4 plane elements.

from fem.mesh_io import read_quad4_2d_abaqus
from fem.assemble import assemble_global_stiffness_sparse
from fem.stiffness import compute_quad4_plane_element_stiffness
from fem.helper import select_node_ids_by_x
from fem.boundary import BoundaryCondition2D, build_load_vector, apply_dirichlet_bc
from fem.solve import solve_linear_system_sparse
from fem.post import (
    export_nodal_displacements_csv,
    export_quad4_plane_element_stress_csv,
    export_quad4_nodal_stress_csv,
    export_vtk_from_csv,
)

# Read mesh and material data from Abaqus input.
mesh = read_quad4_2d_abaqus(
    inp_path=r"examples\plate_with_hole_quad4.inp",
    material_path=r"examples\plate_with_hole_materials.csv",
    material_id=1,
    default_thickness=1.0,
)

# Assemble global stiffness matrix (sparse).
K = assemble_global_stiffness_sparse(
    num_dofs=mesh.num_dofs,
    num_elements=len(mesh.elements),
    get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
    compute_element_stiffness=lambda eid: compute_quad4_plane_element_stiffness(
        mesh,
        mesh.elements[eid],
        node_lookup={node.id: node for node in mesh.nodes},
    ),
)

# Select fixed boundary (x=-50) and loaded boundary (x=50).
nodes_sel_fixed = select_node_ids_by_x(mesh, x_value=-50)
nodes_sel_loaded = select_node_ids_by_x(mesh, x_value=50)
print(nodes_sel_fixed)
print(nodes_sel_loaded)

# Define boundary conditions: fix both DOFs at x=-50 and apply x-traction at x=50.
bc = BoundaryCondition2D()

for node_id in nodes_sel_fixed:
    bc.add_fixed_support(node_id=node_id, components=[0, 1], mesh=mesh)
for node_id in nodes_sel_loaded:
    bc.add_nodal_force(node_id=node_id, component=0, value=5e3, mesh=mesh)

# Build load vector and apply Dirichlet constraints.
F = build_load_vector(mesh, bc)
K_mod, F_mod = apply_dirichlet_bc(K, F, bc)

# Solve for nodal displacements.
U = solve_linear_system_sparse(K_mod, F_mod)

# Export nodal displacements and stresses.
export_nodal_displacements_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad4_nodal_displacements.csv",
)

export_quad4_plane_element_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad4_element_stress.csv",
)

export_quad4_nodal_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad4_nodal_stress.csv",
)

# Export VTK for visualization.
export_vtk_from_csv(
    mesh=mesh,
    disp_csv_path=r"results\plate_with_hole_quad4_nodal_displacements.csv",
    elem_csv_path=r"results\plate_with_hole_quad4_element_stress.csv",
    nodal_stress_csv_path=r"results\plate_with_hole_quad4_nodal_stress.csv",
    vtk_path=r"results\plate_with_hole_quad4.vtk",
)

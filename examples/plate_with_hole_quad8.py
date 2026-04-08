# Example: plate with a hole solved with Quad8 plane elements.

from fem.mesh_io import read_quad8_2d_abaqus, read_quad4_2d_abaqus
from fem.stiffness import compute_quad8_plane_element_stiffness, compute_quad4_plane_element_stiffness
from fem.assemble import assemble_global_stiffness_sparse
from fem.helper import select_node_ids_by_x, select_node_ids_by_coord
from fem.boundary import BoundaryCondition2D, build_load_vector, apply_dirichlet_bc
from fem.solve import solve_linear_system_sparse
from fem.post import (
    export_nodal_displacements_csv,
    export_quad8_plane_element_stress_csv,
    export_quad8_nodal_stress_csv,
    export_vtk_from_csv,
    convert_nodal_solution_into_polar_coord,
    export_quad4_nodal_stress_csv,
    export_quad4_plane_element_stress_csv,
)

# Read mesh and material data from Abaqus input.
mesh = read_quad8_2d_abaqus(
    inp_path=r"examples\plate_with_hole_quad8.inp",
    material_path=r"examples\plate_with_hole_materials.csv",
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

# Select boundaries: fix x at left, fix y at one corner, and prescribe x-displacement on right.
nodes_sel_fixed_1 = select_node_ids_by_x(mesh, x_value=0.0)
nodes_sel_fixed_2 = select_node_ids_by_coord(mesh, x=0.0, y=0.0)
nodes_sel_loaded = select_node_ids_by_x(mesh, x_value=300.0)

print("Fixed nodes at x=0:", nodes_sel_fixed_1)
print("Fixed node at (0,0):", nodes_sel_fixed_2)
print("Loaded nodes at x=300:", nodes_sel_loaded)

# Define boundary conditions: fix ux at x=0, fix uy at (0,0), impose ux on x=300.
bc = BoundaryCondition2D()

for node_id in nodes_sel_fixed_1:
    bc.add_fixed_support(node_id=node_id, components=[0], mesh=mesh)
for node_id in nodes_sel_fixed_2:
    bc.add_fixed_support(node_id=node_id, components=[1], mesh=mesh)
for node_id in nodes_sel_loaded:
    bc.add_displacement(node_id=node_id, component=0, value=0.1, mesh=mesh)

# Build load vector and apply Dirichlet constraints.
F = build_load_vector(mesh, bc)
K_mod, F_mod = apply_dirichlet_bc(K, F, bc)

# Solve for nodal displacements.
U = solve_linear_system_sparse(K_mod, F_mod)

# Export nodal displacements and stresses.
export_nodal_displacements_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad8_nodal_displacements.csv",
)

export_quad8_nodal_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad8_nodal_stress.csv",
)

export_quad8_plane_element_stress_csv(
    mesh=mesh,
    U=U,
    path=r"results\plate_with_hole_quad8_element_stress.csv",
)

# Export VTK with polar stress/disp for visualization around the hole.
export_vtk_from_csv(
    mesh=mesh,
    disp_csv_path=r"results\plate_with_hole_quad8_nodal_displacements.csv",
    elem_csv_path=r"results\plate_with_hole_quad8_element_stress.csv",
    nodal_stress_csv_path=r"results\plate_with_hole_quad8_nodal_stress.csv",
    vtk_path=r"results\plate_with_hole_quad8_polar.vtk",
    polar=True,
    polar_center=(150.0, 50.0),
)

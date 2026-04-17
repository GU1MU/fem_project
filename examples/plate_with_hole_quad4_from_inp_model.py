from fem.assemble import assemble_global_stiffness_sparse
from fem.boundary import apply_dirichlet_bc, build_load_vector
from fem.mesh_io import read_abaqus_inp_as_model_data
from fem.solve import solve_linear_system_sparse
from fem.stiffness import compute_quad4_plane_element_stiffness


model_data = read_abaqus_inp_as_model_data(r"examples\plate_with_hole_quad4.inp")
mesh = model_data.mesh
bc = model_data.boundary
node_lookup = {node.id: node for node in mesh.nodes}

K = assemble_global_stiffness_sparse(
    num_dofs=mesh.num_dofs,
    num_elements=len(mesh.elements),
    get_element_dofs=lambda eid: mesh.element_dofs(mesh.elements[eid]),
    compute_element_stiffness=lambda eid: compute_quad4_plane_element_stiffness(
        mesh,
        mesh.elements[eid],
        node_lookup=node_lookup,
    ),
)
F = build_load_vector(mesh, bc)
K_mod, F_mod = apply_dirichlet_bc(K, F, bc)
U = solve_linear_system_sparse(K_mod, F_mod)

print(
    "Solved plate_with_hole_quad4.inp "
    f"with {len(mesh.nodes)} nodes, {len(mesh.elements)} elements, and {len(U)} DOFs."
)

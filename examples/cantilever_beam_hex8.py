# Example: cantilever beam solved with Hex8 solid elements.

from fem.io import read_hex8_3d_abaqus
from fem.assemble import assemble_global_stiffness_sparse
from fem import boundary, selection
from fem import solvers
import fem.post as post

# Read mesh and material data from Abaqus input.
mesh = read_hex8_3d_abaqus(
    inp_path=r"examples\cantilever_beam_hex8.inp",
    material_path=r"examples\cantilever_beam_materials.csv",
    material_id=1,
)

# Geometry extents
xs = [node.x for node in mesh.nodes]
zs = [node.z for node in mesh.nodes]
x_min, x_max = min(xs), max(xs)
z_min, z_max = min(zs), max(zs)

# Assemble global stiffness matrix (sparse).
K = assemble_global_stiffness_sparse(mesh)

# Select fixed boundary (x=x_min) and loaded nodes (x=x_max, z=z_max).
nodes_sel_fixed = selection.nodes.by_x(mesh, x_min)
nodes_sel_loaded = selection.nodes.by_coord(mesh, x=x_max, z=z_max)

print("Fixed nodes at x=x_min:", nodes_sel_fixed)
print("Loaded nodes at x=x_max, z=z_max:", nodes_sel_loaded)

# Define boundary conditions:
# fully fixed at x=x_min, concentrated nodal force on the free-end top edge.
bc = boundary.condition.BoundaryCondition()

for node_id in nodes_sel_fixed:
    bc.add_fixed_support(node_id=node_id, components=[0, 1, 2], mesh=mesh)

# Match Abaqus *Cload exactly: each loaded node gets Fz = -50.
for node_id in nodes_sel_loaded:
    bc.add_nodal_force(node_id=node_id, component=2, value=-50.0, mesh=mesh)

# Build load vector and apply Dirichlet constraints.
F = boundary.loads.build_load_vector(mesh, bc)
K_mod, F_mod = boundary.constraints.apply_dirichlet(K, F, bc)

# Solve for nodal displacements.
U = solvers.linear.solve(K_mod, F_mod)

# Export nodal displacements and stresses.
post.displacement.export.nodal(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_hex8_nodal_displacements.csv",
)

post.stress.export.element(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_hex8_element_stress.csv",
)

post.stress.export.nodal(
    mesh=mesh,
    U=U,
    path=r"results\cantilever_beam_hex8_nodal_stress.csv",
)

# Export VTK for visualization.
post.vtk.export.from_csv(
    mesh=mesh,
    disp_csv_path=r"results\cantilever_beam_hex8_nodal_displacements.csv",
    elem_csv_path=r"results\cantilever_beam_hex8_element_stress.csv",
    nodal_stress_csv_path=r"results\cantilever_beam_hex8_nodal_stress.csv",
    vtk_path=r"results\cantilever_beam_hex8.vtk",
)

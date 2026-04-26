# Example: cantilever beam solved with Hex8 solid elements.

from fem import abaqus
import fem.post as post

# Read Abaqus model data and solve its first analysis step.
model = abaqus.read(r"examples\cantilever_beam_hex8.inp")
mesh = model.mesh
step = model.get_step()

print("Model:", model.name)
print("Step:", step.name if step is not None else "None")
print("Node sets:", sorted(model.node_sets))
print("Element sets:", sorted(model.element_sets))

U = model.solve(step)

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

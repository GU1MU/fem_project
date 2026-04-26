# Example: cantilever beam solved with Hex8 solid elements.

from fem import abaqus, post, solvers

# Read Abaqus model data and solve its first analysis step.
model = abaqus.read(r"examples\cantilever_beam_hex8.inp")
step = model.get_step()

print("Model:", model.name)
print("Step:", step.name if step is not None else "None")
print("Node sets:", sorted(model.node_sets))
print("Element sets:", sorted(model.element_sets))

result = solvers.static_linear.solve(model, step)

# Export nodal displacements, stresses, and VTK for visualization.
post.vtk.export.from_result(result, output_dir=r"results")

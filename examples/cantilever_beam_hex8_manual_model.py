# Example: hand-written FEMModel built from an Abaqus Hex8 mesh.

from fem import selection
from fem.core import FEMModel
from fem.io import inp


# Read only mesh topology and coordinates from Abaqus input.
mesh = inp.read_hex8(r"examples\cantilever_beam_hex8.inp")

model = FEMModel.from_mesh(mesh, name="cantilever_beam_hex8_manual_model")

# Define material and section data by hand.
model.add_material("steel", E=220000.0, nu=0.3, rho=7800.0)
model.add_element_set("solid", selection.elements.all(mesh))
model.assign_section("solid", "steel")

# Define node sets from geometric selections.
xs = [node.x for node in mesh.nodes]
zs = [node.z for node in mesh.nodes]
x_min, x_max = min(xs), max(xs)
z_max = max(zs)

fixed = model.add_node_set("fixed", selection.nodes.by_x(mesh, x_min))
loaded = model.add_node_set("loaded", selection.nodes.by_coord(mesh, x=x_max, z=z_max))

print("Fixed nodes:", fixed.node_ids)
print("Loaded nodes:", loaded.node_ids)

# Define a static loading step by hand.
model.add_step("load")
model.add_displacement("load", "fixed", 1, 3)
model.add_nodal_load("load", "loaded", 3, -50.0)

result = model.run("load", output_dir=r"results")

print("Step:", result.step.name)
print("First loaded node displacement:", result.displacement(loaded.node_ids[0]))

result.export_vtk()

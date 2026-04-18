import os
import tempfile
import textwrap
import unittest

from fem.abaqus.convert import (
    build_boundary_from_inp_model as build_boundary_from_convert,
    build_mesh_from_inp_model as build_mesh_from_convert,
)
from fem.mesh_io import (
    build_boundary_from_inp_model as build_boundary_from_mesh_io,
    build_mesh_from_inp_model as build_mesh_from_mesh_io,
    read_abaqus_inp_as_model_data,
    read_abaqus_inp_model,
)


class AbaqusModuleConversionTests(unittest.TestCase):
    def test_convert_module_matches_mesh_io_public_wrappers(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.0, 1.0
            *Element, type=CPS3, elset=EALL
            1, 1, 2, 3
            *Nset, nset=FIXED
            1
            *Nset, nset=LOADNODE
            3
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Density
            7.85e-09
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Boundary
            FIXED, ENCASTRE
            *Cload
            LOADNODE, 2, -25.0
            *End Step
            """
        )

        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".inp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(inp_text)
            inp_path = handle.name

        try:
            model = read_abaqus_inp_model(inp_path)
            mesh_from_convert = build_mesh_from_convert(model)
            mesh_from_mesh_io = build_mesh_from_mesh_io(model)
            boundary_from_convert = build_boundary_from_convert(model, mesh_from_convert)
            boundary_from_mesh_io = build_boundary_from_mesh_io(model, mesh_from_mesh_io)
            model_data = read_abaqus_inp_as_model_data(inp_path)
        finally:
            os.remove(inp_path)

        self.assertEqual(type(mesh_from_convert), type(mesh_from_mesh_io))
        self.assertEqual(mesh_from_convert.nodes, mesh_from_mesh_io.nodes)
        self.assertEqual(mesh_from_convert.elements, mesh_from_mesh_io.elements)

        self.assertEqual(type(boundary_from_convert), type(boundary_from_mesh_io))
        self.assertEqual(
            boundary_from_convert.prescribed_displacements,
            boundary_from_mesh_io.prescribed_displacements,
        )
        self.assertEqual(
            boundary_from_convert.nodal_forces,
            boundary_from_mesh_io.nodal_forces,
        )

        self.assertEqual(model_data.mesh.nodes, mesh_from_mesh_io.nodes)
        self.assertEqual(model_data.mesh.elements, mesh_from_mesh_io.elements)
        self.assertEqual(
            model_data.boundary.prescribed_displacements,
            boundary_from_mesh_io.prescribed_displacements,
        )
        self.assertEqual(
            model_data.boundary.nodal_forces,
            boundary_from_mesh_io.nodal_forces,
        )
        self.assertEqual(model_data.model, model)
        self.assertEqual(model_data.step.name, "LOAD-STEP")


if __name__ == "__main__":
    unittest.main()

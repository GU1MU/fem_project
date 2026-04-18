import os
import tempfile
import textwrap
import unittest

import fem.abaqus as abaqus
import fem.abaqus.parser as abaqus_parser
import fem.mesh_io as mesh_io


class AbaqusModuleParserTests(unittest.TestCase):
    def test_mesh_io_is_public_boundary_and_delegates_to_internal_parser(self) -> None:
        # Architecture: mesh_io is the public input boundary. The low-level
        # implementation can live elsewhere but mesh_io should own the public symbol.
        self.assertEqual(mesh_io.read_abaqus_inp_model.__module__, "fem.mesh_io")
        self.assertEqual(
            abaqus_parser.read_abaqus_inp_model.__module__, "fem.abaqus.parser"
        )
        self.assertIsNot(mesh_io.read_abaqus_inp_model, abaqus_parser.read_abaqus_inp_model)

        # We should not advertise a second public-looking entry point.
        self.assertFalse(hasattr(abaqus, "read_abaqus_inp_model"))

        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 0.0, 1.0, 0.0
            4, 0.0, 0.0, 1.0
            *Element, type=C3D4, elset=SOLID-SET
            1, 1, 2, 3, 4
            *Nset, nset=FIXED
            1, 2
            *Elset, elset=SOLID-SET
            1
            *Solid Section, elset=SOLID-SET, material=STEEL
            0.25,
            *End Part
            *Assembly, name=ASSEMBLY
            *Instance, name=PART-1-1, part=PART-1
            *End Instance
            *End Assembly
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Density
            7.85e-09
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0, 1e-05, 0.1
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
            mesh_io_model = mesh_io.read_abaqus_inp_model(inp_path)
            module_model = abaqus_parser.read_abaqus_inp_model(inp_path)
        finally:
            os.remove(inp_path)

        self.assertEqual(module_model.part_name, mesh_io_model.part_name)
        self.assertEqual(module_model.instance_name, mesh_io_model.instance_name)
        self.assertEqual(module_model.nodes, mesh_io_model.nodes)
        self.assertEqual(module_model.element_blocks, mesh_io_model.element_blocks)
        self.assertEqual(module_model.nsets, mesh_io_model.nsets)
        self.assertEqual(module_model.elsets, mesh_io_model.elsets)
        self.assertEqual(module_model.materials, mesh_io_model.materials)
        self.assertEqual(module_model.sections, mesh_io_model.sections)
        self.assertEqual(module_model.steps, mesh_io_model.steps)


if __name__ == "__main__":
    unittest.main()

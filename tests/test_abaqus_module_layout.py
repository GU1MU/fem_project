import os
import tempfile
import textwrap
import unittest

import fem.abaqus as abaqus
import fem.mesh_io as mesh_io
from fem.abaqus.model import AbaqusInpModel, InpModelData, InpNode
from fem.mesh_io import (
    AbaqusInpModel as MeshIoAbaqusInpModel,
    InpModelData as MeshIoInpModelData,
    InpNode as MeshIoInpNode,
    read_abaqus_inp_as_model_data,
    read_abaqus_inp_model,
)


class AbaqusModuleLayoutTests(unittest.TestCase):
    def test_package_and_mesh_io_re_exports_remain_available(self) -> None:
        self.assertIs(abaqus.AbaqusInpModel, AbaqusInpModel)
        self.assertIs(abaqus.InpModelData, InpModelData)
        self.assertIs(abaqus.InpNode, InpNode)

        self.assertIs(MeshIoAbaqusInpModel, AbaqusInpModel)
        self.assertIs(MeshIoInpModelData, InpModelData)
        self.assertIs(MeshIoInpNode, InpNode)

        self.assertEqual(mesh_io.AbaqusInpModel.__module__, "fem.mesh_io")
        self.assertEqual(mesh_io.InpModelData.__module__, "fem.mesh_io")
        self.assertEqual(mesh_io.InpNode.__module__, "fem.mesh_io")

    def test_public_reader_returns_internal_module_dataclasses(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 0.0, 1.0, 0.0
            4, 0.0, 0.0, 1.0
            *Element, type=C3D4, elset=EALL
            1, 1, 2, 3, 4
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            1.0,
            *End Part
            *Assembly, name=ASSEMBLY
            *Instance, name=PART-1-1, part=PART-1
            *End Instance
            *End Assembly
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Step, name=STEP-1
            *Static
            0.1, 1.0
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
            model_data = read_abaqus_inp_as_model_data(inp_path)
        finally:
            os.remove(inp_path)

        self.assertIsInstance(model, AbaqusInpModel)
        self.assertIsInstance(model.nodes[1], InpNode)
        self.assertIsInstance(model_data, InpModelData)
        self.assertIsInstance(model_data.model, AbaqusInpModel)
        self.assertIsInstance(model_data.model.nodes[1], InpNode)
        self.assertEqual(type(model).__module__, "fem.mesh_io")
        self.assertEqual(type(model_data).__module__, "fem.mesh_io")
        self.assertEqual(type(model.nodes[1]).__module__, "fem.mesh_io")


if __name__ == "__main__":
    unittest.main()

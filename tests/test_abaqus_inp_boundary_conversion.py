import os
import tempfile
import textwrap
import unittest

from fem.boundary import BoundaryCondition2D, BoundaryCondition3D
from fem.mesh import Element3D, Node3D, TetMesh3D
from fem.mesh_io import (
    InpModelData,
    build_boundary_from_inp_model,
    build_mesh_from_inp_model,
    read_abaqus_inp_as_model_data,
    read_abaqus_inp_model,
)


class AbaqusInpBoundaryConversionTests(unittest.TestCase):
    def _write_inp(self, inp_text: str) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w",
            suffix=".inp",
            delete=False,
            encoding="utf-8",
        )
        handle.write(inp_text)
        handle.close()
        return handle.name

    def _read_model(self, inp_text: str):
        inp_path = self._write_inp(inp_text)
        try:
            return read_abaqus_inp_model(inp_path)
        finally:
            os.remove(inp_path)

    def test_converts_boundary_and_cload_specs_into_boundary_condition_2d(self) -> None:
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
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Boundary
            FIXED, ENCASTRE
            LOADNODE, 1, 1, 0.125
            *Cload
            LOADNODE, 2, -25.0
            *End Step
            """
        )

        model = self._read_model(inp_text)
        inp_path = self._write_inp(inp_text)
        try:
            model_data = read_abaqus_inp_as_model_data(inp_path)
            boundary = model_data.boundary
            mesh = model_data.mesh
        finally:
            os.remove(inp_path)

        self.assertIsInstance(boundary, BoundaryCondition2D)
        self.assertEqual(model_data.step.name, "LOAD-STEP")
        self.assertEqual(boundary.prescribed_displacements[mesh.global_dof(1, 0)], 0.0)
        self.assertEqual(boundary.prescribed_displacements[mesh.global_dof(1, 1)], 0.0)
        self.assertEqual(boundary.prescribed_displacements[mesh.global_dof(3, 0)], 0.125)
        self.assertEqual(boundary.nodal_forces[mesh.global_dof(3, 1)], -25.0)

    def test_convenience_wrapper_returns_model_mesh_boundary_and_selected_step(self) -> None:
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
            *Step, name=STEP-1
            *Static
            0.1, 1.0
            *Boundary
            FIXED, 1, 1, 0.0
            *End Step
            *Step, name=STEP-2
            *Static
            0.1, 1.0
            *Boundary
            FIXED, ENCASTRE
            *Cload
            LOADNODE, 1, 12.0
            *End Step
            """
        )

        inp_path = self._write_inp(inp_text)
        try:
            model_data = read_abaqus_inp_as_model_data(inp_path, step_name="STEP-2")
        finally:
            os.remove(inp_path)

        self.assertIsInstance(model_data, InpModelData)
        self.assertEqual(model_data.step.name, "STEP-2")
        self.assertEqual(model_data.model.steps[1].name, "STEP-2")
        self.assertEqual(model_data.boundary.nodal_forces[model_data.mesh.global_dof(3, 0)], 12.0)

    def test_expands_encastre_using_mesh_dofs_for_3d_meshes(self) -> None:
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
            *Nset, nset=FIXED
            1
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Boundary
            FIXED, ENCASTRE
            *End Step
            """
        )

        model = self._read_model(inp_text)
        mesh = TetMesh3D(
            nodes=[
                Node3D(id=1, x=0.0, y=0.0, z=0.0),
                Node3D(id=2, x=1.0, y=0.0, z=0.0),
                Node3D(id=3, x=0.0, y=1.0, z=0.0),
                Node3D(id=4, x=0.0, y=0.0, z=1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Tet4",
                    props={"E": 210000.0, "nu": 0.3},
                )
            ],
        )

        boundary = build_boundary_from_inp_model(model, mesh)

        self.assertIsInstance(boundary, BoundaryCondition3D)
        self.assertEqual(
            boundary.prescribed_displacements,
            {
                mesh.global_dof(1, 0): 0.0,
                mesh.global_dof(1, 1): 0.0,
                mesh.global_dof(1, 2): 0.0,
            },
        )

    def test_rejects_unsupported_dload_conversion_instead_of_ignoring_it(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.0, 1.0
            *Element, type=CPS3, elset=EALL
            1, 1, 2, 3
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Dload
            EALL, P1, 5.0
            *End Step
            """
        )

        inp_path = self._write_inp(inp_text)
        try:
            with self.assertRaisesRegex(ValueError, r"Unsupported \*Dload conversion"):
                read_abaqus_inp_as_model_data(inp_path)
        finally:
            os.remove(inp_path)

    def test_accumulates_repeated_3d_cload_entries(self) -> None:
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
            *Nset, nset=LOADNODE
            2
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Cload
            LOADNODE, 3, 10.0
            LOADNODE, 3, -3.5
            *End Step
            """
        )

        model = self._read_model(inp_text)
        mesh = TetMesh3D(
            nodes=[
                Node3D(id=1, x=0.0, y=0.0, z=0.0),
                Node3D(id=2, x=1.0, y=0.0, z=0.0),
                Node3D(id=3, x=0.0, y=1.0, z=0.0),
                Node3D(id=4, x=0.0, y=0.0, z=1.0),
            ],
            elements=[
                Element3D(
                    id=1,
                    node_ids=[1, 2, 3, 4],
                    type="Tet4",
                    props={"E": 210000.0, "nu": 0.3, "rho": 7850.0},
                )
            ],
        )

        boundary = build_boundary_from_inp_model(model, mesh)

        self.assertEqual(boundary.nodal_forces[mesh.global_dof(2, 2)], 6.5)

    def test_rejects_reversed_boundary_dof_range(self) -> None:
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
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Boundary
            FIXED, 2, 1, 0.0
            *End Step
            """
        )

        inp_path = self._write_inp(inp_text)
        try:
            with self.assertRaisesRegex(ValueError, r"first_dof > last_dof"):
                read_abaqus_inp_as_model_data(inp_path)
        finally:
            os.remove(inp_path)

    def test_convenience_wrapper_supports_valid_3d_solid_inp(self) -> None:
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
            *Nset, nset=FIXED
            1
            *Nset, nset=LOADNODE
            2
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            1.0,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            *Density
            7850.0
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0
            *Boundary
            FIXED, ENCASTRE
            *Cload
            LOADNODE, 1, 25.0
            *End Step
            """
        )

        inp_path = self._write_inp(inp_text)
        try:
            model_data = read_abaqus_inp_as_model_data(inp_path)
        finally:
            os.remove(inp_path)

        self.assertIsInstance(model_data, InpModelData)
        self.assertIsInstance(model_data.boundary, BoundaryCondition3D)
        self.assertIsInstance(model_data.mesh, TetMesh3D)
        self.assertEqual(model_data.mesh.elements[0].type, "Tet4")
        self.assertEqual(model_data.boundary.nodal_forces[model_data.mesh.global_dof(2, 0)], 25.0)
        self.assertEqual(model_data.boundary.prescribed_displacements[model_data.mesh.global_dof(1, 2)], 0.0)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import textwrap
import unittest

from fem.mesh import PlaneMesh2D
from fem.mesh_io import build_mesh_from_inp_model, read_abaqus_inp_model


class BuildMeshFromInpModelTests(unittest.TestCase):
    def _read_model(self, inp_text: str):
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".inp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(inp_text)
            inp_path = handle.name

        try:
            return read_abaqus_inp_model(inp_path)
        finally:
            os.remove(inp_path)

    def test_builds_plane_mesh_and_maps_section_material_properties(self) -> None:
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
            *Density
            7.85e-09
            """
        )

        model = self._read_model(inp_text)
        mesh = build_mesh_from_inp_model(model)

        self.assertIsInstance(mesh, PlaneMesh2D)
        self.assertEqual([node.id for node in mesh.nodes], [1, 2, 3])
        self.assertEqual(len(mesh.elements), 1)

        element = mesh.elements[0]
        self.assertEqual(element.id, 1)
        self.assertEqual(element.node_ids, [1, 2, 3])
        self.assertEqual(element.type, "Tri3Plane")
        self.assertEqual(element.props["material_name"], "STEEL")
        self.assertEqual(element.props["section_type"], "SOLID SECTION")
        self.assertEqual(element.props["thickness"], 0.25)
        self.assertEqual(element.props["E"], 210000.0)
        self.assertEqual(element.props["nu"], 0.3)
        self.assertEqual(element.props["rho"], 7.85e-09)

    def test_rejects_incompatible_mixed_element_families(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.0, 1.0
            4, 1.0, 1.0
            *Element, type=CPS3, elset=TRIS
            1, 1, 2, 3
            *Element, type=C3D4, elset=SOLIDS
            2, 1, 2, 3, 4
            *Elset, elset=TRIS
            1
            *Elset, elset=SOLIDS
            2
            *Solid Section, elset=TRIS, material=STEEL
            0.25,
            *Solid Section, elset=SOLIDS, material=STEEL
            1.0,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"incompatible mixed element families",
        ):
            build_mesh_from_inp_model(model)

    def test_ignores_assembly_scope_nodes_that_reuse_part_node_ids(self) -> None:
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
            *End Part
            *Assembly, name=ASSEMBLY
            *Instance, name=PART-1-1, part=PART-1
            *End Instance
            *Node
            1, 99.0, 99.0, 99.0
            *End Assembly
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)
        self.assertEqual(model.nodes[1].coordinates, (0.0, 0.0))

        mesh = build_mesh_from_inp_model(model)
        self.assertEqual((mesh.nodes[0].x, mesh.nodes[0].y), (0.0, 0.0))

    def test_rejects_missing_node_connectivity(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            *Element, type=CPS3, elset=EALL
            1, 1, 2, 3
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"Element 1 references missing node 3",
        ):
            build_mesh_from_inp_model(model)

    def test_rejects_malformed_2d_node_dimensionality(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0
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
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"Node 1 must have at least 2 coordinates",
        ):
            build_mesh_from_inp_model(model)

    def test_fixes_inverted_quad4_connectivity_before_returning_mesh(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *Element, type=CPS4, elset=EALL
            1, 1, 4, 3, 2
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)
        mesh = build_mesh_from_inp_model(model)

        self.assertIsInstance(mesh, PlaneMesh2D)
        self.assertEqual(len(mesh.elements), 1)
        self.assertEqual(mesh.elements[0].type, "Quad4Plane")
        self.assertEqual(mesh.elements[0].node_ids, [1, 2, 3, 4])

    def test_rejects_malformed_tri3_connectivity_arity(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.0, 1.0
            4, 1.0, 1.0
            *Element, type=CPS3, elset=EALL
            1, 1, 2, 3, 4
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"Element 1 must have exactly 3 node IDs for plane_tri3",
        ):
            build_mesh_from_inp_model(model)

    def test_rejects_malformed_quad4_connectivity_arity(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            5, 2.0, 0.0
            *Element, type=CPS4, elset=EALL
            1, 1, 2, 3, 4, 5
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"Element 1 must have exactly 4 node IDs for plane_quad4",
        ):
            build_mesh_from_inp_model(model)

    def test_rejects_malformed_quad8_connectivity_arity(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            5, 0.5, 0.0
            6, 1.0, 0.5
            7, 0.5, 1.0
            8, 0.0, 0.5
            *Element, type=CPS8, elset=EALL
            1, 1, 2, 3, 4, 5, 6, 7
            *Elset, elset=EALL
            1
            *Solid Section, elset=EALL, material=STEEL
            0.25,
            *Material, name=STEEL
            *Elastic
            210000.0, 0.3
            """
        )

        model = self._read_model(inp_text)

        with self.assertRaisesRegex(
            ValueError,
            r"Element 1 must have exactly 8 node IDs for plane_quad8",
        ):
            build_mesh_from_inp_model(model)


if __name__ == "__main__":
    unittest.main()

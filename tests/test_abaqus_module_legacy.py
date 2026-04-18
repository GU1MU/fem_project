import unittest

import fem.mesh_io as mesh_io
from fem.abaqus import legacy as abaqus_legacy


class AbaqusLegacyModuleTests(unittest.TestCase):
    @staticmethod
    def _node_signature(node):
        return (
            node.id,
            getattr(node, "x", None),
            getattr(node, "y", None),
            getattr(node, "z", None),
        )

    @staticmethod
    def _element_signature(element):
        return (
            element.id,
            tuple(element.node_ids),
            element.type,
            tuple(sorted(element.props.items())),
        )

    def _assert_public_reader_matches_internal_reader(
        self,
        *,
        public_reader,
        internal_reader,
        reader_kwargs,
        expected_node_count,
        expected_element_count,
        expected_element_type,
    ) -> None:
        public_mesh = public_reader(**reader_kwargs)
        internal_mesh = internal_reader(
            **reader_kwargs,
            read_materials_as_dict=mesh_io.read_materials_as_dict,
            get_float_from_material=mesh_io._get_float_from_material,
        )

        self.assertEqual(len(public_mesh.nodes), expected_node_count)
        self.assertEqual(len(public_mesh.elements), expected_element_count)
        self.assertEqual(public_mesh.elements[0].type, expected_element_type)
        self.assertEqual(
            [self._node_signature(node) for node in public_mesh.nodes],
            [self._node_signature(node) for node in internal_mesh.nodes],
        )
        self.assertEqual(
            [self._element_signature(element) for element in public_mesh.elements],
            [self._element_signature(element) for element in internal_mesh.elements],
        )

    def test_legacy_module_keeps_specialized_readers_internal(self) -> None:
        reader_names = (
            "tri3_2d_abaqus",
            "quad4_2d_abaqus",
            "quad8_2d_abaqus",
            "tet10_3d_abaqus",
            "tet4_3d_abaqus",
            "hex8_3d_abaqus",
        )

        for reader_name in reader_names:
            with self.subTest(reader_name=reader_name):
                self.assertFalse(hasattr(abaqus_legacy, f"read_{reader_name}"))
                self.assertTrue(hasattr(abaqus_legacy, f"_read_{reader_name}"))

    def test_mesh_io_remains_public_boundary_for_quad4_reader(self) -> None:
        self._assert_public_reader_matches_internal_reader(
            public_reader=mesh_io.read_quad4_2d_abaqus,
            internal_reader=abaqus_legacy._read_quad4_2d_abaqus,
            reader_kwargs={
                "inp_path": "examples/plate_with_hole_quad4.inp",
                "material_id": 1,
                "material_path": "examples/plate_with_hole_materials.csv",
                "default_thickness": 1.0,
            },
            expected_node_count=960,
            expected_element_count=894,
            expected_element_type="Quad4Plane",
        )

    def test_mesh_io_remains_public_boundary_for_other_legacy_readers(self) -> None:
        cases = (
            (
                "quad8",
                mesh_io.read_quad8_2d_abaqus,
                abaqus_legacy._read_quad8_2d_abaqus,
                {
                    "inp_path": "examples/plate_with_hole_quad8.inp",
                    "material_id": 1,
                    "material_path": "examples/plate_with_hole_materials.csv",
                    "default_thickness": 1.0,
                },
                7542,
                2452,
                "Quad8Plane",
            ),
            (
                "tet4",
                mesh_io.read_tet4_3d_abaqus,
                abaqus_legacy._read_tet4_3d_abaqus,
                {
                    "inp_path": "examples/cantilever_beam_tet4.inp",
                    "material_id": 1,
                    "material_path": "examples/cantilever_beam_materials.csv",
                },
                1206,
                4957,
                "Tet4",
            ),
            (
                "tet10",
                mesh_io.read_tet10_3d_abaqus,
                abaqus_legacy._read_tet10_3d_abaqus,
                {
                    "inp_path": "examples/cantilever_beam_tet10.inp",
                    "material_id": 1,
                    "material_path": "examples/cantilever_beam_materials.csv",
                },
                8040,
                4957,
                "Tet10",
            ),
            (
                "hex8",
                mesh_io.read_hex8_3d_abaqus,
                abaqus_legacy._read_hex8_3d_abaqus,
                {
                    "inp_path": "examples/cantilever_beam_hex8.inp",
                    "material_id": 1,
                    "material_path": "examples/cantilever_beam_materials.csv",
                },
                425,
                256,
                "Hex8",
            ),
        )

        for (
            case_name,
            public_reader,
            internal_reader,
            reader_kwargs,
            expected_node_count,
            expected_element_count,
            expected_element_type,
        ) in cases:
            with self.subTest(case_name=case_name):
                self._assert_public_reader_matches_internal_reader(
                    public_reader=public_reader,
                    internal_reader=internal_reader,
                    reader_kwargs=reader_kwargs,
                    expected_node_count=expected_node_count,
                    expected_element_count=expected_element_count,
                    expected_element_type=expected_element_type,
                )


if __name__ == "__main__":
    unittest.main()

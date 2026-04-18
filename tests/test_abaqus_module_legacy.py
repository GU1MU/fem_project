import unittest

from fem.abaqus import legacy as abaqus_legacy
from fem.mesh_io import read_quad4_2d_abaqus


class AbaqusLegacyModuleTests(unittest.TestCase):
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
        facade_mesh = read_quad4_2d_abaqus(
            inp_path="examples/plate_with_hole_quad4.inp",
            material_id=1,
            material_path="examples/plate_with_hole_materials.csv",
            default_thickness=1.0,
        )

        self.assertEqual(len(facade_mesh.nodes), 960)
        self.assertEqual(len(facade_mesh.elements), 894)
        self.assertEqual(facade_mesh.elements[0].type, "Quad4Plane")


if __name__ == "__main__":
    unittest.main()

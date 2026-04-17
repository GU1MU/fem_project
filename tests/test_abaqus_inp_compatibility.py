import unittest

from fem.mesh_io import read_abaqus_inp_as_model_data, read_quad4_2d_abaqus


class AbaqusInpCompatibilityTests(unittest.TestCase):
    def test_read_abaqus_inp_as_model_data_reads_existing_quad4_example(self) -> None:
        model_data = read_abaqus_inp_as_model_data("examples/plate_with_hole_quad4.inp")

        self.assertEqual(model_data.step.name, "Step-1")
        self.assertEqual(len(model_data.mesh.nodes), 960)
        self.assertEqual(len(model_data.mesh.elements), 894)
        self.assertGreater(len(model_data.boundary.prescribed_displacements), 0)
        self.assertGreater(len(model_data.boundary.nodal_forces), 0)

    def test_existing_quad4_reader_still_returns_same_mesh_shape(self) -> None:
        mesh = read_quad4_2d_abaqus(
            inp_path="examples/plate_with_hole_quad4.inp",
            material_id=1,
            material_path="examples/plate_with_hole_materials.csv",
            default_thickness=1.0,
        )

        self.assertEqual(len(mesh.nodes), 960)
        self.assertEqual(len(mesh.elements), 894)


if __name__ == "__main__":
    unittest.main()

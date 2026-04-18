import unittest

from fem.mesh_io import (
    _get_float_from_material,
    read_materials_as_dict,
    read_quad4_2d_abaqus,
)
from fem.abaqus.legacy import read_quad4_2d_abaqus as read_quad4_2d_abaqus_legacy


class AbaqusLegacyModuleTests(unittest.TestCase):
    def test_quad4_legacy_reader_matches_mesh_io_facade_on_existing_example(self) -> None:
        legacy_mesh = read_quad4_2d_abaqus_legacy(
            inp_path="examples/plate_with_hole_quad4.inp",
            material_id=1,
            material_path="examples/plate_with_hole_materials.csv",
            default_thickness=1.0,
            read_materials_as_dict=read_materials_as_dict,
            get_float_from_material=_get_float_from_material,
        )

        facade_mesh = read_quad4_2d_abaqus(
            inp_path="examples/plate_with_hole_quad4.inp",
            material_id=1,
            material_path="examples/plate_with_hole_materials.csv",
            default_thickness=1.0,
        )

        self.assertEqual(type(facade_mesh), type(legacy_mesh))
        self.assertEqual(len(facade_mesh.nodes), len(legacy_mesh.nodes))
        self.assertEqual(len(facade_mesh.elements), len(legacy_mesh.elements))

        self.assertEqual(
            [(node.id, node.x, node.y) for node in facade_mesh.nodes[:5]],
            [(node.id, node.x, node.y) for node in legacy_mesh.nodes[:5]],
        )
        self.assertEqual(
            [
                (element.id, tuple(element.node_ids), dict(element.props))
                for element in facade_mesh.elements[:5]
            ],
            [
                (element.id, tuple(element.node_ids), dict(element.props))
                for element in legacy_mesh.elements[:5]
            ],
        )


if __name__ == "__main__":
    unittest.main()

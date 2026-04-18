import unittest

import fem.abaqus.convert as abaqus_convert
import fem.mesh_io as mesh_io


class AbaqusModuleConversionTests(unittest.TestCase):
    def test_convert_module_keeps_conversion_entry_points_internal(self) -> None:
        self.assertFalse(hasattr(abaqus_convert, "build_mesh_from_inp_model"))
        self.assertFalse(hasattr(abaqus_convert, "build_boundary_from_inp_model"))
        self.assertFalse(hasattr(abaqus_convert, "read_abaqus_inp_as_model_data"))

    def test_mesh_io_no_longer_keeps_internal_abaqus_conversion_helpers(self) -> None:
        self.assertFalse(hasattr(mesh_io, "_build_element_id_to_elset_names"))
        self.assertFalse(hasattr(mesh_io, "_resolve_inp_section_material_props"))
        self.assertFalse(hasattr(mesh_io, "_select_inp_step"))
        self.assertFalse(hasattr(mesh_io, "_resolve_inp_set_target"))
        self.assertFalse(hasattr(mesh_io, "_resolve_inp_target_ids"))
        self.assertFalse(hasattr(mesh_io, "_validate_abaqus_dof"))
        self.assertFalse(hasattr(mesh_io, "_classify_inp_element_family"))
        self.assertFalse(hasattr(mesh_io, "_build_plane_nodes_from_inp_model"))
        self.assertFalse(hasattr(mesh_io, "_build_solid_nodes_from_inp_model"))


if __name__ == "__main__":
    unittest.main()

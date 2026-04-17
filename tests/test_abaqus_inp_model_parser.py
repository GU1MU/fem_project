import os
import tempfile
import textwrap
import unittest

from fem.mesh_io import read_abaqus_inp_model


class ReadAbaqusInpModelTests(unittest.TestCase):
    def test_reads_supported_keywords_into_intermediate_model(self) -> None:
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
            model = read_abaqus_inp_model(inp_path)
        finally:
            os.remove(inp_path)

        self.assertEqual(model.part_name, "PART-1")
        self.assertEqual(model.instance_name, "PART-1-1")

        self.assertEqual(sorted(model.nodes.keys()), [1, 2, 3, 4])
        self.assertEqual(model.nodes[1].id, 1)
        self.assertEqual(model.nodes[1].coordinates, (0.0, 0.0, 0.0))

        self.assertEqual(len(model.element_blocks), 1)
        block = model.element_blocks[0]
        self.assertEqual(block.element_type, "C3D4")
        self.assertEqual(block.elset, "SOLID-SET")
        self.assertEqual(len(block.elements), 1)
        self.assertEqual(block.elements[0].id, 1)
        self.assertEqual(block.elements[0].node_ids, (1, 2, 3, 4))

        self.assertEqual(model.nsets, {"part": {"FIXED": [1, 2]}, "assembly": {}})
        self.assertEqual(model.elsets, {"part": {"SOLID-SET": [1]}, "assembly": {}})

        self.assertEqual(len(model.materials), 1)
        material = model.materials["STEEL"]
        self.assertEqual(material.name, "STEEL")
        self.assertEqual(material.elastic, (210000.0, 0.3))
        self.assertEqual(material.density, 7.85e-09)

        self.assertEqual(len(model.sections), 1)
        section = model.sections[0]
        self.assertEqual(section.section_type, "SOLID SECTION")
        self.assertEqual(section.parameters["elset"], "SOLID-SET")
        self.assertEqual(section.parameters["material"], "STEEL")
        self.assertEqual(section.material_name, "STEEL")
        self.assertEqual(section.elset, "SOLID-SET")
        self.assertEqual(section.data, [("0.25",)])

        self.assertEqual(len(model.steps), 1)
        step = model.steps[0]
        self.assertEqual(step.name, "LOAD-STEP")
        self.assertEqual(step.static_parameters, (0.1, 1.0, 1e-05, 0.1))

    def test_expands_generate_and_preserves_set_scope(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 2.0, 0.0, 0.0
            4, 3.0, 0.0, 0.0
            *Element, type=C3D4, elset=COMMON
            1, 1, 2, 3, 4
            *Nset, nset=COMMON, generate
            1, 4, 1
            *Elset, elset=COMMON, generate
            1, 3, 1
            *End Part
            *Assembly, name=ASSEMBLY
            *Instance, name=PART-1-1, part=PART-1
            *End Instance
            *Nset, nset=COMMON, generate
            10, 14, 2
            *Elset, elset=COMMON
            9, 11
            *End Assembly
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
        finally:
            os.remove(inp_path)

        self.assertEqual(
            model.nsets,
            {
                "part": {"COMMON": [1, 2, 3, 4]},
                "assembly": {"COMMON": [10, 12, 14]},
            },
        )
        self.assertEqual(
            model.elsets,
            {
                "part": {"COMMON": [1, 2, 3]},
                "assembly": {"COMMON": [9, 11]},
            },
        )

    def test_rejects_malformed_node_element_and_set_data(self) -> None:
        cases = [
            (
                "node",
                textwrap.dedent(
                    """\
                    *Part, name=PART-1
                    *Node
                    1, 0.0, , 0.0
                    """
                ),
                r"Malformed \*Node",
            ),
            (
                "element",
                textwrap.dedent(
                    """\
                    *Part, name=PART-1
                    *Node
                    1, 0.0, 0.0, 0.0
                    2, 1.0, 0.0, 0.0
                    3, 0.0, 1.0, 0.0
                    4, 0.0, 0.0, 1.0
                    *Element, type=C3D4, elset=EALL
                    1, 1, 2, , 4
                    """
                ),
                r"Malformed \*Element",
            ),
            (
                "set",
                textwrap.dedent(
                    """\
                    *Part, name=PART-1
                    *Node
                    1, 0.0, 0.0, 0.0
                    *Nset, nset=BAD
                    1, , 3
                    """
                ),
                r"Malformed \*Nset",
            ),
        ]

        for name, inp_text, message in cases:
            with self.subTest(name=name):
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".inp",
                    delete=False,
                    encoding="utf-8",
                ) as handle:
                    handle.write(inp_text)
                    inp_path = handle.name

                try:
                    with self.assertRaisesRegex(ValueError, message):
                        read_abaqus_inp_model(inp_path)
                finally:
                    os.remove(inp_path)

    def test_accepts_trailing_comma_in_set_records(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0, 0.0
            *Nset, nset=SINGLE
            5,
            *Elset, elset=EALL
            9,
            *End Part
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
        finally:
            os.remove(inp_path)

        self.assertEqual(model.nsets, {"part": {"SINGLE": [5]}, "assembly": {}})
        self.assertEqual(model.elsets, {"part": {"EALL": [9]}, "assembly": {}})

    def test_accepts_trailing_comma_in_density_record(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Material, name=STEEL
            *Density
            7850.,
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
        finally:
            os.remove(inp_path)

        self.assertEqual(model.materials["STEEL"].density, 7850.0)


if __name__ == "__main__":
    unittest.main()

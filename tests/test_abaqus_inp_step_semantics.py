import os
import tempfile
import textwrap
import unittest

from fem.mesh_io import read_abaqus_inp_model


class ReadAbaqusInpStepSemanticsTests(unittest.TestCase):
    def test_reads_step_level_loads_constraints_and_unhandled_specs(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Part, name=PART-1
            *Node
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 1.0, 1.0, 0.0
            4, 0.0, 1.0, 0.0
            *Element, type=CPS4, elset=EALL
            1, 1, 2, 3, 4
            *Nset, nset=FIXED
            1,
            *Nset, nset=LOADNODE
            3,
            *Elset, elset=EALL
            1,
            *End Part
            *Assembly, name=ASSEMBLY
            *Instance, name=PART-1-1, part=PART-1
            *End Instance
            *End Assembly
            *Step, name=LOAD-STEP
            *Static
            0.1, 1.0, 1e-05, 0.1
            *Boundary, op=NEW, amplitude=RAMP
            FIXED, ENCASTRE
            LOADNODE, 1, 2, 0.0,
            *Cload, amplitude=STEP
            LOADNODE, 2, -25.0
            *Dload, op=NEW
            EALL, GRAV, 9.81, 0.0, -1.0, 0.0,
            *Surface, type=ELEMENT, name=TOP
            EALL, S3
            *Coupling, constraint name=TIE-1, ref node=999, surface=TOP
            *Kinematic
            1, 6
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

        self.assertEqual(len(model.steps), 1)
        step = model.steps[0]

        self.assertEqual(step.name, "LOAD-STEP")
        self.assertEqual(step.static_parameters, (0.1, 1.0, 1e-05, 0.1))

        self.assertEqual(len(step.boundary_specs), 2)
        self.assertEqual(step.boundary_specs[0].target, "FIXED")
        self.assertEqual(step.boundary_specs[0].boundary_type, "ENCASTRE")
        self.assertEqual(step.boundary_specs[0].parameters, {"op": "NEW", "amplitude": "RAMP"})
        self.assertIsNone(step.boundary_specs[0].first_dof)
        self.assertIsNone(step.boundary_specs[0].last_dof)
        self.assertIsNone(step.boundary_specs[0].value)
        self.assertEqual(step.boundary_specs[1].target, "LOADNODE")
        self.assertIsNone(step.boundary_specs[1].boundary_type)
        self.assertEqual(step.boundary_specs[1].parameters, {"op": "NEW", "amplitude": "RAMP"})
        self.assertEqual(step.boundary_specs[1].first_dof, 1)
        self.assertEqual(step.boundary_specs[1].last_dof, 2)
        self.assertEqual(step.boundary_specs[1].value, 0.0)

        self.assertEqual(len(step.cload_specs), 1)
        self.assertEqual(step.cload_specs[0].target, "LOADNODE")
        self.assertEqual(step.cload_specs[0].parameters, {"amplitude": "STEP"})
        self.assertEqual(step.cload_specs[0].dof, 2)
        self.assertEqual(step.cload_specs[0].magnitude, -25.0)

        self.assertEqual(len(step.dload_specs), 1)
        self.assertEqual(step.dload_specs[0].target, "EALL")
        self.assertEqual(step.dload_specs[0].parameters, {"op": "NEW"})
        self.assertEqual(step.dload_specs[0].load_type, "GRAV")
        self.assertEqual(step.dload_specs[0].magnitude, 9.81)
        self.assertEqual(step.dload_specs[0].components, (0.0, -1.0, 0.0))

        self.assertEqual([spec.keyword for spec in step.unhandled_specs], ["SURFACE", "COUPLING", "KINEMATIC"])
        self.assertEqual(step.unhandled_specs[0].parameters["type"], "ELEMENT")
        self.assertEqual(step.unhandled_specs[0].parameters["name"], "TOP")
        self.assertEqual(step.unhandled_specs[0].data_lines, [("EALL", "S3")])
        self.assertEqual(step.unhandled_specs[1].parameters["constraint name"], "TIE-1")
        self.assertEqual(step.unhandled_specs[1].parameters["ref node"], "999")
        self.assertEqual(step.unhandled_specs[1].parameters["surface"], "TOP")
        self.assertEqual(step.unhandled_specs[1].data_lines, [])
        self.assertEqual(step.unhandled_specs[2].data_lines, [("1", "6")])

    def test_does_not_model_unsupported_procedure_as_static_step(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Step, name=DYN-STEP
            *Dynamic
            0.01, 1.0
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

        self.assertEqual(model.steps, [])

    def test_rejects_static_outside_step(self) -> None:
        inp_text = textwrap.dedent(
            """\
            *Static
            0.1, 1.0
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
            with self.assertRaisesRegex(ValueError, r"\*Static outside \*Step"):
                read_abaqus_inp_model(inp_path)
        finally:
            os.remove(inp_path)

    def test_rejects_step_load_cards_before_supported_procedure(self) -> None:
        cases = [
            (
                "boundary",
                textwrap.dedent(
                    """\
                    *Step, name=BAD-ORDER
                    *Boundary, op=NEW
                    NSET-1, 1, 1, 0.0
                    *End Step
                    """
                ),
                r"\*Boundary encountered before supported step procedure",
            ),
            (
                "cload",
                textwrap.dedent(
                    """\
                    *Step, name=BAD-ORDER
                    *Cload, amplitude=STEP
                    NSET-1, 2, -5.0
                    *End Step
                    """
                ),
                r"\*Cload encountered before supported step procedure",
            ),
            (
                "dload",
                textwrap.dedent(
                    """\
                    *Step, name=BAD-ORDER
                    *Dload, op=NEW
                    EALL, GRAV, 9.81, 0.0, -1.0, 0.0
                    *End Step
                    """
                ),
                r"\*Dload encountered before supported step procedure",
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


if __name__ == "__main__":
    unittest.main()

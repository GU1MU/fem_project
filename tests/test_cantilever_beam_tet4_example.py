import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from fem.mesh_io import read_tet4_3d_abaqus


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "examples" / "cantilever_beam_tet4.py"
RESULT_PATHS = [
    REPO_ROOT / "results" / "cantilever_beam_tet4_nodal_displacements.csv",
    REPO_ROOT / "results" / "cantilever_beam_tet4_element_stress.csv",
    REPO_ROOT / "results" / "cantilever_beam_tet4_nodal_stress.csv",
    REPO_ROOT / "results" / "cantilever_beam_tet4.vtk",
]


def _write_material_csv(path: Path) -> None:
    path.write_text("material_id,E,nu,rho\n1,210000000000.0,0.3,7850\n", encoding="utf-8")


class _WorkspaceTempDir:
    def __enter__(self) -> Path:
        root = REPO_ROOT / "_tmp_test_artifacts"
        root.mkdir(exist_ok=True)
        self.path = root / f"case_{uuid4().hex}"
        self.path.mkdir()
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class CantileverBeamTet4ExampleTests(unittest.TestCase):
    def test_read_tet4_abaqus_ignores_assembly_reference_nodes(self) -> None:
        with _WorkspaceTempDir() as tmp_path:
            inp_path = tmp_path / "tet4_with_reference_point.inp"
            mat_path = tmp_path / "materials.csv"
            _write_material_csv(mat_path)
            inp_path.write_text(
                "*Part, name=Beam\n"
                "*Node\n"
                "1,0,0,0\n"
                "2,1,0,0\n"
                "3,0,1,0\n"
                "4,0,0,1\n"
                "*Element, type=C3D4\n"
                "1,1,2,3,4\n"
                "*End Part\n"
                "*Assembly, name=Assembly\n"
                "*Instance, name=BeamInstance, part=Beam\n"
                "*End Instance\n"
                "*Node\n"
                "1,10,10,10\n"
                "*End Assembly\n",
                encoding="utf-8",
            )

            mesh = read_tet4_3d_abaqus(
                str(inp_path),
                material_id=1,
                material_path=str(mat_path),
            )

            self.assertEqual(len(mesh.nodes), 4)
            self.assertEqual(len({node.id for node in mesh.nodes}), 4)
            node1 = next(node for node in mesh.nodes if node.id == 1)
            self.assertEqual((node1.x, node1.y, node1.z), (0.0, 0.0, 0.0))

    def test_example_runs_and_exports_results(self) -> None:
        for path in RESULT_PATHS:
            if path.exists():
                path.unlink()

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"

        completed = subprocess.run(
            [sys.executable, str(EXAMPLE_PATH)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        )

        for path in RESULT_PATHS:
            self.assertTrue(path.exists(), msg=f"missing result file: {path}")
            self.assertGreater(path.stat().st_size, 0, msg=f"empty result file: {path}")


if __name__ == "__main__":
    unittest.main()

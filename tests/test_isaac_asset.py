"""Static contract checks for the Windows Isaac Sim asset bridge."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "isaac" / "m20pro_asset.py"
SPEC = importlib.util.spec_from_file_location("m20pro_asset", SCRIPT)
ASSET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSET)


class IsaacAssetTest(unittest.TestCase):
    def test_canonical_urdf_contract(self):
        report = ASSET.inspect_urdf(ROOT / "src" / "m20pro_description")
        self.assertEqual(len(report["joint_names"]), 16)
        self.assertEqual(len(report["wheel_joint_names"]), 4)
        self.assertEqual(report["link_count"], 17)
        self.assertEqual(report["mesh_count"], 17)

    def test_missing_mesh_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            shutil.copytree(ROOT / "src" / "m20pro_description" / "urdf", package / "urdf")
            (package / "meshes").mkdir()
            with self.assertRaisesRegex(ValueError, "Missing or unsafe mesh"):
                ASSET.inspect_urdf(package)


if __name__ == "__main__":
    unittest.main()

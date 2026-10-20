from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m20pro_vla.deploy import create_policy_export, policy_export_plan


class PolicyExportTest(unittest.TestCase):
    def test_export_manifest_without_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "m20_test_policy"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "best.pt"
            checkpoint.write_bytes(b"fake checkpoint bytes")
            (checkpoint_dir / "config.json").write_text("{\"epochs\": 1}", encoding="utf-8")
            contract = root / "contract.yaml"
            contract.write_text("schema: test_contract\n", encoding="utf-8")
            output = root / "export"

            result = create_policy_export(
                checkpoint=checkpoint,
                output_dir=output,
                real_contract=contract,
                image_width=80,
                image_height=48,
            )

            self.assertTrue(Path(result["manifest"]).is_file())
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "m20pro_vla_policy_export_v1")
            self.assertEqual(manifest["policy_id"], "m20_test_policy")
            self.assertFalse(manifest["checkpoint"]["copied"])
            self.assertEqual(manifest["observation_contract"]["front_rgb"]["width"], 80)
            self.assertFalse(manifest["real_runtime"]["per_map_lora_required"])

    def test_dry_run_plan_allows_missing_checkpoint(self):
        plan = policy_export_plan(checkpoint=None, target_runtime="shadow")
        self.assertEqual(plan["schema"], "m20pro_vla_policy_export_plan_v1")
        self.assertFalse(plan["checkpoint_present"])
        self.assertEqual(plan["target_runtime"], "shadow")


if __name__ == "__main__":
    unittest.main()

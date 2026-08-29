from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m20pro_vla.cli import build_parser, main
from m20pro_vla.runtime import RunContext


class CliRuntimeTest(unittest.TestCase):
    def test_parser_registers_system_commands(self):
        parser = build_parser()
        args = parser.parse_args(["smoke", "--dry-run", "--width", "64"])
        self.assertEqual(args.command, "smoke")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.width, 64)
        forwarded, unknown = parser.parse_known_args(["collect", "--dry-run", "--mode", "randomized", "--episodes", "1"])
        self.assertEqual(forwarded.command, "collect")
        self.assertTrue(forwarded.dry_run)
        self.assertIn("--mode", unknown)

    def test_dry_run_does_not_create_runtime_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = main(["smoke", "--dry-run", "--runtime-root", tmp])
            self.assertEqual(result, 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_run_context_records_command_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext.create(
                "test",
                ["m20pro-vla", "smoke"],
                runtime_root=Path(tmp),
                run_id="M20-test-run",
                config={"seed": 7},
            )
            context.write_summary({"passed": True})
            context.finish(True)
            self.assertTrue((context.path / "command.txt").is_file())
            self.assertEqual(json.loads((context.path / "config.json").read_text())["config"]["seed"], 7)
            self.assertEqual(json.loads((context.path / "summary.json").read_text())["run_id"], "M20-test-run")
            self.assertEqual(json.loads((context.path / "status.json").read_text())["status"], "success")


if __name__ == "__main__":
    unittest.main()

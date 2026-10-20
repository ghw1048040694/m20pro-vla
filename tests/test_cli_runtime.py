from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from m20pro_vla.cli import build_parser, command_experiment, main
from m20pro_vla.runtime import RunContext


class CliRuntimeTest(unittest.TestCase):
    def test_parser_registers_system_commands(self):
        parser = build_parser()
        args = parser.parse_args(["smoke", "--dry-run", "--width", "64"])
        self.assertEqual(args.command, "smoke")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.width, 64)
        export_args = parser.parse_args(["export-policy", "--dry-run", "--target-runtime", "shadow"])
        self.assertEqual(export_args.command, "export-policy")
        self.assertTrue(export_args.dry_run)
        self.assertEqual(export_args.target_runtime, "shadow")
        experiment_args = parser.parse_args(
            ["experiment", "--config", "configs/run.json", "--stage", "train", "--dry-run"]
        )
        self.assertEqual(experiment_args.command, "experiment")
        self.assertEqual(experiment_args.stage, "train")

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

    def test_lerobot_owned_stages_dispatch_before_stage_imports(self):
        for stage in ("convert", "train-smolvla"):
            with self.subTest(stage=stage), mock.patch(
                "m20pro_vla.runtime.experiment.build_experiment_plan",
                return_value={"experiment_id": "test"},
            ), mock.patch(
                "m20pro_vla.cli._dispatch_lerobot_stage_environment",
                return_value=23,
            ) as dispatch:
                args = argparse.Namespace(
                    config=Path("configs/experiment.json"),
                    stage=stage,
                    dry_run=False,
                    json=False,
                )
                self.assertEqual(command_experiment(args), 23)
                dispatch.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()

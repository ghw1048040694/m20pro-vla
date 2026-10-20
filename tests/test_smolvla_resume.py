from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m20pro_vla.training.smolvla import smolvla_training_command, validate_smolvla_resume


class SmolVLAResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.output = root / "training"
        self.checkpoint = self.output / "checkpoints/002500"
        self.settings = dict(steps=10000, batch_size=12, seed=1, num_workers=2, log_freq=10, save_freq=2500,
                             repo_id="m20pro/test", resume_checkpoint=str(self.checkpoint))
        self.config = dict(paths=dict(smolvla_checkpoint_dir=str(self.output), lerobot_dataset=str(root / "data")),
                           smolvla=self.settings)
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            path = self.checkpoint / "pretrained_model" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        for name in ("optimizer_param_groups.json", "optimizer_state.safetensors", "scheduler_state.json", "rng_state.safetensors"):
            path = self.checkpoint / "training_state" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (self.checkpoint / "training_state/training_step.json").write_text('{"step":2500}')
        self.saved = {key: self.settings[key] for key in ("steps", "batch_size", "seed", "num_workers", "log_freq", "save_freq")}
        self.saved.update(output_dir=str(self.output), dataset=dict(root=self.config["paths"]["lerobot_dataset"], repo_id="m20pro/test"))
        self.save()

    def save(self) -> None:
        (self.checkpoint / "pretrained_model/train_config.json").write_text(json.dumps(self.saved))

    def test_restores_state_without_policy_path_override(self) -> None:
        self.assertEqual(validate_smolvla_resume(self.config)[1], 2500)
        command = smolvla_training_command(self.config)
        self.assertIn("--resume=true", command)
        self.assertTrue(any(arg.startswith("--config_path=") for arg in command))
        self.assertFalse(any(arg.startswith("--policy.path=") for arg in command))

    def test_missing_optimizer_cannot_silently_restart(self) -> None:
        (self.checkpoint / "training_state/optimizer_state.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_smolvla_resume(self.config)

    def test_changed_budget_or_dataset_is_rejected(self) -> None:
        for key, value in (("steps", 7500), ("batch_size", 8)):
            with self.subTest(key=key):
                with patch.dict(self.settings, {key: value}):
                    with self.assertRaises(ValueError):
                        validate_smolvla_resume(self.config)
        self.saved["dataset"]["root"] += "-other"
        self.save()
        with self.assertRaises(ValueError):
            validate_smolvla_resume(self.config)

    def test_foreign_or_finished_checkpoint_is_rejected(self) -> None:
        with patch.dict(self.settings, {"resume_checkpoint": str(Path(self.temp.name) / "foreign")}):
            with self.assertRaises(ValueError):
                validate_smolvla_resume(self.config)
        (self.checkpoint / "training_state/training_step.json").write_text('{"step":10000}')
        with self.assertRaises(ValueError):
            validate_smolvla_resume(self.config)


if __name__ == "__main__":
    unittest.main()

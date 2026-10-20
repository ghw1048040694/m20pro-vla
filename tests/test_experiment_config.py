from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from m20pro_vla.runtime.experiment import build_experiment_plan, load_experiment_config, experiment_low_level_environment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment.json"


class ExperimentConfigTest(unittest.TestCase):
    def test_one_config_defines_every_stage(self) -> None:
        config = load_experiment_config(CONFIG)
        self.assertEqual(config["schema"], "m20pro_vla_experiment_v1")
        self.assertIn("collection", config)
        self.assertIn("training", config)
        self.assertIn("evaluation", config)
        self.assertIn("smolvla", config)
        plan = build_experiment_plan(CONFIG)
        self.assertEqual(
            set(plan["stages"]),
            {
                "collect",
                "curate",
                "audit",
                "convert",
                "train",
                "train-smolvla",
                "smolvla-eval",
                "collect-recovery",
                "eval",
                "gate",
            },
        )
        self.assertIn("--dataset", plan["stages"]["train"]["args"])
        self.assertIn("--checkpoint", plan["stages"]["eval"]["args"])
        self.assertIn("--episodes-dir", plan["stages"]["smolvla-eval"]["args"])
        self.assertIn("--summary", plan["stages"]["smolvla-eval"]["args"])
        self.assertIn(config["paths"]["raw_dataset"], plan["stages"]["collect"]["args"])
        self.assertEqual(plan["stages"]["curate"]["output"], config["paths"]["dataset"])
        self.assertIn("--recovery-output-dir", plan["stages"]["collect-recovery"]["args"])
        self.assertIn("--episode-ids", plan["stages"]["collect-recovery"]["args"])
        # The closed-loop stage must always target the final checkpoint of the
        # configured training run, so a step-budget change cannot silently leave
        # the promotion evidence pointing at an older checkpoint.
        self.assertTrue(
            plan["stages"]["smolvla-eval"]["checkpoint"].endswith(
                f"checkpoints/{config['smolvla']['steps']:06d}/pretrained_model"
            )
        )
        self.assertEqual(config["smolvla"]["batch_size"], 12)

    def test_smolvla_eval_panel_is_unique_and_matches_collected_data_when_present(self) -> None:
        config = load_experiment_config(CONFIG)
        episode_ids = [item.strip() for item in config["smolvla_evaluation"]["episode_ids"].split(",")]
        self.assertGreaterEqual(len(episode_ids), config["closed_loop_acceptance"]["min_episodes"])
        self.assertEqual(len(episode_ids), len(set(episode_ids)))
        self.assertTrue(all(item.isdigit() for item in episode_ids))

        dataset = ROOT / config["smolvla_evaluation"].get("episodes_dir", config["paths"]["dataset"])
        if dataset.is_dir():
            missing = [item for item in episode_ids if not (dataset / f"episode_{item}.json").is_file()]
            self.assertEqual(missing, [])

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"schema":"unknown"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_low_level_config_pins_an_existing_policy_and_rejects_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.onnx"
            policy.write_bytes(b"test-path-only")
            environment = experiment_low_level_environment({"low_level": {"backend": "v5", "policy_onnx": str(policy)}})
            self.assertEqual(environment["M20_V5_POLICY_ONNX"], str(policy.resolve()))
            policy.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment_low_level_environment({"low_level": {"policy_onnx": str(policy)}})
        self.assertEqual(experiment_low_level_environment({}), {})


if __name__ == "__main__":
    unittest.main()

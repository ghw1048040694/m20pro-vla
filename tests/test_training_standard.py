from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m20pro_vla.training import evaluate_training_candidate


class TrainingStandardTest(unittest.TestCase):
    def test_candidate_requires_offline_and_closed_loop_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            (root / "history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 4,
                            "train_loss": 0.1,
                            "val_loss": 0.2,
                            "val_stop_accuracy": 0.98,
                            "train_phase_accuracy": 0.96,
                            "val_phase_accuracy": 0.94,
                            "train_visual_visible_accuracy": 0.95,
                            "val_visual_visible_accuracy": 0.93,
                            "val_visual_offset_mae_visible": 0.05,
                            "val_visual_area_mae_visible": 0.04,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                json.dumps({"split_unit": "layout_id", "val_episodes": [str(i) for i in range(6)]}),
                encoding="utf-8",
            )
            (root / "distribution_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            evaluation = root / "evaluation.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "episode_count": 10,
                        "success_rate": 0.7,
                        "discovery_rate": 0.9,
                        "strict_pass_rate": 0.6,
                        "min_obstacle_clearance_min_m": 0.2,
                        "episodes": [{"false_stop_before_discovery": False} for _ in range(10)],
                    }
                ),
                encoding="utf-8",
            )
            standard = Path(__file__).resolve().parents[1] / "configs/experiment.json"
            report = evaluate_training_candidate(
                checkpoint=checkpoint,
                eval_summary=evaluation,
                standard_path=standard,
            )
            self.assertTrue(report["eligible_for_shadow"])
            self.assertTrue(report["gates"]["offline"])
            self.assertTrue(report["gates"]["closed_loop"])

            payload = json.loads(evaluation.read_text(encoding="utf-8"))
            payload["strict_pass_rate"] = 0.5
            evaluation.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_training_candidate(
                checkpoint=checkpoint,
                eval_summary=evaluation,
                standard_path=standard,
            )
            self.assertFalse(report["eligible_for_shadow"])
            self.assertFalse(report["closed_loop_gates"]["min_strict_pass_rate"])


if __name__ == "__main__":
    unittest.main()

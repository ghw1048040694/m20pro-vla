from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m20pro_vla.data import M20MuJoCoDistributionThresholds, audit_m20_mujoco_dataset


def write_episode(root: Path, episode_id: int, metadata: dict) -> None:
    (root / f"episode_{episode_id:04d}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / f"episode_{episode_id:04d}.npz").touch()


class DistributionAuditTest(unittest.TestCase):
    def test_balanced_search_dataset_passes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_episode(
                root,
                0,
                {
                    "collection_mode": "search",
                    "layout_id": 0,
                    "terrain_profile": "flat",
                    "task_language": "en",
                    "search_start_mode": "balanced",
                    "search_start_variant": "anchor",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": True,
                    "min_obstacle_clearance": 0.22,
                    "target_label": "red cube",
                },
            )
            write_episode(
                root,
                1,
                {
                    "collection_mode": "search",
                    "layout_id": 0,
                    "terrain_profile": "flat",
                    "task_language": "zh",
                    "search_start_mode": "balanced",
                    "search_start_variant": "varied",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": False,
                    "min_obstacle_clearance": 0.18,
                    "target_label": "green cylinder",
                },
            )
            write_episode(
                root,
                2,
                {
                    "collection_mode": "search",
                    "layout_id": 1,
                    "terrain_profile": "bumps",
                    "task_language": "en",
                    "search_start_mode": "balanced",
                    "search_start_variant": "anchor",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": True,
                    "min_obstacle_clearance": 0.24,
                    "target_label": "yellow box",
                },
            )
            write_episode(
                root,
                3,
                {
                    "collection_mode": "search",
                    "layout_id": 1,
                    "terrain_profile": "bumps",
                    "task_language": "zh",
                    "search_start_mode": "balanced",
                    "search_start_variant": "varied",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": False,
                    "min_obstacle_clearance": 0.15,
                    "target_label": "red cube",
                },
            )
            report = audit_m20_mujoco_dataset(
                root,
                thresholds=M20MuJoCoDistributionThresholds(
                    min_episodes=4,
                    min_layouts=2,
                    min_terrain_profiles=2,
                    min_languages=2,
                    min_search_anchor_episodes=1,
                    min_search_varied_episodes=1,
                    min_search_successes=1,
                    min_target_discovered=1,
                    min_search_clearance_m=0.10,
                ),
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["layout_count"], 2)
            self.assertEqual(report["search_anchor_count"], 2)
            self.assertEqual(report["search_varied_count"], 2)
            self.assertEqual(report["search_success_count"], 2)
            self.assertEqual(report["search_discovery_count"], 4)
            self.assertEqual(report["task_languages"], {"en": 2, "zh": 2})
            self.assertTrue(report["gates"]["search_balanced_anchor_and_varied"])

    def test_narrow_search_dataset_fails_balance_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_episode(
                root,
                0,
                {
                    "collection_mode": "search",
                    "layout_id": 0,
                    "terrain_profile": "flat",
                    "task_language": "en",
                    "search_start_mode": "anchor",
                    "search_start_variant": "anchor",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": True,
                    "min_obstacle_clearance": 0.22,
                    "target_label": "red cube",
                },
            )
            write_episode(
                root,
                1,
                {
                    "collection_mode": "search",
                    "layout_id": 0,
                    "terrain_profile": "flat",
                    "task_language": "en",
                    "search_start_mode": "anchor",
                    "search_start_variant": "anchor",
                    "initial_target_visible": False,
                    "target_discovered": True,
                    "target_discovered_before_reach": True,
                    "success": True,
                    "min_obstacle_clearance": 0.22,
                    "target_label": "green cylinder",
                },
            )
            report = audit_m20_mujoco_dataset(
                root,
                thresholds=M20MuJoCoDistributionThresholds(
                    min_episodes=2,
                    min_layouts=2,
                    min_terrain_profiles=2,
                    min_languages=2,
                    min_search_anchor_episodes=1,
                    min_search_varied_episodes=1,
                    min_search_successes=1,
                    min_target_discovered=1,
                    min_search_clearance_m=0.10,
                ),
            )
            self.assertFalse(report["passed"])
            self.assertFalse(report["gates"]["min_layout_count"])
            self.assertFalse(report["gates"]["min_terrain_profile_count"])
            self.assertFalse(report["gates"]["min_language_count"])
            self.assertFalse(report["gates"]["search_balanced_anchor_and_varied"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path


class TrainSplitTest(unittest.TestCase):
    def test_episode_training_weight_emphasizes_rough_terrain(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/train_m20_mujoco_vla.py"))
        episode_training_weight = ns["episode_training_weight"]
        flat = episode_training_weight(
            {
                "success": False,
                "collection_mode": "search",
                "target_discovered": True,
                "min_target_distance": 1.5,
                "terrain_profile": "flat",
            }
        )
        rough = episode_training_weight(
            {
                "success": False,
                "collection_mode": "search",
                "target_discovered": True,
                "min_target_distance": 1.5,
                "terrain_profile": "bumps",
            }
        )
        self.assertGreater(rough, flat)
        self.assertLessEqual(rough, 1.35)

    def test_episode_training_weight_emphasizes_hard_search_cases(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/train_m20_mujoco_vla.py"))
        episode_training_weight = ns["episode_training_weight"]
        easy = episode_training_weight(
            {
                "success": True,
                "collection_mode": "search",
                "target_discovered": True,
                "target_discovered_before_reach": False,
                "initial_target_visible": True,
                "min_target_distance": 1.6,
                "min_obstacle_clearance": 0.30,
                "search_start_yaw_offset_rad": 0.0,
                "terrain_profile": "flat",
            }
        )
        hard = episode_training_weight(
            {
                "success": True,
                "collection_mode": "search",
                "target_discovered": True,
                "target_discovered_before_reach": True,
                "initial_target_visible": False,
                "min_target_distance": 1.6,
                "min_obstacle_clearance": 0.04,
                "search_start_yaw_offset_rad": 0.60,
                "terrain_profile": "bumps",
            }
        )
        self.assertGreater(hard, easy)
        self.assertLessEqual(hard, 1.35)

    def test_episode_training_weight_includes_partial_search(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/train_m20_mujoco_vla.py"))
        episode_training_weight = ns["episode_training_weight"]
        self.assertEqual(
            episode_training_weight(
                {
                    "success": True,
                    "collection_mode": "search",
                    "target_discovered": True,
                    "min_target_distance": 0.5,
                }
            ),
            1.0,
        )
        partial = episode_training_weight(
            {
                "success": False,
                "collection_mode": "search",
                "target_discovered": True,
                "min_target_distance": 1.8,
            }
        )
        undiscovered = episode_training_weight(
            {
                "success": False,
                "collection_mode": "search",
                "target_discovered": False,
                "min_target_distance": 1.8,
            }
        )
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)
        self.assertLess(undiscovered, partial)

    def test_split_paths_by_layout_uses_layout_groups_when_available(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/train_m20_mujoco_vla.py"))
        split_paths_by_layout = ns["split_paths_by_layout"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for layout_id in range(4):
                for episode_id in range(3):
                    path = root / f"episode_{layout_id}_{episode_id}.npz"
                    path.touch()
                    path.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "layout_id": layout_id,
                                "success": True,
                                "task_text": "go to the red cube",
                                "target_label": "red cube",
                            }
                        ),
                        encoding="utf-8",
                    )
                    paths.append(path)
            train_paths, val_paths, split_unit, val_layouts = split_paths_by_layout(
                paths,
                val_fraction=0.25,
                seed=1234,
            )
            self.assertEqual(split_unit, "layout_id")
            self.assertEqual(len(val_layouts), 1)
            train_layouts = {
                json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["layout_id"]
                for path in train_paths
            }
            val_layouts_seen = {
                json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["layout_id"]
                for path in val_paths
            }
            self.assertTrue(train_layouts.isdisjoint(val_layouts_seen))
            self.assertEqual(len(train_paths) + len(val_paths), len(paths))


if __name__ == "__main__":
    unittest.main()

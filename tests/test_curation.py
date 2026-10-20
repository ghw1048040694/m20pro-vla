import json
import tempfile
import unittest
from pathlib import Path

from m20pro_vla.data import M20MuJoCoDistributionThresholds, curate_episode_view


def episode(root: Path, number: int, **changes) -> None:
    payload = {
        "episode_id": number,
        "collection_mode": "search",
        "quality_passed": True,
        "layout_id": 1,
        "terrain_profile": "flat",
        "task_language": "en",
        "search_start_mode": "balanced",
        "search_start_variant": "anchor",
        "target_label": "red cube",
        "target_discovered": True,
        "success": True,
        "initial_xy": [0.0, 0.0],
        "final_xy": [1.0, 0.0],
        "min_obstacle_clearance": 0.3,
        "min_base_height": 0.49,
        "max_abs_roll_deg": 1.0,
        "max_abs_pitch_deg": 1.0,
    }
    payload.update(changes)
    (root / f"episode_{number}.json").write_text(json.dumps(payload))
    (root / f"episode_{number}.npz").touch()


class CuratedViewTest(unittest.TestCase):
    def test_filters_stationary_source_and_links_valid_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, recovery, output = root / "source", root / "recovery", root / "curated"
            source.mkdir()
            recovery.mkdir()
            episode(source, 1)
            episode(source, 2, final_xy=[0.001, 0.0], success=False)
            episode(
                recovery, 3000,
                collection_mode="failure_recovery",
                search_start_variant="varied",
                final_xy=[0.1, 0.0],
            )
            report = curate_episode_view(
                source=source, recovery=recovery, output=output,
                thresholds=M20MuJoCoDistributionThresholds(
                    min_episodes=2, min_layouts=1, min_terrain_profiles=1,
                    min_languages=1, min_search_anchor_episodes=1,
                    min_search_varied_episodes=1, min_search_successes=1,
                    min_target_discovered=1,
                ),
            )
            self.assertEqual(report["selected_episode_count"], 2)
            self.assertEqual(report["excluded_episode_ids"], [2])
            self.assertTrue((output / "episode_1.npz").is_symlink())
            self.assertTrue((output / "episode_3000.npz").is_symlink())
            self.assertFalse((output / "episode_2.npz").exists())
            self.assertTrue(report["audit"]["passed"])


if __name__ == "__main__":
    unittest.main()

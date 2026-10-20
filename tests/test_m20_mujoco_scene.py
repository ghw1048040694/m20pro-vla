from __future__ import annotations

import tempfile
import unittest
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.low_level import (
    M20LowLevelController,
    M20V5PolicyController,
    resolve_onnx_path,
)
from m20pro_vla.planning import SearchMPCConfig, SearchMPCPlanner
from m20pro_vla.planning.search_mpc import _route_command
from m20pro_vla.sim.mujoco import (
    ASSET,
    ObjectSpec,
    ObstacleSpec,
    SceneLightSpec,
    build_scene,
    obstacle_blocks_segment,
    segment_intersects_aabb,
    TerrainSpec,
)


class MuJoCoSceneTest(unittest.TestCase):
    def test_search_final_approach_uses_slow_arc_instead_of_skid_turn(self) -> None:
        bearing = 0.75
        for distance, expected_forward in ((0.67, 0.08), (0.70, 0.08), (1.00, 0.0)):
            with self.subTest(distance=distance):
                goal = np.array((distance * np.cos(bearing), distance * np.sin(bearing)))
                command, _, measured_bearing = _route_command(
                    np.zeros(2),
                    0.0,
                    goal,
                    distance,
                    final_approach=True,
                )
                self.assertAlmostEqual(measured_bearing, bearing)
                self.assertEqual(command.forward, expected_forward)
                self.assertEqual(command.yaw, 0.15)
                self.assertFalse(command.stop)

    def test_obstacle_blocks_forward_line_of_sight(self) -> None:
        obstacle = ObstacleSpec(
            name="search_wall",
            kind="box",
            position=(0.84, 0.0, 0.55),
            size=(0.08, 0.62, 0.55),
        )
        self.assertTrue(obstacle_blocks_segment((0.0, 0.0), (1.8, 0.0), obstacle, padding=0.04))
        self.assertFalse(obstacle_blocks_segment((0.0, 0.92), (1.8, 0.92), obstacle, padding=0.04))

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_build_scene_emits_obstacle_body(self) -> None:
        obstacle = ObstacleSpec(
            name="search_wall",
            kind="box",
            position=(0.84, 0.0, 0.55),
            size=(0.08, 0.62, 0.55),
        )
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, [], obstacles=[obstacle], task_object_collisions=False)
            root = ET.parse(scene_path).getroot()
            self.assertIsNotNone(root.find("./worldbody/body[@name='search_wall']"))
            self.assertIsNotNone(root.find("./worldbody/body[@name='search_wall']/geom[@name='search_wall_geom']"))

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_build_scene_emits_custom_light(self) -> None:
        light = SceneLightSpec(
            pos=(0.3, -0.2, 5.7),
            direction=(0.1, 0.0, -0.99),
            diffuse=(0.7, 0.8, 0.9),
            ambient=(0.03, 0.04, 0.05),
            specular=(0.01, 0.02, 0.03),
        )
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, [], obstacles=[], task_object_collisions=False, light=light)
            root = ET.parse(scene_path).getroot()
            light_node = root.find("./worldbody/light[@name='overhead']")
            self.assertIsNotNone(light_node)
            assert light_node is not None
            self.assertEqual(light_node.attrib["pos"], "0.3000 -0.2000 5.7000")
            self.assertEqual(light_node.attrib["diffuse"], "0.7000 0.8000 0.9000")
            self.assertEqual(light_node.attrib["ambient"], "0.0300 0.0400 0.0500")
            self.assertEqual(light_node.attrib["specular"], "0.0100 0.0200 0.0300")

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_policy_cameras_are_upright(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, [], obstacles=[], task_object_collisions=False)
            root = ET.parse(scene_path).getroot()
            front = root.find(".//camera[@name='front_rgb']")
            rear = root.find(".//camera[@name='rear_rgb']")
            self.assertIsNotNone(front)
            self.assertIsNotNone(rear)
            assert front is not None and rear is not None
            self.assertEqual(front.attrib["xyaxes"], "0 -1 0 0.174 0 0.985")
            self.assertEqual(rear.attrib["xyaxes"], "0 1 0 -0.174 0 0.985")

    def test_search_layout_templates_are_occluded(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        templates = ns["SEARCH_LAYOUT_TEMPLATES"]
        jitter = ns["SEARCH_SLOT_JITTER_XY"]
        obstacles = ns["SEARCH_OBSTACLE_TEMPLATES"]
        self.assertEqual(len(templates), 4)
        self.assertEqual(len(obstacles), 4)
        rng = __import__("numpy").random.default_rng(20260731)
        for template in templates[:3]:
            for slot_xy in template:
                for obstacle in obstacles[:3]:
                    self.assertTrue(obstacle_blocks_segment((0.0, 0.0), slot_xy, obstacle, padding=0.04))
                    for _ in range(20):
                        jittered = (
                            float(slot_xy[0] + rng.uniform(-float(jitter[0]), float(jitter[0]))),
                            float(slot_xy[1] + rng.uniform(-float(jitter[1]), float(jitter[1]))),
                        )
                        self.assertTrue(obstacle_blocks_segment((0.0, 0.0), jittered, obstacle, padding=0.04))
        for slot_xy in templates[3]:
            self.assertTrue(obstacle_blocks_segment((0.0, 0.0), slot_xy, obstacles[3], padding=0.04))
            for _ in range(20):
                jittered = (
                    float(slot_xy[0] + rng.uniform(-float(jitter[0]), float(jitter[0]))),
                    float(slot_xy[1] + rng.uniform(-float(jitter[1]), float(jitter[1]))),
                )
                self.assertTrue(obstacle_blocks_segment((0.0, 0.0), jittered, obstacles[3], padding=0.04))

    def test_search_clutter_templates_keep_route_lane_open(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        templates = ns["SEARCH_LAYOUT_TEMPLATES"]
        primary_templates = ns["SEARCH_OBSTACLE_TEMPLATES"]
        center_clutter_templates = tuple(ns["SEARCH_CLUTTER_TEMPLATES"])
        outer_clutter_templates = tuple(ns["SEARCH_OUTER_CLUTTER_TEMPLATES"])
        clutter_templates = center_clutter_templates + outer_clutter_templates
        config = SearchMPCConfig()
        self.assertEqual(len(clutter_templates), 7)
        for template_id, layout in enumerate(templates):
            for slot_xy in layout:
                if template_id < 3:
                    primaries = primary_templates[:3]
                else:
                    primaries = primary_templates[3:]
                for primary in primaries:
                    route_side_sign = 1.0 if float(slot_xy[1] - primary.position[1]) >= 0.0 else -1.0
                    lane_y = float(primary.position[1]) + route_side_sign * (float(primary.size[1]) + config.lane_margin)
                    before = (
                        float(primary.position[0]) - config.pre_obstacle_margin_x,
                        lane_y,
                    )
                    after = (
                        float(primary.position[0]) + float(primary.size[0]) + config.post_obstacle_margin_x,
                        lane_y,
                    )
                    for clutter in clutter_templates:
                        self.assertFalse(obstacle_blocks_segment(before, after, clutter, padding=0.04))
                        self.assertGreater(
                            abs(lane_y - float(clutter.position[1])),
                            float(clutter.size[1]) + 0.04 - 1.0e-6,
                        )
                    for clutter in outer_clutter_templates:
                        self.assertGreater(
                            abs(lane_y - float(clutter.position[1])) - float(clutter.size[1]),
                            0.60,
                        )

    def test_search_layout_generator_emits_three_obstacle_occlusions(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        rng = __import__("numpy").random.default_rng(20260731)
        objects, targets, obstacles, template_id, obstacle_id, clutter_id, outer_id = ns["make_search_layout"](rng)
        self.assertEqual(len(objects), 3)
        self.assertEqual(len(targets), 3)
        self.assertEqual(len(obstacles), 3)
        self.assertGreaterEqual(template_id, 0)
        self.assertGreaterEqual(obstacle_id, 0)
        self.assertGreaterEqual(clutter_id, 0)
        self.assertGreaterEqual(outer_id, 0)
        for target_xy in targets.values():
            self.assertFalse(ns["_target_visible_from"]((0.0, 0.0), target_xy, obstacles))
        self.assertIsNotNone(ns["_minimum_obstacle_clearance"]((0.0, 0.0), obstacles))

    def test_search_layout_appearance_seed_changes_rgba_without_geometry(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        rng_a = __import__("numpy").random.default_rng(20260731)
        rng_b = __import__("numpy").random.default_rng(20260731)
        objects_a, targets_a, obstacles_a, *_ = ns["make_search_layout"](rng_a, seed=111)
        objects_b, targets_b, obstacles_b, *_ = ns["make_search_layout"](rng_b, seed=222)
        self.assertEqual(set(targets_a), set(targets_b))
        for label in targets_a:
            self.assertTrue(np.allclose(targets_a[label], targets_b[label]))
        for obstacle_a, obstacle_b in zip(obstacles_a, obstacles_b):
            self.assertTrue(np.allclose(np.asarray(obstacle_a.position), np.asarray(obstacle_b.position)))
            self.assertTrue(np.allclose(np.asarray(obstacle_a.size), np.asarray(obstacle_b.size)))
        self.assertTrue(
            any(not np.allclose(np.asarray(obj_a.rgba), np.asarray(obj_b.rgba)) for obj_a, obj_b in zip(objects_a, objects_b))
        )

    def test_rough_terrain_search_layouts_are_closer_but_still_occluded(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        flat_rng = __import__("numpy").random.default_rng(20260731)
        rough_rng = __import__("numpy").random.default_rng(20260731)
        _, flat_targets, flat_obstacles, *_ = ns["make_search_layout"](flat_rng, terrain_profile="flat")
        _, rough_targets, rough_obstacles, *_ = ns["make_search_layout"](rough_rng, terrain_profile="bumps")
        self.assertEqual(set(flat_targets), set(rough_targets))
        for label in flat_targets:
            self.assertLess(float(rough_targets[label][0]), float(flat_targets[label][0]))
            self.assertFalse(ns["_target_visible_from"]((0.0, 0.0), rough_targets[label], rough_obstacles))
            self.assertFalse(ns["_target_visible_from"]((0.0, 0.0), flat_targets[label], flat_obstacles))

    def test_search_terrain_profiles_cycle_by_layout(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        terrain_for_layout = ns["_search_terrain_profile_for_layout"]
        self.assertEqual(
            [terrain_for_layout(index) for index in range(8)],
            ["flat", "slope", "bumps", "step", "flat", "slope", "bumps", "step"],
        )

    def test_search_start_yaw_offsets_are_varied(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        helper = ns["_search_initial_yaw_for"]
        offsets = {
            round(
                float(
                    helper(
                        seed=20260731,
                        layout_id=layout_id,
                        target_label=label,
                        start_xy=np.array((0.0, 0.0), dtype=np.float64),
                        target_xy=np.array((3.5, 0.2 + 0.1 * layout_id), dtype=np.float64),
                    )[1]
                ),
                3,
            )
            for layout_id in range(4)
            for label in ("red cube", "green cylinder", "yellow box")
        }
        self.assertGreater(len(offsets), 1)
        self.assertTrue(any(abs(offset) > 0.0 for offset in offsets))

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_build_scene_accepts_rough_terrain_for_search(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        rng = __import__("numpy").random.default_rng(20260731)
        objects, _, obstacles, *_ = ns["make_search_layout"](rng)
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(
                scene_path,
                objects,
                obstacles=obstacles,
                task_object_collisions=False,
                terrain=TerrainSpec("slope"),
            )
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            self.assertGreater(model.ngeom, 0)

    def test_search_start_templates_keep_targets_occluded(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        helper = ns["_search_initial_xy_for"]
        start_positions = set()
        for layout_id in range(4):
            rng = __import__("numpy").random.default_rng(20260731 + layout_id)
            _, targets, obstacles, *_ = ns["make_search_layout"](rng)
            for target_label, target_xy in targets.items():
                start_xy, template_id = helper(
                    seed=20260731,
                    layout_id=layout_id,
                    target_label=target_label,
                )
                start_positions.add((round(float(start_xy[0]), 3), round(float(start_xy[1]), 3), int(template_id)))
                self.assertFalse(ns["_target_visible_from"](start_xy, target_xy, obstacles))
        self.assertGreater(len(start_positions), 1)

    def test_search_start_mode_balanced_keeps_anchor_and_varied_variants(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        rng = __import__("numpy").random.default_rng(20260731)
        _, targets, obstacles, *_ = ns["make_search_layout"](rng)
        target_xy = targets["red cube"]
        variants = ns["_search_initial_pose_variants_for"](
            mode="balanced",
            seed=20260731,
            layout_id=0,
            target_label="red cube",
            target_xy=target_xy,
        )
        self.assertEqual([variant[0] for variant in variants], ["anchor", "varied"])
        anchor = variants[0]
        self.assertTrue(np.allclose(anchor[1], np.array((0.0, 0.0), dtype=np.float64)))
        self.assertEqual(anchor[2], -1)
        self.assertEqual(anchor[3], 0.0)
        self.assertEqual(anchor[4], 0.0)
        for _, start_xy, _, _, _ in variants:
            self.assertFalse(ns["_target_visible_from"](start_xy, target_xy, obstacles))

    def test_mujoco_collector_has_bilingual_task_templates(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"))
        templates = ns["TASK_TEMPLATES"]
        languages = {item[1] for item in templates}
        self.assertEqual(len(templates), 24)
        self.assertEqual(languages, {"en", "zh"})
        fixed = ns["_task_language_for"](
            target_label="red cube",
            episode_id=0,
            seed=20260731,
            language_mode="fixed",
        )
        self.assertEqual(fixed["task_text"], "go to the red cube")
        varied_languages = {
            ns["_task_language_for"](
                target_label=label,
                episode_id=episode_id,
                seed=20260731,
                language_mode="varied",
            )["task_language"]
            for episode_id in range(24)
            for label in ("red cube", "green cylinder", "yellow box")
        }
        self.assertEqual(varied_languages, {"en", "zh"})

    def test_replay_object_specs_use_metadata_rgba(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/play_m20_mujoco_vla.py"))
        specs = ns["object_specs"](
            {
                "objects": [
                    {
                        "name": "red_cube",
                        "position": [1.0, 0.0],
                        "rgba": [0.12, 0.34, 0.56, 1.0],
                        "size": [0.17, 0.18, 0.19],
                    }
                ]
            }
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].rgba, (0.12, 0.34, 0.56, 1.0))
        self.assertEqual(specs[0].size, (0.17, 0.18, 0.19))

    def test_replay_visibility_and_clearance_helpers_are_obstacle_aware(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/play_m20_mujoco_vla.py"))
        obstacle = ObstacleSpec(
            name="screen",
            kind="box",
            position=(1.0, 0.0, 0.5),
            size=(0.10, 0.30, 0.5),
        )
        target_xy = np.array((2.0, 0.0), dtype=np.float64)
        self.assertFalse(ns["target_visible_from"](np.array((0.0, 0.0), dtype=np.float64), target_xy, [obstacle]))
        self.assertTrue(ns["target_visible_from"](np.array((0.0, 0.9), dtype=np.float64), target_xy, [obstacle]))
        self.assertAlmostEqual(
            ns["minimum_obstacle_clearance"](np.array((1.0, 0.55), dtype=np.float64), [obstacle]),
            0.25,
            places=6,
        )

    def test_hidden_search_gate_requires_discovery_clearance_and_success(self) -> None:
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/mujoco/evaluate_m20_mujoco_hidden_search.py"))
        base_report = {
            "search_required": True,
            "initial_target_visible": False,
            "target_discovered": True,
            "target_discovered_before_reach": True,
            "false_stop_before_discovery": False,
            "min_obstacle_clearance": 0.18,
            "success": True,
        }
        self.assertTrue(ns["hidden_search_passed"](base_report, 0.10))
        low_clearance = dict(base_report, min_obstacle_clearance=0.04)
        self.assertFalse(ns["hidden_search_passed"](low_clearance, 0.10))
        false_stop = dict(base_report, false_stop_before_discovery=True)
        self.assertFalse(ns["hidden_search_passed"](false_stop, 0.10))
        summary = ns["summarize_reports"]([base_report, low_clearance], 0.10)
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["strict_pass_count"], 1)
        self.assertFalse(summary["passed"])

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_low_level_reset_accepts_initial_xy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, [], obstacles=[], task_object_collisions=False)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0, base_xy=(0.07, -0.03))
            self.assertAlmostEqual(float(data.qpos[0]), 0.07, places=6)
            self.assertAlmostEqual(float(data.qpos[1]), -0.03, places=6)

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_active_search_policy_uses_mpc_scoring_for_obstacles(self) -> None:
        obstacle = ObstacleSpec(
            name="search_wall",
            kind="box",
            position=(2.022822837243513, 0.0, 0.5205213822947318),
            size=(0.07892867288141192, 0.24000000000000002, 0.5205213822947318),
        )
        target_xy = np.array((3.50, 0.46), dtype=np.float64)
        objects = [
            ObjectSpec(
                name="red_cube",
                label="red cube",
                kind="box",
                position=(float(target_xy[0]), float(target_xy[1])),
                rgba=(0.90, 0.08, 0.05, 1.0),
                size=(0.16, 0.16, 0.16),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, objects, obstacles=[obstacle], task_object_collisions=False)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0)
            planner = SearchMPCPlanner(
                model,
                target_xy,
                [obstacle],
                SearchMPCConfig(use_route_planner_when_obstacles=False),
            )
            command, info = planner.recommend(data, controller)
            self.assertEqual(command.shape, (4,))
            self.assertIn("candidate_scores", info)
            self.assertNotIn("route_phase", info)
            self.assertFalse(bool(info["goal_is_target"]))
            self.assertGreater(len(info["candidate_scores"]), 0)
            self.assertIn("visible_transitions", info)
            self.assertGreaterEqual(int(info["visible_transitions"]), 0)

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_search_route_planner_returns_target_once_visible(self) -> None:
        obstacle = ObstacleSpec(
            name="side_panel",
            kind="box",
            position=(0.65, 1.05, 0.50),
            size=(0.10, 0.24, 0.50),
        )
        target_xy = np.array((3.50, 0.00), dtype=np.float64)
        objects = [
            ObjectSpec(
                name="red_cube",
                label="red cube",
                kind="box",
                position=(float(target_xy[0]), float(target_xy[1])),
                rgba=(0.90, 0.08, 0.05, 1.0),
                size=(0.16, 0.16, 0.16),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(scene_path, objects, obstacles=[obstacle], task_object_collisions=False)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0)
            planner = SearchMPCPlanner(model, target_xy, [obstacle], SearchMPCConfig())
            command, info = planner.recommend(data, controller)
            self.assertEqual(command.shape, (4,))
            self.assertTrue(bool(info["goal_is_target"]))
            self.assertTrue(
                np.allclose(np.asarray(info["planning_goal"]), target_xy, atol=1.0e-6)
            )

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_search_route_planner_prefers_blocking_obstacle(self) -> None:
        decoy = ObstacleSpec(
            name="decoy_panel",
            kind="box",
            position=(0.60, 1.00, 0.50),
            size=(0.10, 0.24, 0.50),
        )
        blocker = ObstacleSpec(
            name="screen",
            kind="box",
            position=(1.50, 0.00, 0.50),
            size=(0.08, 0.24, 0.50),
        )
        target_xy = np.array((3.50, 0.00), dtype=np.float64)
        objects = [
            ObjectSpec(
                name="red_cube",
                label="red cube",
                kind="box",
                position=(float(target_xy[0]), float(target_xy[1])),
                rgba=(0.90, 0.08, 0.05, 1.0),
                size=(0.16, 0.16, 0.16),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.xml"
            build_scene(
                scene_path,
                objects,
                obstacles=[decoy, blocker],
                task_object_collisions=False,
            )
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0)
            planner = SearchMPCPlanner(model, target_xy, [decoy, blocker], SearchMPCConfig())
            command, info = planner.recommend(data, controller)
            self.assertEqual(command.shape, (4,))
            self.assertFalse(bool(info["goal_is_target"]))
            planning_goal = np.asarray(info["planning_goal"], dtype=np.float64)
            self.assertGreater(planning_goal[0], 1.0)

    def _planner_rollout_probe_scene(
        self,
        tmp: str,
        target_xy: np.ndarray,
        obstacles: list[ObstacleSpec],
    ) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Build the minimal occluded-target scene the rollout tests need."""
        objects = [
            ObjectSpec(
                name="red_cube",
                label="red cube",
                kind="box",
                position=(float(target_xy[0]), float(target_xy[1])),
                rgba=(0.90, 0.08, 0.05, 1.0),
                size=(0.16, 0.16, 0.16),
            )
        ]
        scene_path = Path(tmp) / "scene.xml"
        build_scene(
            scene_path,
            objects,
            obstacles=obstacles,
            task_object_collisions=False,
        )
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        return model, mujoco.MjData(model)

    def _rollout_probe_inputs(self) -> tuple[np.ndarray, list[ObstacleSpec]]:
        blocker = ObstacleSpec(
            name="screen",
            kind="box",
            position=(1.50, 0.00, 0.50),
            size=(0.08, 0.24, 0.50),
        )
        return np.array((3.50, 0.00), dtype=np.float64), [blocker]

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_search_planner_rollout_follows_the_executor(self) -> None:
        """The rollout model is whatever executes, never a hardcoded class."""
        target_xy, obstacles = self._rollout_probe_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            model, data = self._planner_rollout_probe_scene(tmp, target_xy, obstacles)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0)
            planner = SearchMPCPlanner(
                model,
                target_xy,
                obstacles,
                SearchMPCConfig(use_route_planner_when_obstacles=False),
            )
            self.assertEqual(planner._executor_backend(controller), "analytic")
            rollout = planner._rollout_controller(controller)
            self.assertIs(type(rollout), M20LowLevelController)
            # One instance is shared by every candidate sequence.
            self.assertIs(planner._rollout_controller(controller), rollout)
            command, info = planner.recommend(data, controller)
            self.assertEqual(command.shape, (4,))
            self.assertGreater(len(info["candidate_scores"]), 0)

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_search_planner_rollout_uses_the_executing_v5_expert(self) -> None:
        """A v5 episode must also *predict* with v5, not with the analytic law."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        if not resolve_onnx_path().is_file():
            self.skipTest("v5 policy weights are not present")
        target_xy, obstacles = self._rollout_probe_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            model, data = self._planner_rollout_probe_scene(tmp, target_xy, obstacles)
            controller = M20V5PolicyController(model)
            controller.reset(data, 0.0)
            planner = SearchMPCPlanner(
                model,
                target_xy,
                obstacles,
                SearchMPCConfig(use_route_planner_when_obstacles=False),
            )
            self.assertEqual(planner._executor_backend(controller), "v5")
            self.assertIs(
                type(planner._rollout_controller(controller)),
                M20V5PolicyController,
            )
            command, info = planner.recommend(data, controller)
            self.assertEqual(command.shape, (4,))
            self.assertGreater(len(info["candidate_scores"]), 0)

    @unittest.skipUnless(ASSET.is_file(), "MuJoCo asset is not built")
    def test_search_planner_rejects_a_rollout_backend_mismatch(self) -> None:
        """Pinning a backend that disagrees with the executor must fail loudly."""
        target_xy, obstacles = self._rollout_probe_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            model, data = self._planner_rollout_probe_scene(tmp, target_xy, obstacles)
            controller = M20LowLevelController(model)
            controller.reset(data, 0.0)
            pinned = SearchMPCPlanner(
                model,
                target_xy,
                obstacles,
                SearchMPCConfig(use_route_planner_when_obstacles=False, backend="v5"),
            )
            with self.assertRaises(TypeError):
                pinned.recommend(data, controller)
            unknown = SearchMPCPlanner(
                model,
                target_xy,
                obstacles,
                SearchMPCConfig(
                    use_route_planner_when_obstacles=False,
                    backend="does-not-exist",
                ),
            )
            with self.assertRaises(ValueError):
                unknown.recommend(data, controller)


if __name__ == "__main__":
    unittest.main()

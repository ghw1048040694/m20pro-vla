"""Contract tests for structured (S2/S3) scene collection.

Wiring the new scenes into the collector is only useful if every episode it
emits is *solvable*: the target sits inside a room and the start pose is
outside the building, so the only legal route runs through one or more
doorways. These tests guard the four properties that keep that honest:

1. the plan builder emits one plan per object, each carrying the scene walls
   and its scene provenance, with episode ids that stay shard-friendly;
2. every emitted plan is reachable on the privileged grid (start -> doorway ->
   target) and the target starts occluded, so the policy cannot trivially
   shortcut the search;
3. the collector refuses a structured request that would silently produce a
   useless dataset -- the open-plane teacher never leaves the start pose, and a
   short budget truncates the episode while still passing every quality gate;
4. the legacy S1 request is untouched by those guards.
"""

from __future__ import annotations

import argparse
import json
import runpy
import unittest
from pathlib import Path

import numpy as np

from m20pro_vla.planning import GlobalPlanner, GlobalPlannerConfig
from m20pro_vla.sim.corridor import in_room

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mujoco/collect_m20_mujoco_vla.py"

SEED = 20261005
LAYOUTS = 3
OFFSET = 0
S3_ROOMS = {"north_room", "south_room", "end_room"}


def _args(**overrides) -> argparse.Namespace:
    base = {
        "scene": "s1",
        "scene_episode": "sampled",
        "mode": "search",
        "layouts": LAYOUTS,
        "layout_offset": OFFSET,
        "steps": 2800,
        "seed": SEED,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class StructuredSceneCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = runpy.run_path(str(SCRIPT))

    def _plans(self, scene: str, episode_kind: str, **overrides):
        args = _args(scene=scene, scene_episode=episode_kind, **overrides)
        rng = np.random.default_rng(SEED + 1009 * OFFSET)
        return self.ns["_structured_plans"](args, rng)

    def _episode(self, scene: str, episode_kind: str):
        if episode_kind == "default":
            return (
                self.ns["room_default_episode"]()
                if scene == "s2"
                else self.ns["corridor_default_episode"]()
            )
        sampler = self.ns["room_sample_episode"] if scene == "s2" else self.ns["corridor_sample_episode"]
        return sampler(np.random.default_rng(SEED + 1009 * OFFSET), attempts=64)

    @staticmethod
    def _s3_room(episode, point) -> str:
        for wing in episode.spec.rooms:
            if in_room(wing, point, margin=-0.05):
                return wing.name
        return "outside"

    @classmethod
    def _inside_building(cls, scene: str, episode, point) -> bool:
        if scene == "s2":
            x0, x1 = episode.spec.interior_x
            y0, y1 = episode.spec.interior_y
            return bool(x0 <= point[0] <= x1 and y0 <= point[1] <= y1)
        if cls._s3_room(episode, point) in S3_ROOMS:
            return True
        cx0, cx1 = episode.spec.corridor_x
        cy0, cy1 = episode.spec.corridor_y
        return bool(cx0 <= point[0] <= cx1 and cy0 <= point[1] <= cy1)

    # ------------------------------------------------------------------ plans

    def test_one_plan_per_object_with_walls_and_provenance(self) -> None:
        for scene in ("s2", "s3"):
            with self.subTest(scene=scene):
                plans = self._plans(scene, "sampled")
                self.assertEqual(len(plans), LAYOUTS * 3)
                self.assertEqual(len({plan.episode_id for plan in plans}), len(plans))
                for plan in plans:
                    self.assertEqual(plan.scene_kind, scene)
                    self.assertTrue(plan.scene_name)
                    self.assertIn(plan.target_label, {obj.label for obj in plan.objects})
                    # Walls must travel with the plan: build_scene renders them and
                    # both the teacher and the global planner read them from here.
                    self.assertGreaterEqual(len(plan.obstacles), 5)
                    self.assertTrue(all(obstacle.kind == "box" for obstacle in plan.obstacles))
                    self.assertTrue(plan.search_required)
                    # Episode ids are layout-major, so parallel shards cannot collide.
                    self.assertEqual(plan.episode_id // 3, plan.layout_id)

    def test_structured_walls_helper_matches_the_plan_payload(self) -> None:
        for scene in ("s2", "s3"):
            with self.subTest(scene=scene):
                episode = self._episode(scene, "default")
                walls = self.ns["_structured_scene_walls"](episode, scene)
                plan = self._plans(scene, "default")[0]
                self.assertEqual(len(walls), len(plan.obstacles))

    def test_default_episode_yields_exactly_one_layout(self) -> None:
        # Repeating the one hand-checked building per layout would only duplicate
        # geometry under fresh episode ids.
        plans = self._plans("s2", "default", layouts=5)
        self.assertEqual(len(plans), 3)
        self.assertEqual({plan.layout_id for plan in plans}, {OFFSET})

    def test_every_plan_is_solvable_on_the_privileged_grid(self) -> None:
        for scene in ("s2", "s3"):
            for episode_kind in ("default", "sampled"):
                with self.subTest(scene=scene, episode_kind=episode_kind):
                    plans = self._plans(scene, episode_kind)
                    self.assertTrue(plans)
                    for plan in plans:
                        self.assertFalse(
                            plan.initial_target_visible,
                            "a structured target that starts visible is not a search episode",
                        )
                        result = GlobalPlanner(plan.obstacles, GlobalPlannerConfig()).plan(
                            tuple(plan.start_xy), tuple(plan.target_xy)
                        )
                        self.assertTrue(result.reachable, result.reason)
                        self.assertGreater(result.cost_m, 1.0)

    def test_start_is_outside_and_target_is_inside_a_room(self) -> None:
        for scene in ("s2", "s3"):
            for episode_kind in ("default", "sampled"):
                with self.subTest(scene=scene, episode_kind=episode_kind):
                    episode = self._episode(scene, episode_kind)
                    for plan in self._plans(scene, episode_kind):
                        self.assertFalse(
                            self._inside_building(scene, episode, plan.start_xy),
                            "the start pose must be outside the building",
                        )
                        if scene == "s2":
                            self.assertTrue(self._inside_building(scene, episode, plan.target_xy))
                        else:
                            self.assertIn(self._s3_room(episode, plan.target_xy), S3_ROOMS)

    # ------------------------------------------------------------ arg guards

    def test_structured_scene_requires_the_search_teacher(self) -> None:
        for scene in ("s2", "s3"):
            with self.subTest(scene=scene):
                with self.assertRaises(ValueError) as caught:
                    self.ns["_validate_structured_scene_args"](_args(scene=scene, mode="randomized"))
                self.assertIn("--mode search", str(caught.exception))

    def test_visible_approach_has_safe_observable_travel_and_preserves_scene(self) -> None:
        hidden = self._plans("s2", "sampled")
        visible = self._plans("s2", "sampled", structured_start_mode="visible-approach")
        self.assertEqual(len(hidden), len(visible))
        for before, after in zip(hidden, visible):
            self.assertEqual(before.objects, after.objects)
            self.assertEqual(before.obstacles, after.obstacles)
            self.assertFalse(after.search_required)
            self.assertTrue(after.initial_target_visible)
            self.assertEqual(after.search_start_variant, "visible-approach")
            distance = float(np.linalg.norm(after.target_xy - after.start_xy))
            self.assertGreaterEqual(distance, 1.6)
            self.assertLessEqual(distance, 3.0)
            self.assertLessEqual(abs(after.search_start_yaw_offset), 0.55)
            self.assertTrue(GlobalPlanner(after.obstacles).plan(tuple(after.start_xy), tuple(after.target_xy)).reachable)

    def test_visible_approach_budget_and_scene_are_validated(self) -> None:
        self.assertTrue(self.ns["_validate_structured_scene_args"](
            _args(scene="s2", structured_start_mode="visible-approach", steps=1200)))
        with self.assertRaises(ValueError):
            self.ns["_validate_structured_scene_args"](
                _args(scene="s2", structured_start_mode="visible-approach", steps=599))
        with self.assertRaises(ValueError):
            self.ns["_validate_structured_scene_args"](
                _args(scene="s3", structured_start_mode="visible-approach"))

    def test_structured_scene_rejects_a_budget_that_truncates(self) -> None:
        floor = self.ns["STRUCTURED_SCENE_MIN_STEPS"]
        with self.assertRaises(ValueError) as caught:
            self.ns["_validate_structured_scene_args"](_args(scene="s2", steps=floor - 1))
        self.assertIn("steps", str(caught.exception))
        self.assertTrue(self.ns["_validate_structured_scene_args"](_args(scene="s2", steps=floor)))

    def test_s1_request_is_not_guarded_as_structured(self) -> None:
        self.assertFalse(self.ns["_validate_structured_scene_args"](_args(scene="s1", mode="randomized")))
        self.assertFalse(self.ns["_validate_structured_scene_args"](_args(scene="s1", steps=240)))

    def test_defaults_keep_the_legacy_scene_and_the_strict_contact_budget(self) -> None:
        # The new knobs must be inert unless asked for: an unmodified invocation
        # has to reproduce the historical S1 open-plane contract, including the
        # "zero obstacle contact" rule that structured runs loosen deliberately.
        import sys

        argv = sys.argv
        sys.argv = ["collect_m20_mujoco_vla.py"]
        try:
            parsed = self.ns["parse_args"]()
        finally:
            sys.argv = argv
        self.assertEqual(parsed.scene, "s1")
        self.assertEqual(parsed.mode, "randomized")
        self.assertEqual(parsed.quality_max_obstacle_contact_steps, 0)


class PlanOnlyEpisodeTest(unittest.TestCase):
    """A declared (never collected) episode must carry exactly what the closed-
    loop player reads back, because a ``--plan-only`` run writes nothing else.

    This is the mechanism behind the S3 held-out set: it must be reproducible
    without spending a teacher rollout, yet still rebuild into a solvable scene.
    """

    # Every key ``play_m20_smolvla`` / ``play_m20_mujoco_vla`` reads from the
    # episode JSON. A missing one only surfaces as a KeyError deep inside a run.
    PLAYER_KEYS = (
        "target_xy_privileged_label_only",
        "target_label",
        "task_text",
        "initial_xy",
        "initial_yaw",
        "warmup_steps",
        "objects",
        "obstacles",
        "scene_light",
        "steps",
        "scene_kind",
        "scene_name",
        "scene_episode",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = runpy.run_path(str(SCRIPT))

    def _plans(self):
        args = argparse.Namespace(
            scene="s3",
            scene_episode="sampled",
            mode="search",
            layouts=1,
            layout_offset=3000,
            steps=2600,
            warmup_steps=35,
            search_curriculum_success_radius=0.95,
            language_mode="sampled",
            seed=SEED,
        )
        rng = np.random.default_rng(SEED + 1009 * args.layout_offset)
        plans = self.ns["_structured_plans"](args, rng)
        self.assertTrue(plans, "the S3 sampler returned no plans")
        return plans, args

    def test_carries_every_key_the_player_reads(self) -> None:
        plans, args = self._plans()
        plan = plans[0]
        meta = self.ns["_plan_only_metadata"](args, plan)
        for key in self.PLAYER_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, meta)
        self.assertEqual(meta["episode_id"], plan.episode_id)
        self.assertEqual(meta["layout_id"], plan.layout_id)
        self.assertEqual(meta["scene_kind"], "s3")
        self.assertTrue(meta["plan_only"])
        self.assertEqual(meta["steps"], 2600)
        self.assertEqual(len(meta["objects"]), len(plan.objects))
        self.assertGreaterEqual(len(meta["obstacles"]), 5)
        self.assertTrue(all(item.get("size") for item in meta["obstacles"]))
        self.assertEqual(meta["target_xy_privileged_label_only"], plan.target_xy.tolist())

    def test_declared_episode_omits_rollout_fields(self) -> None:
        # Nothing a rollout alone can produce may leak in, or a declared set would
        # masquerade as a collected one in the dataset summary.
        plans, args = self._plans()
        meta = self.ns["_plan_only_metadata"](args, plans[0])
        for key in ("success", "min_target_distance", "target_reached_step", "steps_executed"):
            with self.subTest(key=key):
                self.assertNotIn(key, meta)

    def test_metadata_survives_json_round_trip(self) -> None:
        plans, args = self._plans()
        meta = self.ns["_plan_only_metadata"](args, plans[0])
        self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_plan_only_defaults_off(self) -> None:
        import sys

        argv = sys.argv
        sys.argv = ["collect_m20_mujoco_vla.py"]
        try:
            parsed = self.ns["parse_args"]()
        finally:
            sys.argv = argv
        self.assertFalse(parsed.plan_only)


if __name__ == "__main__":
    unittest.main()

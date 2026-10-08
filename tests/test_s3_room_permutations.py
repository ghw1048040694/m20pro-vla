import argparse
import dataclasses
import json
from pathlib import Path
import runpy
import unittest

import numpy as np

from m20pro_vla.sim.corridor import (
    ROOM_OBJECT_PERMUTATIONS, all_walls, sample_episode, scene_report,
)
from m20pro_vla.planning import GlobalPlanner


class S3RoomPermutationTests(unittest.TestCase):
    def test_all_assignments_preserve_geometry_and_scene_contract(self):
        base = sample_episode(np.random.default_rng(42))
        for order in ROOM_OBJECT_PERMUTATIONS:
            episode = sample_episode(np.random.default_rng(42), room_object_order=order)
            self.assertIsNotNone(episode)
            self.assertEqual(all_walls(base.spec), all_walls(episode.spec))
            self.assertEqual(base.start_xy, episode.start_xy)
            self.assertEqual(base.start_yaw, episode.start_yaw)
            self.assertEqual(tuple(wing.object_name for wing in episode.spec.rooms), order)
            self.assertTrue(scene_report(episode, resolution=0.06)['ok'])
            for obj in episode.objects:
                self.assertTrue(GlobalPlanner(all_walls(episode.spec)).plan(episode.start_xy, obj.position).reachable)
        self.assertEqual(base, sample_episode(np.random.default_rng(42), room_object_order=ROOM_OBJECT_PERMUTATIONS[0]))

    def test_invalid_assignments_are_refused(self):
        for order in (('red_cube',) * 3, ('green_cylinder', 'yellow_box', 'unknown'), ()):
            with self.assertRaises(ValueError):
                sample_episode(np.random.default_rng(42), room_object_order=order)

    def test_collector_balances_assignment_and_preserves_three_task_counterfactuals(self):
        ns = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/mujoco/collect_m20_mujoco_vla.py'))
        args = argparse.Namespace(scene='s3', scene_episode='sampled', s3_room_assignment='balanced-permutations',
            mode='search', layouts=6, layout_offset=6000, seed=20261008, steps=2600,
            warmup_steps=35, language_mode='varied', search_curriculum_success_radius=0.95)
        self.assertTrue(ns['_validate_structured_scene_args'](args))
        plans = ns['_structured_plans'](args, np.random.default_rng(args.seed + 1009 * args.layout_offset))
        self.assertEqual(len(plans), 18)
        for index in range(6):
            group = plans[index*3:index*3+3]
            assignment = dict(group[0].room_object_assignment)
            self.assertEqual(tuple(assignment.values()), ROOM_OBJECT_PERMUTATIONS[index])
            self.assertEqual(len({plan.target_label for plan in group}), 3)
            for plan in group:
                self.assertEqual(plan.objects, group[0].objects)
                self.assertEqual(plan.obstacles, group[0].obstacles)
                np.testing.assert_array_equal(plan.start_xy, group[0].start_xy)
                self.assertFalse(plan.initial_target_visible)
                meta = ns['_plan_only_metadata'](args, plan)
                self.assertEqual(meta['room_object_assignment_privileged_metadata_only'], assignment)
                self.assertEqual(json.loads(json.dumps(meta)), meta)
        for scene, episode in (('s2', 'sampled'), ('s3', 'default')):
            args.scene, args.scene_episode = scene, episode
            with self.assertRaises(ValueError):
                ns['_validate_structured_scene_args'](args)

    def test_observation_teacher_rejects_metadata_only_and_non_s3(self):
        ns = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/mujoco/collect_m20_mujoco_vla.py'))
        args=argparse.Namespace(scene='s3', scene_episode='sampled', mode='search',steps=8000,
                                s3_search_teacher='observe-then-route',metadata_only=False)
        self.assertTrue(ns['_validate_structured_scene_args'](args))
        for scene,metadata in (('s2',False),('s3',True)):
            args.scene,args.metadata_only=scene,metadata
            with self.assertRaises(ValueError): ns['_validate_structured_scene_args'](args)


if __name__ == '__main__':
    unittest.main()

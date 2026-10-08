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

    def test_interior_handoff_requires_observation_teacher(self):
        ns = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/mujoco/collect_m20_mujoco_vla.py'))
        args=argparse.Namespace(scene='s3',scene_episode='sampled',mode='search',steps=18000,
            s3_search_teacher='observe-then-route',s3_discovery_handoff='interior-center',metadata_only=False)
        self.assertTrue(ns['_validate_structured_scene_args'](args))
        args.s3_search_teacher='privileged-target'
        with self.assertRaises(ValueError):ns['_validate_structured_scene_args'](args)


    def test_same_room_center_handoff_changes_real_planner_stop_distance(self):
        from types import SimpleNamespace
        from m20pro_vla.planning import SearchMPCPlanner, SearchMPCConfig
        ns = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/mujoco/collect_m20_mujoco_vla.py'))
        center = np.asarray([0., 0.]); grid = GlobalPlanner(())
        old = SearchMPCPlanner(None, center, (), SearchMPCConfig(stop_distance=.65), global_planner=grid)
        data = SimpleNamespace(qpos=np.array([.5, 0., .57, 0., 0., 0., 1.]))
        old_action, _ = old._global_route_recommend(data)
        self.assertEqual(old_action[3], 1.)
        new = ns['_room_search_planner_for_decision'](old, None, center, (), grid, 'handoff')
        self.assertIsNot(new, old)
        self.assertEqual(new.config.stop_distance, .20)
        action, _ = new._global_route_recommend(data)
        self.assertGreater(action[0], 0.)
        self.assertEqual(action[3], 0.)
        self.assertIs(ns['_room_search_planner_for_decision'](new,None,center,(),grid,'handoff'),new)
        # A phase change must restore the original target radius even at the same coordinates.
        target = ns['_room_search_planner_for_decision'](new,None,center,(),grid,'target')
        self.assertEqual(target.config.stop_distance, .65)
        self.assertIs(ns['_room_search_planner_for_decision'](target,None,center,(),grid,'target'),target)
        changed = ns['_room_search_planner_for_decision'](target,None,np.array([1.,0.]),(),grid,'target')
        np.testing.assert_array_equal(changed.target_xy, [1.,0.])


if __name__ == '__main__':
    unittest.main()

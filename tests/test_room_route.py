import unittest
from dataclasses import replace
import numpy as np
from m20pro_vla.planning import GlobalPlanner, SearchMPCConfig
from m20pro_vla.planning.room_route import RoomSearchRoutePlanner


class RoomRouteTests(unittest.TestCase):
    def setUp(self):
        self.corner = (3.620376008587874, .7065435527079567)
        self.goal = (3.727134524851276, 2.743030969244465)
        self.base = GlobalPlanner(()).plan((3.25, .68), self.goal)
        self.planner = RoomSearchRoutePlanner(None, np.array(self.goal), (), SearchMPCConfig())

    def plan(self, start, prefix=()):
        return replace(self.base, waypoints=(start, *prefix, self.goal))

    def establish_completed_corner(self):
        self.planner._global_plan = self.plan((3.25, .68), (self.corner,))
        _, index, final = self.planner._global_route_goal(np.array([3.2735, .682]))
        self.assertEqual(index, 2)
        self.assertTrue(final)

    def test_refreshed_same_corner_does_not_reverse_progress_after_small_pose_drift(self):
        self.establish_completed_corner()
        pose = np.array([3.2706716531308384, .6874500646683999])
        self.assertGreater(np.linalg.norm(pose - self.corner), .35)
        self.planner._global_plan = self.plan(tuple(pose), (self.corner,))
        self.planner._global_waypoint_index = 0
        goal, index, final = self.planner._global_route_goal(pose)
        np.testing.assert_array_equal(goal, self.goal)
        self.assertEqual(index, 2)
        self.assertTrue(final)

    def test_new_detour_point_must_be_followed(self):
        self.establish_completed_corner()
        pose = np.array([3.27067, .68745])
        new_corner = (3.0, 1.4)
        self.planner._global_plan = self.plan(tuple(pose), (new_corner,))
        self.planner._global_waypoint_index = 0
        goal, index, final = self.planner._global_route_goal(pose)
        np.testing.assert_array_equal(goal, new_corner)
        self.assertFalse(final)

    def test_off_path_pose_does_not_skip_recovery_waypoint(self):
        self.establish_completed_corner()
        pose = np.array([2.4, .0])
        self.planner._global_plan = self.plan(tuple(pose), (self.corner,))
        self.planner._global_waypoint_index = 0
        goal, _, final = self.planner._global_route_goal(pose)
        np.testing.assert_array_equal(goal, self.corner)
        self.assertFalse(final)

    def test_new_suffix_must_not_reuse_old_progress(self):
        self.establish_completed_corner()
        pose = np.array([3.27067, .68745])
        extra = (3.7, 1.5)
        self.planner._global_plan = self.plan(tuple(pose), (self.corner, extra))
        self.planner._global_waypoint_index = 0
        goal, _, final = self.planner._global_route_goal(pose)
        np.testing.assert_array_equal(goal, self.corner)
        self.assertFalse(final)

    def test_direct_path_reset_clears_saved_progress(self):
        self.establish_completed_corner()
        self.planner._reset_global_plan()
        self.assertIsNone(self.planner._room_previous_plan)
        pose = np.array([3.27067, .68745])
        self.planner._global_plan = self.plan(tuple(pose), (self.corner,))
        goal, _, final = self.planner._global_route_goal(pose)
        np.testing.assert_array_equal(goal, self.corner)
        self.assertFalse(final)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from m20pro_vla.planning.teacher_motion import ResponsiveTeacherMotion
from m20pro_vla.low_level import M20V5PolicyController


class ResponsiveTeacherTest(unittest.TestCase):
    def run_command(self, teacher, rays=None, desired=(.22, 0, 0, 0), goal=(4, 0), xy=(0, 0), scan=False):
        return teacher.command(desired, xy=xy, yaw=0, local_goal=goal,
                               lidar=np.full(72, 5.) if rays is None else rays, scan=scan)

    def test_retreat_requires_clear_rear_and_preserves_sign(self):
        rays = np.full(72, 5.); rays[32:41] = .6
        t = ResponsiveTeacherMotion()
        a, info = self.run_command(t, rays)
        self.assertLess(a[0], 0); self.assertEqual(info['phase'], 'rear_clear_retreat')
        rays[0] = .4
        a, info = self.run_command(t, rays)
        self.assertTrue(np.array_equal(a, [0, 0, 0, 1]))
        self.assertEqual(info['phase'], 'recovery_settle')

    def test_blocked_both_directions_cannot_reverse(self):
        rays = np.full(72, 5.); rays[32:41] = .4; rays[0] = .4
        a, info = self.run_command(ResponsiveTeacherMotion(), rays)
        self.assertTrue(np.array_equal(a, [0, 0, 0, 1]))
        self.assertEqual(info['shield'], 'emergency_stop')

    def test_stop_wins_over_retreat(self):
        t = ResponsiveTeacherMotion(); rays = np.full(72, 5.); rays[32:41] = .6
        self.run_command(t, rays)
        a, info = self.run_command(t, rays, desired=(.4, 0, .4, 1))
        self.assertTrue(np.array_equal(a, [0, 0, 0, 1])); self.assertIsNone(t.retreat_start)

    def test_large_turn_does_not_drive_forward(self):
        t = ResponsiveTeacherMotion()
        for _ in range(100):
            a, info = self.run_command(t, goal=(0, 4))
            self.assertEqual(a[0], 0)
        self.assertGreater(a[2], .39)

    def test_clear_straight_accelerates_and_stop_resumes(self):
        t = ResponsiveTeacherMotion()
        for _ in range(120): a, _ = self.run_command(t)
        self.assertGreater(a[0], .49)
        self.run_command(t, desired=(0, 0, 0, 1))
        a, _ = self.run_command(t)
        self.assertGreater(a[0], 0); self.assertEqual(a[3], 0)

    def test_invalid_lidar_stops(self):
        rays = np.full(72, 5.); rays[0] = np.nan
        a, _ = self.run_command(ResponsiveTeacherMotion(), rays)
        self.assertTrue(np.array_equal(a, [0, 0, 0, 1]))

    def test_small_heading_error_does_not_hold_fast_drive_at_point_one(self):
        t = ResponsiveTeacherMotion(previous=np.array([0., 0., .1, 0.]))
        for _ in range(150): a, _ = self.run_command(t, goal=(4, .20))
        self.assertGreater(a[0], .49)
        self.assertAlmostEqual(a[2], 0., places=5)

    def test_legacy_controller_command_unchanged(self):
        c = M20V5PolicyController.__new__(M20V5PolicyController)
        self.assertTrue(np.array_equal(c._command([.5, 0, .4, 0]).as_array(), [.4, 0, .15, 0]))
        c.teacher_motion_limits = (.5, .4)
        self.assertTrue(np.array_equal(c._command([.5, 0, .4, 0]).as_array(), [.5, 0, .4, 0]))


if __name__ == '__main__': unittest.main()

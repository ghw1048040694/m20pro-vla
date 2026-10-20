import unittest
import numpy as np
from m20pro_vla.sim.keyboard_teleop import keyboard_request, executed_request, STOP


class KeyboardContractTests(unittest.TestCase):
    def test_release_timeout_and_space_stop(self):
        for keys, fresh in [([], True), (['KeyW'], False), (['KeyW', 'Space'], True), (['KeyW', 'KeyS'], True)]:
            np.testing.assert_array_equal(keyboard_request(keys, fresh), STOP)

    def test_signed_reverse_and_fast(self):
        self.assertEqual(keyboard_request(['KeyS'], True)[0], -.2)
        self.assertEqual(keyboard_request(['KeyW', 'ShiftLeft'], True)[0], .5)
        self.assertEqual(keyboard_request(['KeyA'], True)[2], .4)
        self.assertEqual(keyboard_request(['KeyD'], True)[2], -.4)

    def test_directional_safety(self):
        scan = np.full(72, 10.)
        scan[0] = .4
        a, why = executed_request(keyboard_request(['KeyS'], True), STOP, scan)
        np.testing.assert_array_equal(a, STOP)
        self.assertEqual(why, 'emergency_stop')
        a, why = executed_request(keyboard_request(['KeyW'], True), STOP, scan)
        self.assertGreater(a[0], 0)
        scan[:] = 10; scan[36] = .4
        a, why = executed_request(keyboard_request(['KeyS'], True), STOP, scan)
        self.assertLess(a[0], 0)

    def test_stop_then_resume(self):
        scan = np.full(72, 10.)
        stopped, _ = executed_request(STOP, np.array([.35, 0, .1, 0]), scan)
        np.testing.assert_array_equal(stopped, STOP)
        move, _ = executed_request(keyboard_request(['KeyW'], True), stopped, scan)
        self.assertGreater(move[0], 0)
        self.assertEqual(move[3], 0)

    def test_no_fast_sharp_arc(self):
        a = keyboard_request(['KeyW', 'KeyA', 'ShiftLeft'], True)
        self.assertLessEqual(a[0], .18)
        self.assertLessEqual(a[2], .15)
        a, _ = executed_request(keyboard_request(['KeyA'], True), np.array([.5, 0, 0, 0]), np.full(72, 10.))
        self.assertLessEqual(a[2], .15)
        self.assertLess(a[0], .5)

    def test_invalid_lidar(self):
        a, why = executed_request(keyboard_request(['KeyW'], True), STOP, np.full(72, np.nan))
        np.testing.assert_array_equal(a, STOP)
        self.assertEqual(why, 'invalid_lidar')


if __name__ == '__main__':
    unittest.main()

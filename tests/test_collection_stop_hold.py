"""Regression for a successful-looking collection that resumes motion at its end."""
from pathlib import Path
import runpy
import unittest
import numpy as np


class CollectionStopHoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/mujoco/collect_m20_mujoco_vla.py'))
        cls.update = staticmethod(ns['_update_search_stop_tail'])

    def tail_count(self, commands):
        count = 0
        for command in commands:
            count = self.update(count, np.asarray(command, dtype=np.float64))
        return count

    def test_15056_resumed_motion_cannot_pass_terminal_hold(self):
        # Observed episode15056:20 stops, then forward0.14 with stop0.
        commands = [(0., 0., 0., 1.)] * 20 + [(.14, 0., -.01789, 0.)]
        self.assertEqual(self.tail_count(commands), 0)

    def test_collection_terminates_on_twentieth_actual_stop(self):
        commands = [(.14, 0., 0., 0.)] * 981 + [(0., 0., 0., 1.)] * 20
        count = 0
        for index, command in enumerate(commands):
            count = self.update(count, np.asarray(command))
            if count >= 20:
                break
        self.assertEqual(index, 1000)
        self.assertEqual(count, 20)

    def test_non_consecutive_stops_do_not_accumulate(self):
        commands = [(0., 0., 0., 1.)] * 19 + [(.14, 0., 0., 0.)] + [(0., 0., 0., 1.)] * 19
        self.assertEqual(self.tail_count(commands), 19)

    def test_motion_or_nonfinite_label_resets_hold(self):
        for command in ((.14, 0., 0., 1.), (0., 0., .01, 1.), (np.nan, 0., 0., 1.)):
            with self.subTest(command=command):
                self.assertEqual(self.update(19, np.asarray(command)), 0)


if __name__ == '__main__':
    unittest.main()

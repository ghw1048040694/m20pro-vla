import unittest
import math
from m20pro_vla.planning.room_search import RoomSearchSchedule


class ScanAnchorTests(unittest.TestCase):
    def schedule(self):
        return RoomSearchSchedule(((0.,0.),(3.,0.)), scan_start_radius=.25)

    def test_scan_requires_entry_anchor_not_only_outer_area(self):
        s=self.schedule()
        self.assertEqual(s.advance((.7,0.),0.,0)['mode'],'explore')
        self.assertEqual(s.advance((.3,0.),.1,0)['mode'],'explore')
        self.assertEqual(s.advance((.24,0.),.2,0)['mode'],'scan')
        self.assertEqual(s.swept,0.)

    def test_scan_keeps_actual_yaw_progress_during_turn_drift(self):
        s=self.schedule();s.advance((.24,0.),0.,0)
        for i in range(1,35):
            decision=s.advance((.6,0.),i*.1,0)
        self.assertEqual(s.completed_room_scans,1)
        self.assertEqual(s.room_index,1)
        self.assertEqual(decision['mode'],'explore')

    def test_leaving_outer_area_still_aborts_scan(self):
        s=self.schedule();s.advance((.2,0.),0.,0);s.advance((.6,0.),.1,0)
        self.assertEqual(s.advance((.76,0.),.2,0)['mode'],'explore')
        self.assertEqual(s.swept,0.)
        self.assertIsNone(s.scan_yaw)
        self.assertEqual(s.advance((.6,0.),.3,0)['mode'],'explore')

    def test_discovery_can_interrupt_entry_without_privileged_room_identity(self):
        s=self.schedule()
        for i in range(3):decision=s.advance((.6,0.),0.,5)
        self.assertEqual(decision['mode'],'target')
        self.assertTrue(s.discovered)
        self.assertEqual(s.completed_room_scans,0)

    def test_invalid_anchor_is_rejected_and_default_retains_original_behavior(self):
        for radius in (0.,-.1,.8,math.nan,math.inf):
            with self.assertRaises(ValueError):RoomSearchSchedule(((0.,0.),),scan_start_radius=radius)
        self.assertEqual(RoomSearchSchedule(((0.,0.),)).advance((.7,0.),0.,0)['mode'],'scan')


if __name__=='__main__':unittest.main()

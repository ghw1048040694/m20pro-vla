import math
import unittest
from m20pro_vla.planning.room_search import RoomSearchSchedule


class RoomSearchTests(unittest.TestCase):
    centers=((3.,3.),(5.,-3.),(8.,0.))

    def test_hidden_targets_have_identical_exploration_and_no_goal_identity(self):
        schedules=[RoomSearchSchedule(self.centers) for _ in range(6)]
        for xy,yaw,pixels in [((0.,0.),0.,0),((2.,0.),0.,0),((3.,3.),1.,0)]:
            decisions=[s.advance(xy,yaw,pixels) for s in schedules]
            self.assertTrue(all(d==decisions[0] for d in decisions))
            self.assertNotEqual(decisions[0]['mode'],'target')
        self.assertNotIn('target_xy',schedules[0].__dict__)

    def test_discovery_uses_consecutive_rgb_and_survives_later_occlusion(self):
        schedule=RoomSearchSchedule(self.centers)
        for pixels in (5,0,10,10):
            self.assertNotEqual(schedule.advance((0.,0.),0.,pixels)['mode'],'target')
        self.assertEqual(schedule.advance((0.,0.),0.,10)['mode'],'target')
        self.assertEqual(schedule.advance((0.,0.),0.,0)['mode'],'target')

    def test_room_scan_requires_actual_rotation_and_wraps_heading(self):
        schedule=RoomSearchSchedule(self.centers,sweep_radians=0.3)
        self.assertEqual(schedule.advance(self.centers[0],3.1,0)['mode'],'scan')
        for _ in range(5):
            self.assertEqual(schedule.advance(self.centers[0],3.1,0)['mode'],'scan')
        self.assertEqual(schedule.advance(self.centers[0],-3.1,0)['mode'],'scan')
        for yaw in (-3.0,-2.9,-2.8):
            decision=schedule.advance(self.centers[0],yaw,0)
        self.assertEqual(decision['mode'],'explore')
        self.assertEqual(decision['goal_xy'],self.centers[1])
        self.assertEqual(schedule.completed_room_scans,1)

    def test_invalid_or_teleported_observations_do_not_count_coverage(self):
        schedule=RoomSearchSchedule(self.centers)
        schedule.advance(self.centers[0],0.,0)
        schedule.advance(self.centers[0],2.,0)
        self.assertEqual(schedule.swept,0.)
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): schedule.advance((0.,0.),0.,value)


if __name__=='__main__': unittest.main()

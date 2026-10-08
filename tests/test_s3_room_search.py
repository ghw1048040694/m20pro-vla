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

    def test_heading_jitter_cannot_finish_a_camera_sweep(self):
        schedule=RoomSearchSchedule(self.centers,sweep_radians=0.3)
        schedule.advance(self.centers[0],0.,0)
        for _ in range(30):
            for yaw in (0.05,-0.05):
                self.assertEqual(schedule.advance(self.centers[0],yaw,0)['mode'],'scan')
        self.assertEqual(schedule.completed_room_scans,0)

    def test_interior_discovery_completes_entry_and_keeps_rgb_memory(self):
        schedule=RoomSearchSchedule(self.centers,safe_handoff_bounds=((2.,4.,1.,4.),(4.,6.,-4.,-1.),(7.,9.,-1.,1.)))
        for _ in range(3): decision=schedule.advance((3.,1.5),1.4,6)
        self.assertTrue(schedule.discovered)
        self.assertEqual(decision['mode'],'handoff')
        self.assertEqual(decision['goal_xy'],self.centers[0])
        self.assertEqual(schedule.advance((3.,2.),1.4,0)['mode'],'handoff')
        self.assertEqual(schedule.advance((3.,2.8),1.4,0)['mode'],'target')
        self.assertEqual(schedule.advance((3.,1.5),1.4,0)['mode'],'target')

    def test_corridor_discovery_has_no_extra_room_visit(self):
        schedule=RoomSearchSchedule(self.centers,safe_handoff_bounds=((2.,4.,1.,4.),(4.,6.,-4.,-1.),(7.,9.,-1.,1.)))
        for _ in range(3): decision=schedule.advance((2.6,.45),.29,5)
        self.assertEqual(decision['mode'],'target')
        self.assertIsNone(schedule.handoff_room_index)

    def test_handoff_geometry_is_validated(self):
        for bounds in (((0.,1.,0.,1.),), ((0.,1.,0.,1.),)*3):
            with self.assertRaises(ValueError):RoomSearchSchedule(self.centers,safe_handoff_bounds=bounds)


if __name__=='__main__': unittest.main()

import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from m20pro_vla.data.execution_contract import (
    HUMAN_CONTRACT, load_execution_contract, save_execution_contract,
    validate_execution_contract, physical_action,
)
from m20pro_vla.sim.keyboard_teleop import executed_request, STOP
from m20pro_vla.data.lerobot_adapter import m20_smolvla_state


class HumanExecutionContractTest(unittest.TestCase):
    def test_saved_contract_roundtrip_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_execution_contract(d))
            save_execution_contract(d, HUMAN_CONTRACT)
            self.assertEqual(load_execution_contract(d), HUMAN_CONTRACT)
            wrong = {**HUMAN_CONTRACT, 'yaw_limits': [-.15, .15]}
            with self.assertRaises(ValueError):validate_execution_contract(wrong)

    def test_signed_physical_range_stop_and_invalid_output(self):
        np.testing.assert_array_equal(physical_action([-.25, .1, -.4, 0]), [-.25,0,-.4,0])
        np.testing.assert_array_equal(physical_action([.5, 0, .4, 0]), [.5,0,.4,0])
        np.testing.assert_array_equal(physical_action([.5, 0, .4, .5]), STOP)
        for raw in ([0,0,0], [np.nan,0,0,0]):
            with self.assertRaises(ValueError):physical_action(raw)

    def test_exact_human_50hz_acceleration_and_directional_stop(self):
        scan = np.full(72, 4.);previous = STOP.copy()
        for _ in range(50):previous, _ = executed_request(physical_action([.5,0,0,0]),previous,scan)
        self.assertAlmostEqual(previous[0], .5)
        rear=scan.copy();rear[0]=.3
        command,reason=executed_request(physical_action([-.2,0,0,0]),STOP,rear)
        np.testing.assert_array_equal(command,STOP);self.assertEqual(reason,'emergency_stop')

    def test_large_turn_does_not_accelerate_into_fast_turn_combination(self):
        command,_=executed_request(physical_action([.5,0,.4,0]),np.array([0,0,.4,0]),np.full(72,4.))
        self.assertLessEqual(command[0],.01)
        previous=np.array([.5,0,0,0])
        command,_=executed_request(physical_action([0,0,.4,0]),previous,np.full(72,4.))
        self.assertAlmostEqual(command[0],.49);self.assertAlmostEqual(command[2],.0075)

    def test_smolvla_state_does_not_use_world_or_previous_action_history(self):
        p=np.arange(45,dtype=np.float32);lidar=np.arange(72,dtype=np.float32)
        state=m20_smolvla_state(p,lidar)
        p[:3]=9999
        np.testing.assert_array_equal(m20_smolvla_state(p,lidar),state)
        self.assertEqual(state.shape,(32,))

if __name__=='__main__':unittest.main()

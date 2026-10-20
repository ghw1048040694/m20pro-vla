import unittest
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from m20pro_vla.low_level import LOW_LEVEL_CONTRACT, M20BodyCommand
from m20pro_vla.low_level import controller as controller_module
from m20pro_vla.low_level import gate as gate_module


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/m20pro_low_level_v1.yaml"

# The contract is the single authority for the acceptance numbers; the gate
# keeps them as module constants so its bench functions stay callable without a
# file system. This table is what stops the two from drifting apart - the same
# failure mode that once let the contract document one leg damping value while
# the asset builder silently emitted another.
THRESHOLD_BINDINGS = (
    ("CRUISE_SPEED_MPS", "cruise_bench_speed_mps"),
    ("CRUISE_STEPS", "cruise_bench_steps"),
    ("CRUISE_SETTLED_FRACTION", "cruise_bench_settled_fraction"),
    ("MAX_SETTLED_VERTICAL_VELOCITY_RMS_MPS", "cruise_settled_vertical_velocity_rms_mps_max"),
    ("MAX_SETTLED_ROLL_RATE_RMS_RADPS", "cruise_settled_roll_rate_rms_radps_max"),
    ("MAX_SETTLED_PITCH_RATE_RMS_RADPS", "cruise_settled_pitch_rate_rms_radps_max"),
    ("MAX_SETTLED_BASE_HEIGHT_RANGE_M", "cruise_settled_height_range_m_max"),
    ("CADENCE_STEPS", "cadence_bench_steps"),
    ("CADENCE_REPLAN_STEPS", "cadence_bench_replan_steps"),
    ("CADENCE_SEED", "cadence_bench_seed"),
    ("CADENCE_FORWARD_RANGE_MPS", "cadence_bench_forward_range_mps"),
    ("CADENCE_YAW_RANGE_RADPS", "cadence_bench_yaw_range_radps"),
    ("MAX_CADENCE_VERTICAL_VELOCITY_RMS_MPS", "cadence_settled_vertical_velocity_rms_mps_max"),
    ("MAX_CADENCE_ROLL_RATE_RMS_RADPS", "cadence_settled_roll_rate_rms_radps_max"),
    ("MAX_CADENCE_PITCH_RATE_RMS_RADPS", "cadence_settled_pitch_rate_rms_radps_max"),
    ("MAX_CADENCE_BASE_HEIGHT_RANGE_M", "cadence_settled_height_range_m_max"),
)


class LowLevelContractTest(unittest.TestCase):
    def test_startup_recovery_does_not_latch_wheel_drive(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / ".runtime/mujoco_assets/M20_floating_actuated.xml"))
        data = mujoco.MjData(model)
        controller = controller_module.M20LowLevelController(model)
        controller.reset(data, yaw=0.582906375888185)

        # Exercise release from startup recovery explicitly. Whether warmup
        # happens to dip is asset/friction-dependent and is not the contract.
        controller.safety_recovery_active = True
        for _ in range(35):
            controller.step(data, np.zeros(4))
        self.assertFalse(controller.safety_recovery_active)

        for _ in range(300):
            controller.step(data, np.asarray((0.35, 0.0, 0.15, 0.0)))
        self.assertGreater(float(np.linalg.norm(controller.wheel_target)), 0.5)
        self.assertGreater(float(np.linalg.norm(data.qpos[:2])), 0.05)

    def test_body_command_clipping_is_stable(self):
        command = M20BodyCommand(2.0, -2.0, 2.0, False).clipped()
        self.assertTrue(np.allclose(command.as_array(), (0.4, -0.2, 0.15, 0.0)))

    def test_low_level_contract_is_versioned(self):
        self.assertEqual(LOW_LEVEL_CONTRACT, "m20_low_level_v1")

    def test_turn_dynamics_match_contract(self):
        feedback = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["feedback"]
        self.assertEqual(controller_module.SAFETY_RECOVERY_HEIGHT, feedback["safety_recovery_trigger_height_m"])
        self.assertEqual(controller_module.SAFETY_RECOVERY_CLEAR_HEIGHT, feedback["safety_recovery_clear_height_m"])
        self.assertEqual(controller_module.TURN_STANCE_ATTITUDE_KP, feedback["turn_stance_attitude_kp"])
        self.assertEqual(controller_module.TURN_STANCE_RATE_KD, feedback["turn_stance_rate_kd"])
        self.assertEqual(
            controller_module.TURN_MAX_LEG_STANCE_OFFSET,
            feedback["turn_max_leg_stance_offset_rad"],
        )
        self.assertAlmostEqual(
            controller_module.TURN_STANCE_DEADBAND,
            math.radians(feedback["turn_stance_deadband_deg"]),
        )
        self.assertAlmostEqual(
            controller_module.TURN_STANCE_RAMP,
            math.radians(feedback["turn_stance_ramp_deg"]),
        )
        self.assertEqual(
            controller_module.TURN_STANCE_RELEASE_WHEEL_DELTA,
            feedback["turn_stance_release_wheel_delta"],
        )

    def test_gate_thresholds_match_contract(self):
        gates = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["acceptance_gates"]
        for constant_name, contract_key in THRESHOLD_BINDINGS:
            self.assertIn(contract_key, gates, contract_key)
            constant = getattr(gate_module, constant_name)
            expected = gates[contract_key]
            if isinstance(constant, (tuple, list)):
                self.assertEqual(
                    [float(value) for value in constant],
                    [float(value) for value in expected],
                    constant_name,
                )
            else:
                self.assertEqual(float(constant), float(expected), constant_name)


if __name__ == "__main__":
    unittest.main()

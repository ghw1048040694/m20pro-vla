from __future__ import annotations

import unittest

import numpy as np

from m20pro_vla.data.lerobot_adapter import M20_STATE_NAMES, m20_lerobot_frame_indices, m20_smolvla_state


class LeRobotAdapterTest(unittest.TestCase):
    def test_short_recovery_keeps_motion_and_stop_tail_without_eightfold_stop_inflation(self) -> None:
        actions = np.zeros((69, 4), dtype=np.float32)
        actions[49:, 3] = 1.0
        ordinary = m20_lerobot_frame_indices(actions, frame_stride=2, terminal_stop_repeat=8)
        recovery = m20_lerobot_frame_indices(
            actions, frame_stride=2, terminal_stop_repeat=8, collection_mode="failure_recovery",
        )
        self.assertEqual([i for i in ordinary if actions[i, 3] < 0.5],
                         [i for i in recovery if actions[i, 3] < 0.5])
        self.assertEqual(len(recovery), 35)
        self.assertEqual(sum(actions[i, 3] >= 0.5 for i in recovery), 10)
        self.assertEqual(sum(actions[i, 3] >= 0.5 for i in ordinary), 80)
        with self.assertRaises(ValueError):
            m20_lerobot_frame_indices(actions, frame_stride=2, recovery_terminal_stop_repeat=0)

    def test_terminal_stop_frames_can_be_rebalanced_without_changing_motion_frames(self) -> None:
        actions = np.zeros((8, 4), dtype=np.float32)
        actions[4:, 3] = 1.0

        indices = m20_lerobot_frame_indices(actions, frame_stride=2, terminal_stop_repeat=3)

        self.assertEqual(indices, [0, 2, 4, 4, 4, 6, 6, 6])

    def test_state_is_32d_and_excludes_world_xyz(self) -> None:
        proprio = np.arange(45, dtype=np.float32)
        lidar = np.arange(72, dtype=np.float32)
        state = m20_smolvla_state(proprio, lidar)
        self.assertEqual(state.shape, (32,))
        self.assertEqual(len(M20_STATE_NAMES), 32)
        np.testing.assert_array_equal(state[:20], proprio[3:23])
        np.testing.assert_array_equal(state[20:26], proprio[23:29])
        np.testing.assert_array_equal(
            state[26:], np.array([0, 12, 24, 36, 48, 60], dtype=np.float32)
        )

    def test_state_rejects_wrong_sensor_shape(self) -> None:
        with self.assertRaises(ValueError):
            m20_smolvla_state(np.zeros(44), np.zeros(72))


if __name__ == "__main__":
    unittest.main()

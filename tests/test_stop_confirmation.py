from collections import deque
from types import SimpleNamespace
import unittest
import numpy as np

from m20pro_vla.eval.stop_confirmation import ActionQueueFreshener, StopConfirmation
from m20pro_vla.low_level.shield import lidar_safety_shield
from m20pro_vla.eval.acceptance import apply_evaluation_config
from m20pro_vla.runtime.experiment import experiment_stage_args
import json
from pathlib import Path

GO = np.asarray((.2, 0., 0., 0.))
STOP = np.asarray((0., 0., 0., 1.))


class QueuedPolicy:
    def __init__(self, chunks):
        self._queues = {'action': deque(), 'observation.state': deque(['keep-history'])}
        self.chunks = iter(chunks)
        self.observations = []

    def select(self, observation):
        if not self._queues['action']:
            self.observations.append(observation)
            self._queues['action'].extend(next(self.chunks))
        return self._queues['action'].popleft().copy()


class FreshStopTest(unittest.TestCase):
    def tick(self, policy, freshener, stop, step):
        action, evidence = freshener.predict(policy, lambda: policy.select(step), confirming=stop.pending)
        command = stop.resolve(action, GO, visual_evidence=True, prediction_fresh=evidence.fresh)
        return command, evidence

    def test_queued_future_stop_is_cancelled_by_current_observation(self):
        policy = QueuedPolicy([[GO, STOP, STOP, STOP], [GO]*10])
        freshener = ActionQueueFreshener(True)
        stop = StopConfirmation(required_votes=3, require_fresh=True)
        self.tick(policy, freshener, stop, 0)
        _, candidate = self.tick(policy, freshener, stop, 1)
        self.assertFalse(candidate.fresh)
        self.assertEqual(stop.votes, 0)
        self.assertTrue(stop.pending)
        command, confirmation = self.tick(policy, freshener, stop, 2)
        np.testing.assert_array_equal(command, GO)
        self.assertTrue(confirmation.fresh)
        self.assertTrue(confirmation.forced_replan)
        self.assertFalse(stop.latched)
        self.assertFalse(stop.pending)
        self.assertEqual(policy.observations, [0, 2])
        self.assertEqual(list(policy._queues['observation.state']), ['keep-history'])

    def test_genuine_stop_needs_three_separate_observation_predictions(self):
        policy = QueuedPolicy([[STOP]*10, [STOP]*10, [STOP]*10])
        freshener = ActionQueueFreshener(True)
        stop = StopConfirmation(required_votes=3, require_fresh=True)
        generations = []
        for step in range(3):
            command, evidence = self.tick(policy, freshener, stop, step)
            self.assertTrue(evidence.fresh)
            generations.append(evidence.generation)
            self.assertEqual(stop.latched, step == 2)
        self.assertEqual(generations, [1, 2, 3])
        self.assertEqual(policy.observations, [0, 1, 2])
        np.testing.assert_array_equal(command, STOP)

    def test_visual_block_resets_candidate_and_votes(self):
        stop = StopConfirmation(required_votes=3, require_fresh=True)
        stop.resolve(STOP, GO, visual_evidence=True, prediction_fresh=True)
        command = stop.resolve(STOP, GO, visual_evidence=False, prediction_fresh=True)
        self.assertEqual(stop.votes, 0)
        self.assertFalse(stop.pending)
        np.testing.assert_array_equal(command, GO)

    def test_emergency_lidar_stop_does_not_wait_for_confirmation(self):
        stop = StopConfirmation(required_votes=3, require_fresh=True)
        command = stop.resolve(STOP, GO, visual_evidence=True, prediction_fresh=False)
        scan = np.full(72, 2.)
        scan[32:41] = .4
        safe, reason = lidar_safety_shield(command, scan, stop_distance=.5, slow_distance=1.25)
        np.testing.assert_array_equal(safe, STOP)
        self.assertEqual(reason, 'emergency_stop')
        self.assertFalse(stop.latched)

    def test_opt_out_keeps_legacy_queued_action_votes(self):
        policy = QueuedPolicy([[STOP]*10])
        freshener = ActionQueueFreshener(False)
        stop = StopConfirmation(required_votes=3, require_fresh=False)
        for step in range(3):
            self.tick(policy, freshener, stop, step)
        self.assertTrue(stop.latched)
        self.assertEqual(policy.observations, [0])

    def test_unknown_policy_queue_is_rejected_in_opt_in_mode(self):
        with self.assertRaises(TypeError):
            ActionQueueFreshener(True).predict(SimpleNamespace(), lambda: STOP, confirming=True)

    def test_unified_config_opt_in_and_explicit_cli_opt_out(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/experiment.json').read_text())
        cfg['smolvla_evaluation']['fresh_stop_confirmation'] = True
        self.assertIn('--fresh-stop-confirmation', experiment_stage_args(cfg, 'smolvla-eval'))
        args = SimpleNamespace(fresh_stop_confirmation=None)
        apply_evaluation_config(args, cfg)
        self.assertTrue(args.fresh_stop_confirmation)
        args.fresh_stop_confirmation = False
        apply_evaluation_config(args, cfg)
        self.assertFalse(args.fresh_stop_confirmation)


if __name__ == '__main__':
    unittest.main()

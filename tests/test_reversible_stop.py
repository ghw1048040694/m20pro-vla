from types import SimpleNamespace
import unittest
import numpy as np
from m20pro_vla.eval.stop_confirmation import StopConfirmation
from m20pro_vla.eval.acceptance import terminal_stop_step, apply_evaluation_config

STOP=np.array([0.,0.,0.,1.])
GO=np.array([.2,0.,0.,0.])
BACK=np.array([-.2,0.,.1,0.])

class ReversibleStopTest(unittest.TestCase):
    def resolve(self,c,a,previous=STOP):
        return c.resolve(a,previous,visual_evidence=True,prediction_fresh=False)
    def test_long_stop_then_move_releases_without_extra_vote(self):
        c=StopConfirmation(required_votes=1,require_fresh=False,reversible=True)
        for _ in range(100): np.testing.assert_array_equal(self.resolve(c,STOP),STOP)
        np.testing.assert_array_equal(self.resolve(c,GO),GO)
        self.assertFalse(c.latched)
        self.assertEqual((c.votes,c.pending),(0,False))
    def test_stop_reverse_stop_turn_sequence(self):
        c=StopConfirmation(required_votes=1,require_fresh=False,reversible=True)
        turn=np.array([0.,0.,.15,0.])
        for a in (GO,STOP,BACK,STOP,turn):
            np.testing.assert_array_equal(self.resolve(c,a),a)
        self.assertFalse(c.latched)
    def test_legacy_terminal_latch_remains_available(self):
        c=StopConfirmation(required_votes=1,require_fresh=False)
        self.resolve(c,STOP)
        np.testing.assert_array_equal(self.resolve(c,BACK),STOP)
        self.assertTrue(c.latched)
    def test_transient_pause_cannot_be_counted_as_terminal(self):
        self.assertEqual(terminal_stop_step(stop_step=10,stop_active=False,steps_executed=100,hold_steps=25,reversible=True),-1)
        self.assertEqual(terminal_stop_step(stop_step=-1,stop_active=False,steps_executed=100,hold_steps=25,reversible=True),-1)
    def test_final_hold_boundary_and_no_historical_stop_credit(self):
        self.assertEqual(terminal_stop_step(stop_step=10,stop_active=True,steps_executed=35,hold_steps=25,reversible=True),-1)
        self.assertEqual(terminal_stop_step(stop_step=10,stop_active=True,steps_executed=36,hold_steps=25,reversible=True),10)
        self.assertEqual(terminal_stop_step(stop_step=99,stop_active=True,steps_executed=100,hold_steps=25,reversible=True),-1)
    def test_config_and_explicit_override(self):
        cfg={'smolvla_evaluation':{'reversible_stop':True}}
        args=apply_evaluation_config(SimpleNamespace(reversible_stop=None),cfg)
        self.assertTrue(args.reversible_stop)
        args=apply_evaluation_config(SimpleNamespace(reversible_stop=False),cfg)
        self.assertFalse(args.reversible_stop)

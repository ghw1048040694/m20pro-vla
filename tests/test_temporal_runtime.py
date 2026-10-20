import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec
from m20pro_vla.runtime.temporal_policy import TemporalRuntime

class TemporalRuntimeTests(unittest.TestCase):
    def runtime(self):
        class QueueOnlyPolicy:
            model=SimpleNamespace(spec=TemporalRGBSpec())
            def reset(self):self._queues={'action':deque()}
        policy=QueueOnlyPolicy();policy.reset()
        return TemporalRuntime(policy,None,None,'cpu')
    def frames(self,value=0):return {cam:np.full((2,3,3),value,dtype=np.uint8) for cam in ('front','rear')}
    def test_explicit_bschw_mask_and_source_tick_history(self):
        runtime=self.runtime()
        for tick in range(151):runtime.observe(self.frames(tick%256),tick=tick,task='task',episode='a')
        batch=runtime.batch(np.zeros(32,dtype=np.float32));key='observation.images.front'
        self.assertEqual(tuple(batch[key].shape),(1,3,3,2,3))
        self.assertEqual(batch[key+'_is_pad'].tolist(),[[True,False,False]])
        self.assertEqual(round(float(batch[key][0,1,0,0,0])*255),50)
    def test_queued_actions_do_not_prevent_observation_and_new_task_resets(self):
        runtime=self.runtime();runtime.observe(self.frames(),tick=0,task='a',episode='a');runtime.policy._queues['action'].append(1)
        runtime.observe(self.frames(1),tick=1,task='a',episode='a');self.assertEqual(len(runtime.policy._queues['action']),1)
        runtime.observe(self.frames(2),tick=0,task='b',episode='a');self.assertEqual(len(runtime.policy._queues['action']),0)
        self.assertEqual(runtime.batch(np.zeros(32))['observation.images.front_is_pad'].tolist(),[[True,True,False]])
    def test_duplicate_skipped_or_nonzero_new_context_tick_rejected(self):
        runtime=self.runtime();runtime.observe(self.frames(),tick=0,task='a',episode='a')
        for tick,task in [(0,'a'),(2,'a'),(1,'b')]:
            with self.assertRaises(ValueError):runtime.observe(self.frames(),tick=tick,task=task,episode='a')
    def test_observation_before_inference_and_original_state_required(self):
        runtime=self.runtime()
        with self.assertRaises(ValueError):runtime.batch(np.zeros(32))
        runtime.observe(self.frames(),tick=0,task='a',episode='a')
        for state in (np.zeros(33),np.full(32,np.nan)):
            with self.assertRaises(ValueError):runtime.batch(state)

if __name__=='__main__':unittest.main()

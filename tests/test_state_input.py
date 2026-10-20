import unittest, tempfile, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from m20pro_vla.data.state_input import relative_wheel_spec,relative_wheel_state,load_state_input,save_state_input,WHEEL_INDICES
from m20pro_vla.runtime.temporal_policy import TemporalRuntime
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec

class RelativeWheelStateTests(unittest.TestCase):
    def runtime(self,new=True):
        policy=SimpleNamespace(model=SimpleNamespace(spec=TemporalRGBSpec()),reset=lambda:None)
        if new:policy.state_input=relative_wheel_spec()
        return TemporalRuntime(policy,None,None,'cpu')
    def frames(self):return {c:np.zeros((96,160,3),dtype=np.uint8) for c in ('front','rear')}
    def test_only_four_coordinates_and_no_mutation(self):
        a=np.arange(32,dtype=np.float32);b=a+2;before=b.copy();out=relative_wheel_state(b,a)
        others=[i for i in range(32) if i not in WHEEL_INDICES]
        self.assertTrue(np.array_equal(out[others],b[others]));self.assertTrue(np.array_equal(b,before))
        self.assertTrue(np.array_equal(out[list(WHEEL_INDICES)],np.full(4,2,dtype=np.float32)))
    def test_float32_translation_invariance(self):
        a=np.arange(32,dtype=np.float32);b=a+2;shift=np.zeros(32,dtype=np.float32);shift[list(WHEEL_INDICES)]=[128,-128,256,-256]
        self.assertTrue(np.array_equal(relative_wheel_state(b,a),relative_wheel_state(b+shift,a+shift)))
    def test_runtime_reference_persists_across_chunks(self):
        r=self.runtime();a=np.arange(32,dtype=np.float32)
        for tick in range(21):
            state=a+tick;r.observe(self.frames(),tick=tick,task='a',episode='1',state=state)
            out=r.batch(state)['observation.state'].numpy()[0]
            self.assertTrue(np.array_equal(out,relative_wheel_state(state,a)))
    def test_runtime_new_episode_and_task_reset(self):
        r=self.runtime();a=np.arange(32,dtype=np.float32)
        r.observe(self.frames(),tick=0,task='a',episode='1',state=a)
        r.observe(self.frames(),tick=0,task='b',episode='1',state=a+100)
        self.assertTrue(np.array_equal(r.batch(a+100)['observation.state'].numpy()[0,list(WHEEL_INDICES)],np.zeros(4)))
        with self.assertRaises(ValueError):r.observe(self.frames(),tick=1,task='c',episode='2',state=a)
    def test_runtime_requires_true_tick_zero_state(self):
        r=self.runtime()
        with self.assertRaises(ValueError):r.observe(self.frames(),tick=0,task='a',episode='1')
    def test_legacy_runtime_keeps_absolute_state(self):
        r=self.runtime(False);a=np.arange(32,dtype=np.float32);r.observe(self.frames(),tick=0,task='a',episode='1')
        self.assertTrue(np.array_equal(r.batch(a)['observation.state'].numpy()[0],a))
    def test_strict_saved_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(load_state_input(temp));save_state_input(temp,relative_wheel_spec());self.assertEqual(load_state_input(temp),relative_wheel_spec())
            (Path(temp)/'m20_state_input.json').write_text(json.dumps({'schema':'m20_unknown'}))
            with self.assertRaises(ValueError):load_state_input(temp)
    def test_nonfinite_and_wrong_shape_rejected(self):
        for state in [np.zeros(31),np.full(32,np.nan)]:
            with self.assertRaises(ValueError):relative_wheel_state(state,np.zeros(32))

if __name__=='__main__':unittest.main()

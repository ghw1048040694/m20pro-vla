import unittest
import numpy as np
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec, TemporalRGBBuffer, sample_temporal_rgb


class TemporalRGBTests(unittest.TestCase):
    def frames(self, count):
        a = np.broadcast_to((np.arange(count) % 251).astype(np.uint8)[:,None,None,None], (count,2,3,3)).copy()
        return {'front':a, 'rear':a.copy()}

    def test_online_equals_offline_through_long_occlusion_window(self):
        spec = TemporalRGBSpec()
        frames = self.frames(430)
        online = TemporalRGBBuffer(spec)
        for index in range(len(frames['front'])):
            got, valid = online.observe({key:value[index] for key,value in frames.items()})
            expected, wanted = sample_temporal_rgb(frames,index,spec)
            np.testing.assert_array_equal(valid,wanted)
            for key in spec.cameras: np.testing.assert_array_equal(got[key],expected[key])

    def test_future_change_cannot_change_sample(self):
        frames = self.frames(12)
        spec = TemporalRGBSpec((-5,-1,0))
        before = sample_temporal_rgb(frames,6,spec)
        for value in frames.values(): value[7:]=255
        after = sample_temporal_rgb(frames,6,spec)
        for key in spec.cameras: np.testing.assert_array_equal(before[0][key],after[0][key])

    def test_reset_masks_all_previous_episode_history(self):
        online = TemporalRGBBuffer(TemporalRGBSpec((-2,0)))
        frames = self.frames(4)
        for index in range(4): online.observe({key:value[index] for key,value in frames.items()})
        online.reset()
        current = {key:np.full((2,3,3),200,np.uint8) for key in frames}
        got, valid = online.observe(current)
        np.testing.assert_array_equal(valid,[False,True])
        self.assertTrue(all(np.all(x==200) for x in got.values()))

    def test_external_mutation_cannot_change_saved_history(self):
        online = TemporalRGBBuffer(TemporalRGBSpec((-1,0)))
        current = {key:np.zeros((2,3,3),np.uint8) for key in ('front','rear')}
        online.observe(current)
        for x in current.values():x[:]=255
        got,_=online.observe(current)
        self.assertTrue(np.all(got['front'][0]==0))

    def test_contract_round_trip_and_invalid_future_or_order(self):
        spec=TemporalRGBSpec()
        self.assertEqual(spec,TemporalRGBSpec.from_dict(spec.to_dict()))
        for offsets in ((-1,1,0),(0,-1),(-1,-1,0)):
            with self.assertRaises(ValueError):TemporalRGBSpec(offsets)

    def test_privileged_fields_and_non_rgb_are_rejected(self):
        frames=self.frames(2)
        with self.assertRaises(ValueError):sample_temporal_rgb(frames|{'target_xy':np.zeros((2,2))},1)
        with self.assertRaises(ValueError):sample_temporal_rgb({k:v.astype(float) for k,v in frames.items()},1)

    def test_sparse_history_can_miss_brief_visibility(self):
        frames={key:np.zeros((401,2,3,3),np.uint8) for key in ('front','rear')}
        frames['front'][120]=255
        sampled,valid=sample_temporal_rgb(frames,400)
        self.assertTrue(valid.all())
        self.assertTrue(np.all(sampled['front']==0))


if __name__=='__main__':unittest.main()

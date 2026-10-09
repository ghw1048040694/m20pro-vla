import unittest
from types import SimpleNamespace
import numpy as np
import torch
from m20pro_vla.data.raw_dataset import RawM20Dataset
from m20pro_vla.data.lerobot_adapter import m20_lerobot_frame_indices
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec,TemporalRGBBuffer,sample_temporal_rgb
from m20pro_vla.data.temporal_raw_candidate import TemporalRawDatasetCandidate


def tiny_raw():
    d=RawM20Dataset.__new__(RawM20Dataset);frames=[];records=[];count=520
    for ep in range(2):
        actions=np.zeros((count*2,4),dtype=np.float32);actions[200:204,3]=1;actions[-20:,3]=1
        indices=m20_lerobot_frame_indices(actions,frame_stride=2,terminal_stop_repeat=8)
        mapping=np.asarray(indices)//2
        rgb=np.arange(count,dtype=np.uint16)%256;front=np.broadcast_to(rgb[:,None,None,None],(count,2,3,3)).astype(np.uint8).copy();rear=255-front
        frames.append(dict(action=actions[indices],state=np.zeros((len(indices),32),np.float32),front=front,rear=rear,image_index=mapping));records.append(dict(frames=len(indices),identity='tiny-'+str(ep),episode_id=ep,task='same'))
    d.records=records;d.ends=np.cumsum([r['frames'] for r in records]);d.starts=np.r_[0,d.ends[:-1]];d.num_frames=int(d.ends[-1]);d.task_indices=[0,0];d.settings=dict(source_fps=50,frame_stride=2,terminal_stop_repeat=8,recovery_terminal_stop_repeat=1)
    d.delta_indices={'observation.images.front':[0],'observation.images.rear':[0],'observation.state':[0],'action':list(range(50))};d.meta=SimpleNamespace(fps=25,camera_keys=['observation.images.front','observation.images.rear'],features={k:{'shape':(32,) if k=='observation.state' else (4,)} for k in d.delta_indices});d._episode_arrays=lambda ep:frames[ep]
    return d,frames


class TemporalRawCandidateTests(unittest.TestCase):
    def test_online_offline_every_weighted_row_same_source_ticks(self):
        base,arrays=tiny_raw();wrapped=TemporalRawDatasetCandidate(base);online=[]
        buffer=TemporalRGBBuffer()
        for tick in range(len(arrays[0]['front'])):online.append(buffer.observe({k:arrays[0][k][tick] for k in ('front','rear')}))
        for row,tick in enumerate(arrays[0]['image_index']):
            item=wrapped[row];rgb,valid=online[int(tick)]
            for cam in ('front','rear'):
                key='observation.images.'+cam;self.assertTrue(np.array_equal((item[key].numpy().transpose(0,2,3,1)*255).round().astype(np.uint8),rgb[cam]));self.assertTrue(np.array_equal(~item[key+'_is_pad'].numpy(),valid))
            self.assertAlmostEqual(float(item['timestamp']),float(tick)/25,places=5)

    def test_state_actions_weights_unchanged_and_row_offsets_wrong_after_stop(self):
        base,arrays=tiny_raw();wrapped=TemporalRawDatasetCandidate(base);row=int(np.flatnonzero(arrays[0]['image_index']==150)[0]);item=wrapped[row];plain=base[row]
        self.assertTrue(torch.equal(item['action'],plain['action']));self.assertTrue(torch.equal(item['observation.state'],plain['observation.state']));self.assertEqual(len(base),len(wrapped));self.assertEqual(int(arrays[0]['image_index'][row-100]),64);self.assertEqual(150-100,50)
        expected=arrays[0]['front'][50];self.assertTrue(np.array_equal((item['observation.images.front'][1].numpy().transpose(1,2,0)*255).round().astype(np.uint8),expected))

    def test_episode_boundary_padding_and_no_cache_mutation(self):
        base,arrays=tiny_raw();wrapped=TemporalRawDatasetCandidate(base);second=wrapped[int(base.starts[1])];self.assertEqual(second['observation.images.front_is_pad'].tolist(),[True,True,False]);second['observation.images.rear'].fill_(0);self.assertTrue(np.array_equal(arrays[1]['rear'][0],np.full((2,3,3),255,np.uint8)));self.assertEqual(wrapped.sampling_contract()['image_index_domain'],'unweighted_strided_source_rgb_tick')

    def test_inconsistent_fps_or_row_rgb_history_rejected(self):
        base,_=tiny_raw();base.meta.fps=50
        with self.assertRaises(ValueError):TemporalRawDatasetCandidate(base)
        base,_=tiny_raw();base.delta_indices['observation.images.front']=[-400,-100,0]
        with self.assertRaises(ValueError):TemporalRawDatasetCandidate(base)

if __name__=='__main__':unittest.main()

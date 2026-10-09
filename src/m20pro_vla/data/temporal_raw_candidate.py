"""Isolated raw RGB history wrapper on source ticks, not stop-reweighted rows.

No official factory, working reader or runtime is modified. Current state/actions
and stop row weights are inherited; only RGB history and source timestamp change.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .temporal_rgb import TemporalRGBSpec


class TemporalRawDatasetCandidate(Dataset):
    def __init__(self, dataset, spec=TemporalRGBSpec()):
        keys=['observation.images.front','observation.images.rear']
        if spec.fps != 25 or len(spec.offsets) != 3 or dataset.meta.fps != spec.fps or list(dataset.meta.camera_keys) != keys:
            raise ValueError('Temporal RGB requires the same 25Hz front/rear source ticks')
        if dataset.meta.features['observation.state']['shape'] != (32,):
            raise ValueError('Original 32D state must remain')
        if any(dataset.delta_indices.get(k) not in (None,[0]) for k in keys):
            raise ValueError('Base RGB deltas must be current-only, without row-based history')
        self.dataset=dataset;self.spec=spec
        self.meta=dataset.meta

    @property
    def num_frames(self):return self.dataset.num_frames

    @property
    def num_episodes(self):return self.dataset.num_episodes

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        item=self.dataset[index]
        episode=int(item['episode_index']);local=int(index)-int(self.dataset.starts[episode])
        arrays=self.dataset._episode_arrays(episode)
        # Immutable cache mapping refers to unweighted strided source RGB frames.
        tick=int(arrays['image_index'][local]);query=tick+np.asarray(self.spec.offsets)
        valid=query>=0;clipped=np.maximum(query,0)
        if tick>=len(arrays['front']) or tick<0:raise ValueError('Source tick outside episode')
        for cam in self.spec.cameras:
            image=arrays[cam]
            if image.dtype!=np.uint8 or image.ndim!=4 or image.shape[-1]!=3:
                raise ValueError('Expected raw uint8 RGB source cache')
            if len(image)!=len(arrays['front']):raise ValueError('Camera source lengths differ')
            item['observation.images.'+cam]=torch.from_numpy(np.array(image[clipped].transpose(0,3,1,2),copy=True)).float()/255
            item['observation.images.'+cam+'_is_pad']=torch.from_numpy(~valid.copy())
        item['timestamp']=torch.tensor(tick/self.spec.fps,dtype=torch.float32)
        return item

    def sampling_contract(self):
        return dict(schema='m20_temporal_raw_sampling_candidate_v1',temporal_rgb=self.spec.to_dict(),
            image_index_domain='unweighted_strided_source_rgb_tick',
            state_action_index_domain='original_stop_reweighted_training_row',
            base_delta_indices={k:list(v) for k,v in self.dataset.delta_indices.items()},
            source_settings=dict(self.dataset.settings),
            source_identities=[dict(identity=r['identity'],episode_id=r['episode_id'],frames=r['frames']) for r in self.dataset.records])

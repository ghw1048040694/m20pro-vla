"""Experimental CPU raw-array reader. Not enabled in the unified training CLI.

Caches strided RGB once without video encoding. Training rows retain the same
stop repeats, proprioception projection and episode-local delta padding.
Raw RGB is not pixel-identical to the legacy lossy H264 decoding.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from importlib.metadata import version

import numpy as np
import torch
from torch.utils.data import Dataset

from .lerobot_adapter import _episode_pairs, m20_lerobot_frame_indices, m20_smolvla_state


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _encode_stats(stats):
    return {key: {name: np.asarray(value).tolist() for name, value in values.items()}
            for key, values in stats.items()}


def _decode_stats(stats):
    return {key: {name: np.asarray(value) for name, value in values.items()}
            for key, values in stats.items()}


def _prepare_episode(npz, metadata, cache_root, settings):
    from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
    identity = _sha(npz) + ':' + _sha(metadata)
    contract = dict(settings=settings, code_sha256=_sha(__file__),
                    adapter_sha256=_sha(Path(__file__).with_name('lerobot_adapter.py')),
                    lerobot_version=version('lerobot'))
    key = hashlib.sha256((identity + json.dumps(contract, sort_keys=True)).encode()).hexdigest()
    target = cache_root/key

    def read_receipt():
        receipt = json.loads((target/'receipt.json').read_text())
        if receipt['identity'] != identity or receipt['contract'] != contract:
            raise ValueError('Raw cache contract differs')
        for name, digest in receipt['artifacts'].items():
            if _sha(target/name) != digest:
                raise ValueError('Raw cache artifact changed: ' + name)
        return receipt | {'root': str(target)}

    if target.exists():
        return read_receipt()
    raw = json.loads(metadata.read_text())
    with tempfile.TemporaryDirectory(prefix='.raw-', dir=cache_root) as temp:
        temp = Path(temp)
        with np.load(npz, allow_pickle=False) as arrays:
            action = arrays['action'].astype(np.float32)
            state = m20_smolvla_state(arrays['proprio'], arrays['lidar'])
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError('Non-finite action/state')
            if len(action) != len(state):
                raise ValueError('Mismatched raw lengths')
            indices = np.asarray(m20_lerobot_frame_indices(action,
                frame_stride=settings['frame_stride'], terminal_stop_repeat=settings['terminal_stop_repeat'],
                recovery_terminal_stop_repeat=settings['recovery_terminal_stop_repeat'],
                collection_mode=raw.get('collection_mode', 'legacy')), dtype=np.int64)
            if not len(indices):
                raise ValueError('Empty raw episode')
            mapped = indices // settings['frame_stride']
            np.save(temp/'image_index.npy', mapped)
            np.save(temp/'action.npy', action[indices])
            np.save(temp/'state.npy', state[indices])
            stats = {key: get_feature_stats(value, axis=0, keepdims=False)
                     for key, value in [('action', action[indices]), ('observation.state', state[indices])]}
            image_shape = None
            for cam in ('front', 'rear'):
                image = arrays[cam+'_rgb']
                if image.dtype != np.uint8 or image.ndim != 4 or image.shape[-1] != 3 or len(image) != len(action):
                    raise ValueError('Expected aligned uint8 NHWC RGB')
                if image_shape is not None and image.shape[1:] != image_shape:
                    raise ValueError('Camera shapes differ')
                image_shape = image.shape[1:]
                image = image[::settings['frame_stride']]
                np.save(temp/(cam+'.npy'), image)
                sampled = image[mapped[sample_indices(len(mapped))]].transpose(0,3,1,2)
                values = get_feature_stats(sampled, axis=(0,2,3), keepdims=True)
                stats['observation.images.'+cam] = {
                    k: v if k=='count' else np.squeeze(v / 255.0, axis=0) for k,v in values.items()}
                del image, sampled
        if identity != _sha(npz) + ':' + _sha(metadata):
            raise ValueError('Raw source changed during preparation')
        receipt = dict(identity=identity, contract=contract, episode_id=raw['episode_id'],
            task=raw['task_text'], mode=raw.get('collection_mode', 'legacy'), frames=len(indices),
            image_shape=list(image_shape), stats=_encode_stats(stats),
            artifacts={p.name: _sha(p) for p in temp.glob('*.npy')})
        (temp/'receipt.json').write_text(json.dumps(receipt)+'\n')
        try:
            temp.rename(target)
        except OSError:
            # Another preparer published the same immutable content key.
            if not target.is_dir():
                raise
    return read_receipt()


class RawM20Dataset(Dataset):
    """Map-style reader; preparation is CPU-only and per-worker mappings bounded."""
    def __init__(self, source, cache_root, *, frame_stride=2, source_fps=50,
                 terminal_stop_repeat=8, recovery_terminal_stop_repeat=1,
                 delta_indices=None, max_open_episodes=4):
        from lerobot.datasets.compute_stats import aggregate_stats
        import pandas as pd
        if source_fps <= 0 or frame_stride <= 0 or source_fps % frame_stride:
            raise ValueError('Invalid frame rate/stride')
        if max_open_episodes < 1:
            raise ValueError('Mapping bound must be positive')
        self.source = Path(source); cache_root = Path(cache_root); cache_root.mkdir(parents=True, exist_ok=True)
        self.settings = dict(source_fps=source_fps, frame_stride=frame_stride,
            terminal_stop_repeat=terminal_stop_repeat, recovery_terminal_stop_repeat=recovery_terminal_stop_repeat)
        self.records = [_prepare_episode(a,b,cache_root,self.settings) for a,b in _episode_pairs(self.source)]
        ids = [r['episode_id'] for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate raw episode IDs')
        if any(r['image_shape'] != self.records[0]['image_shape'] for r in self.records):
            raise ValueError('Episode camera shapes differ')
        self.ends = np.cumsum([r['frames'] for r in self.records]); self.starts = np.r_[0,self.ends[:-1]]
        self.num_frames = int(self.ends[-1]); self.num_episodes = len(self.records)
        self.delta_indices = delta_indices or {}
        self.max_open_episodes = max_open_episodes; self._maps = OrderedDict(); self._pid = os.getpid()
        self.tasks = list(dict.fromkeys(r['task'] for r in self.records))
        self.task_indices = [self.tasks.index(r['task']) for r in self.records]
        features = {'action': {'dtype':'float32','shape':(4,)},
                    'observation.state': {'dtype':'float32','shape':(32,)}}
        for cam in ('front','rear'):
            features['observation.images.'+cam] = {'dtype':'image','shape':tuple(self.records[0]['image_shape']),
                'names':['height','width','channel']}
        if any(key not in features or not value or any(not isinstance(i,int) for i in value)
               for key,value in self.delta_indices.items()):
            raise ValueError('Invalid delta feature/indices')
        self.meta = SimpleNamespace(fps=source_fps//frame_stride, features=features, robot_type='m20pro',
            total_frames=self.num_frames, total_episodes=self.num_episodes, camera_keys=['observation.images.front','observation.images.rear'],
            stats=aggregate_stats([_decode_stats(r['stats']) for r in self.records]),
            episodes=pd.DataFrame({'dataset_from_index':self.starts,'dataset_to_index':self.ends}),
            tasks=pd.DataFrame({'task_index':range(len(self.tasks))},index=self.tasks))

    def __len__(self):
        return self.num_frames

    def _close_maps(self):
        for arrays in self._maps.values():
            for value in arrays.values():
                value._mmap.close()
        self._maps.clear()

    def __getstate__(self):
        state = self.__dict__.copy(); state['_maps'] = OrderedDict(); state['_pid'] = None
        return state

    def _episode_arrays(self, episode):
        if self._pid != os.getpid():
            self._close_maps(); self._pid = os.getpid()
        if episode not in self._maps:
            root = Path(self.records[episode]['root'])
            self._maps[episode] = {name: np.load(root/(name+'.npy'), mmap_mode='r', allow_pickle=False)
                for name in ('action','state','front','rear','image_index')}
            while len(self._maps)>self.max_open_episodes:
                _, arrays = self._maps.popitem(last=False)
                for value in arrays.values(): value._mmap.close()
        self._maps.move_to_end(episode)
        return self._maps[episode]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < self.num_frames:
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends,index,side='right'))
        local = index-int(self.starts[episode]); arrays = self._episode_arrays(episode)
        size = self.records[episode]['frames']
        item = dict(index=torch.tensor(index), episode_index=torch.tensor(episode), frame_index=torch.tensor(local),
            timestamp=torch.tensor(local/self.meta.fps,dtype=torch.float32), task_index=torch.tensor(self.task_indices[episode]),
            task=self.records[episode]['task'])
        for key in self.meta.features:
            offsets = self.delta_indices.get(key)
            query = local + np.asarray(offsets if offsets is not None else [0])
            clipped = np.clip(query,0,size-1)
            if key.startswith('observation.images.'):
                cam = key.rsplit('.',1)[1]
                data = arrays[cam][arrays['image_index'][clipped]].transpose(0,3,1,2)
                tensor = torch.from_numpy(np.array(data,copy=True)).float()/255
                if len(tensor)==1: tensor=tensor.squeeze(0)
            else:
                name = 'action' if key=='action' else 'state'
                tensor = torch.from_numpy(np.array(arrays[name][clipped],copy=True))
                if offsets is None: tensor=tensor.squeeze(0)
            item[key]=tensor
            if offsets is not None:
                item[key+'_is_pad']=torch.from_numpy((query<0)|(query>=size))
        return item

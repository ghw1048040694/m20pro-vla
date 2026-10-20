"""Isolated causal RGB sampling candidate; not wired to training or deployment."""
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TemporalRGBSpec:
    offsets: tuple[int, ...] = (-400, -100, 0)
    fps: int = 25
    cameras: tuple[str, ...] = ('front', 'rear')

    def __post_init__(self):
        if (not self.offsets or self.offsets[-1] != 0
                or any(type(x) is not int or x > 0 for x in self.offsets)
                or tuple(sorted(set(self.offsets))) != self.offsets):
            raise ValueError('Offsets must be unique increasing nonpositive integers ending in zero')
        if type(self.fps) is not int or self.fps <= 0 or self.cameras != ('front', 'rear'):
            raise ValueError('Expected positive integer fps and front/rear cameras')

    def to_dict(self):
        return dict(schema='m20_temporal_rgb_candidate_v1', offsets=list(self.offsets),
                    fps=self.fps, cameras=list(self.cameras))

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'schema', 'offsets', 'fps', 'cameras'} or value['schema'] != 'm20_temporal_rgb_candidate_v1':
            raise ValueError('Unknown temporal RGB contract')
        return cls(tuple(value['offsets']), value['fps'], tuple(value['cameras']))


def _validate_frames(frames, cameras):
    if set(frames) != set(cameras):
        raise ValueError('Only front and rear RGB inputs are allowed')
    shape = None
    for key in cameras:
        array = np.asarray(frames[key])
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or len(array) == 0:
            raise ValueError('RGB must be nonempty uint8 [T,H,W,3]')
        if shape is not None and array.shape != shape:
            raise ValueError('Camera shapes must match')
        shape = array.shape
    return shape


def sample_temporal_rgb(frames, index, spec=TemporalRGBSpec()):
    """Sample one episode only. Missing past frames repeat frame zero but are masked."""
    shape = _validate_frames(frames, spec.cameras)
    if type(index) is not int or not 0 <= index < shape[0]:
        raise ValueError('Index must be within this episode')
    query = index + np.asarray(spec.offsets)
    return ({key: np.array(frames[key][np.maximum(query, 0)], copy=True) for key in spec.cameras},
            query >= 0)


class TemporalRGBBuffer:
    """One append per 25-Hz policy observation, including while an action chunk is queued.

    Reset on episode or task change. No teacher event or simulator labels are accepted.
    Sparse historical frames do not guarantee remembering a brief target appearance.
    """
    def __init__(self, spec=TemporalRGBSpec()):
        self.spec = spec
        self.reset()

    def reset(self):
        self._frames = deque(maxlen=1-self.spec.offsets[0])
        self._first = None
        self._step = -1

    def observe(self, frames):
        wrapped = {key: np.asarray(value)[None] for key, value in frames.items()}
        _validate_frames(wrapped, self.spec.cameras)
        current = {key: np.array(frames[key], copy=True) for key in self.spec.cameras}
        if self._first is not None and any(current[key].shape != self._first[key].shape for key in self.spec.cameras):
            raise ValueError('Camera shape changed within episode')
        self._step += 1
        if self._first is None:
            self._first = current
        self._frames.append(current)
        valid = np.asarray([self._step+offset >= 0 for offset in self.spec.offsets])
        result = {key: np.stack([self._frames[offset-1][key] if ok else self._first[key]
                    for offset, ok in zip(self.spec.offsets, valid)]) for key in self.spec.cameras}
        return result, valid

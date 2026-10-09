"""Isolated RGB evidence bank; not connected to a learner or action selection.

All three colour slots are updated from observed RGB, independently of the task.
No discovery event, room identity, target position, or simulator state is used.
"""
from dataclasses import dataclass
import numpy as np

from m20pro_vla.data.temporal_rgb import _validate_frames
from m20pro_vla.data.visibility import TARGET_PIXEL_THRESHOLD, target_color_mask

COLOUR_SLOTS = ('red cube', 'green cylinder', 'yellow box')
CAMERAS = ('front', 'rear')


@dataclass(frozen=True)
class VisualEvidenceSpec:
    fps: int = 25

    def __post_init__(self):
        if type(self.fps) is not int or self.fps != 25:
            raise ValueError('This candidate requires the existing 25-Hz observation stream')

    def to_dict(self):
        return dict(schema='m20_visual_evidence_candidate_v1', fps=self.fps,
                    pixel_threshold=TARGET_PIXEL_THRESHOLD, colour_slots=list(COLOUR_SLOTS))

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'schema', 'fps', 'pixel_threshold', 'colour_slots'}:
            raise ValueError('Unexpected evidence contract fields')
        if (value['schema'] != 'm20_visual_evidence_candidate_v1'
                or value['pixel_threshold'] != TARGET_PIXEL_THRESHOLD
                or value['colour_slots'] != list(COLOUR_SLOTS)):
            raise ValueError('Unknown evidence contract or changed visibility criterion')
        return cls(value['fps'])


@dataclass
class VisualEvidenceSample:
    images: dict[str, np.ndarray]
    valid: np.ndarray
    age_ticks: np.ndarray
    # Slot names and ages describe observed colour evidence, not object locations.
    colour_slots: tuple[str, ...] = COLOUR_SLOTS


def sample_visual_evidence(frames, index, spec=VisualEvidenceSpec()):
    """Independent offline prefix query over one episode's 25-Hz RGB observations.

    Missing slots repeat current RGB and are masked. Age -1 means no evidence.
    >=80 pixels across front and rear is evidence, not proof of an object.
    """
    shape = _validate_frames(frames, CAMERAS)
    if type(index) is not int or not 0 <= index < shape[0]:
        raise ValueError('Index must be inside this episode')
    indices = []
    for label in COLOUR_SLOTS:
        counts = []
        for camera in CAMERAS:
            prefix = frames[camera][:index+1]
            # Flatten time and rows; colour predicates have no spatial dependency.
            mask = target_color_mask(prefix.reshape(-1, shape[2], 3), label)
            counts.append(mask.reshape(index+1, -1).sum(axis=1))
        visible = np.flatnonzero(sum(counts) >= TARGET_PIXEL_THRESHOLD)
        indices.append(int(visible[-1]) if len(visible) else -1)
    valid = np.asarray(indices) >= 0
    ages = np.where(valid, index - np.asarray(indices), -1)
    selected = np.where(valid, indices, index)
    return VisualEvidenceSample(
        {camera: np.array(frames[camera][selected], copy=True) for camera in CAMERAS},
        valid, ages)


class VisualEvidenceBuffer:
    """Keep the latest qualifying observed pair for each colour until reset.

    Observe on every policy tick, including while an action chunk is queued.
    Caller must reset on episode/task change. Ages do not expire; the learner
    would need explicit age conditioning and evidence validity, still unwired.
    """
    def __init__(self, spec=VisualEvidenceSpec()):
        self.spec = spec
        self.reset()

    def reset(self):
        self._step = -1
        self._shape = None
        self._saved = [None] * len(COLOUR_SLOTS)
        self._seen_at = np.full(len(COLOUR_SLOTS), -1, dtype=np.int64)

    def observe(self, frames):
        shape = _validate_frames({k: np.asarray(v)[None] for k, v in frames.items()}, CAMERAS)
        if self._shape is not None and shape[1:] != self._shape:
            raise ValueError('Camera shape changed inside episode')
        self._shape = shape[1:]
        self._step += 1
        for slot, label in enumerate(COLOUR_SLOTS):
            count = sum(int(target_color_mask(frames[camera], label).sum()) for camera in CAMERAS)
            if count >= TARGET_PIXEL_THRESHOLD:
                self._saved[slot] = {camera: np.array(frames[camera], copy=True) for camera in CAMERAS}
                self._seen_at[slot] = self._step
        valid = self._seen_at >= 0
        return VisualEvidenceSample(
            {camera: np.stack([self._saved[slot][camera] if ok else frames[camera]
                               for slot, ok in enumerate(valid)]) for camera in CAMERAS},
            valid.copy(), np.where(valid, self._step-self._seen_at, -1))

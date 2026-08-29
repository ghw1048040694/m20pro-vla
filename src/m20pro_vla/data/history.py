"""Learner-only short-history features for M20 MuJoCo VLA policies.

The features in this module are intentionally built only from inputs that are
available to the policy at runtime: front/rear policy RGB, the target label
implied by language, and the previous executed body command.  They do not use
target XY, object IDs, segmentation, or simulator geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .visibility import target_color_mask


HISTORY_FEATURE_LABELS = (
    "current_visible",
    "current_x_offset",
    "current_area",
    "current_rear_source",
    "seen_latch",
    "last_visible_decay",
    "recently_lost",
    "ema_x_offset",
    "ema_area",
    "peak_area",
    "area_delta_1",
    "area_delta_lag",
    "prev_forward_norm",
    "prev_lateral_norm",
    "prev_yaw_norm",
    "prev_stop",
)
HISTORY_FEATURE_DIM = len(HISTORY_FEATURE_LABELS)
DEFAULT_HISTORY_PIXEL_THRESHOLD = 80


def target_observation_features(
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    target_label: str,
    *,
    pixel_threshold: int = DEFAULT_HISTORY_PIXEL_THRESHOLD,
) -> np.ndarray:
    """Return current target-pixel observation as ``[visible, x, area, rear]``.

    The area scale matches the existing visual-geometry auxiliary head:
    logarithmic in target-color pixel count and normalized by policy image
    area.  ``x`` is the horizontal centroid offset in the camera that contains
    more target-color pixels.
    """
    front_mask = target_color_mask(front_rgb, target_label)
    rear_mask = target_color_mask(rear_rgb, target_label)
    front_count = int(front_mask.sum())
    rear_count = int(rear_mask.sum())
    total = front_count + rear_count
    if total < int(pixel_threshold):
        return np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    use_rear = rear_count > front_count
    mask = rear_mask if use_rear else front_mask
    ys, xs = np.nonzero(mask)
    del ys
    height, width = mask.shape
    x_offset = (float(xs.mean()) - 0.5 * float(width - 1)) / max(1.0, 0.5 * float(width))
    area = min(1.0, float(np.log1p(total) / np.log1p(width * height)))
    return np.asarray(
        (
            1.0,
            float(np.clip(x_offset, -1.0, 1.0)),
            float(np.clip(area, 0.0, 1.0)),
            1.0 if use_rear else 0.0,
        ),
        dtype=np.float32,
    )


def normalize_previous_action(action: np.ndarray | None) -> np.ndarray:
    """Normalize previous body command into a bounded learner-only feature."""
    if action is None:
        return np.zeros((4,), dtype=np.float32)
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size < 4:
        padded = np.zeros((4,), dtype=np.float32)
        padded[: values.size] = values
        values = padded
    return np.asarray(
        (
            float(np.clip(values[0] / 0.35, -1.0, 1.0)),
            float(np.clip(values[1] / 0.10, -1.0, 1.0)),
            float(np.clip(values[2] / 0.15, -1.0, 1.0)),
            1.0 if float(values[3]) >= 0.5 else 0.0,
        ),
        dtype=np.float32,
    )


@dataclass
class VisualHistoryTracker:
    """Online state for the short-history feature contract."""

    pixel_threshold: int = DEFAULT_HISTORY_PIXEL_THRESHOLD
    ema_alpha: float = 0.18
    last_visible_tau_steps: float = 80.0
    delta_lag: int = 8
    step: int = 0
    seen_latch: bool = False
    last_visible_step: int = -1
    ema_x_offset: float = 0.0
    ema_area: float = 0.0
    peak_area: float = 0.0
    previous_area: float = 0.0
    area_history: list[float] = field(default_factory=list)

    def observe(
        self,
        front_rgb: np.ndarray,
        rear_rgb: np.ndarray,
        target_label: str,
        *,
        previous_action: np.ndarray | None = None,
    ) -> np.ndarray:
        current = target_observation_features(
            front_rgb,
            rear_rgb,
            target_label,
            pixel_threshold=int(self.pixel_threshold),
        )
        visible = bool(current[0] >= 0.5)
        current_x = float(current[1])
        current_area = float(current[2])
        if visible:
            self.seen_latch = True
            self.last_visible_step = int(self.step)
            self.ema_x_offset = (
                (1.0 - float(self.ema_alpha)) * self.ema_x_offset
                + float(self.ema_alpha) * current_x
            )
            self.ema_area = (
                (1.0 - float(self.ema_alpha)) * self.ema_area
                + float(self.ema_alpha) * current_area
            )
            self.peak_area = max(current_area, 0.998 * self.peak_area)
        else:
            self.ema_area *= 0.985
            self.peak_area *= 0.992

        if self.last_visible_step >= 0:
            visible_age = max(0, int(self.step) - int(self.last_visible_step))
            last_visible_decay = math.exp(-float(visible_age) / max(1.0, float(self.last_visible_tau_steps)))
        else:
            visible_age = 10**9
            last_visible_decay = 0.0
        recently_lost = (
            float(last_visible_decay)
            if self.seen_latch and not visible and visible_age <= int(2 * self.last_visible_tau_steps)
            else 0.0
        )
        lag_index = max(0, len(self.area_history) - int(self.delta_lag))
        lag_area = self.area_history[lag_index] if self.area_history else 0.0
        deltas = (
            float(np.clip(current_area - self.previous_area, -1.0, 1.0)),
            float(np.clip(current_area - lag_area, -1.0, 1.0)),
        )
        previous_action_features = normalize_previous_action(previous_action)
        feature = np.asarray(
            (
                float(current[0]),
                float(current[1]),
                float(current[2]),
                float(current[3]),
                1.0 if self.seen_latch else 0.0,
                float(last_visible_decay),
                float(recently_lost),
                float(np.clip(self.ema_x_offset, -1.0, 1.0)),
                float(np.clip(self.ema_area, 0.0, 1.0)),
                float(np.clip(self.peak_area, 0.0, 1.0)),
                deltas[0],
                deltas[1],
                *previous_action_features.tolist(),
            ),
            dtype=np.float32,
        )
        self.previous_area = current_area
        self.area_history.append(current_area)
        self.step += 1
        return feature


def build_visual_history_feature_trace(
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    actions: np.ndarray,
    target_label: str,
    *,
    executed_actions: np.ndarray | None = None,
    pixel_threshold: int = DEFAULT_HISTORY_PIXEL_THRESHOLD,
) -> np.ndarray:
    """Build the online-equivalent history feature trace for an episode."""
    front = np.asarray(front_rgb)
    rear = np.asarray(rear_rgb)
    if front.shape != rear.shape or front.ndim != 4 or front.shape[-1] != 3:
        raise ValueError("front/rear RGB arrays must both have shape [T, H, W, 3]")
    label_actions = np.asarray(actions)
    previous_source = np.asarray(executed_actions) if executed_actions is not None else label_actions
    if previous_source.shape[0] < front.shape[0]:
        raise ValueError("previous action source must have at least T rows")
    tracker = VisualHistoryTracker(pixel_threshold=int(pixel_threshold))
    features = np.zeros((front.shape[0], HISTORY_FEATURE_DIM), dtype=np.float32)
    previous_action = np.zeros((4,), dtype=np.float32)
    for index in range(front.shape[0]):
        if index > 0:
            previous_action = previous_source[index - 1]
        features[index] = tracker.observe(
            front[index],
            rear[index],
            target_label,
            previous_action=previous_action,
        )
    return features

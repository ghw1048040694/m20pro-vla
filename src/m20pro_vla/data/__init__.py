"""Dataset schemas and conversion helpers."""

from .curation import curate_episode_view
from .distribution import M20MuJoCoDistributionThresholds, audit_m20_mujoco_dataset
from .history import (
    DEFAULT_HISTORY_PIXEL_THRESHOLD,
    HISTORY_FEATURE_DIM,
    HISTORY_FEATURE_LABELS,
    VisualHistoryTracker,
    build_visual_history_feature_trace,
    normalize_previous_action,
    target_observation_features,
)
from .lerobot_adapter import (
    M20_ACTION_NAMES,
    M20_STATE_NAMES,
    convert_m20_to_lerobot,
    m20_smolvla_state,
)
from .visibility import (
    TARGET_PIXEL_THRESHOLD,
    first_visual_target_visible_step,
    target_color_mask,
    target_pixel_count,
    visual_target_visible_trace,
)

__all__ = [
    "M20MuJoCoDistributionThresholds",
    "M20_ACTION_NAMES",
    "M20_STATE_NAMES",
    "DEFAULT_HISTORY_PIXEL_THRESHOLD",
    "HISTORY_FEATURE_DIM",
    "HISTORY_FEATURE_LABELS",
    "TARGET_PIXEL_THRESHOLD",
    "VisualHistoryTracker",
    "audit_m20_mujoco_dataset",
    "build_visual_history_feature_trace",
    "convert_m20_to_lerobot",
    "curate_episode_view",
    "first_visual_target_visible_step",
    "normalize_previous_action",
    "m20_smolvla_state",
    "target_color_mask",
    "target_observation_features",
    "target_pixel_count",
    "visual_target_visible_trace",
]

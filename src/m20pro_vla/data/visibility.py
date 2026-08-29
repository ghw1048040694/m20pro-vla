"""Visual target visibility labels for M20 MuJoCo VLA datasets.

These helpers operate on the exact low-resolution RGB streams fed to the
policy.  They intentionally differ from geometric line-of-sight labels:
line-of-sight can become true long before the target is large/clear enough in
the policy camera input.
"""

from __future__ import annotations

import numpy as np


# High-resolution M20 policies use 160x96 front/rear RGB.  A threshold of 5-20
# pixels made hidden targets count as "discovered" from tiny edge/specular
# fragments at episode start, which poisoned search->approach phase labels.
# 80 pixels is still far below the close/stop scale but large enough to require
# a stable target-color blob in the actual policy input.
TARGET_PIXEL_THRESHOLD = 80


def target_color_mask(rgb: np.ndarray, target_label: str) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb image must have shape [H, W, 3]")
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    label = str(target_label)
    if label == "red cube":
        return (red > 140) & (green < 95) & (blue < 95) & (red > green + 45) & (red > blue + 45)
    if label == "green cylinder":
        return (green > 110) & (red < 120) & (blue < 120) & (green > red + 35) & (green > blue + 35)
    if label == "yellow box":
        return (red > 135) & (green > 105) & (blue < 115) & (red > blue + 45) & (green > blue + 35)
    raise ValueError(f"unsupported target label for visibility mask: {target_label!r}")


def target_pixel_count(rgb: np.ndarray, target_label: str) -> int:
    return int(target_color_mask(rgb, target_label).sum())


def visual_target_visible_trace(
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    target_label: str,
    *,
    pixel_threshold: int = TARGET_PIXEL_THRESHOLD,
) -> np.ndarray:
    front = np.asarray(front_rgb)
    rear = np.asarray(rear_rgb)
    if front.shape != rear.shape or front.ndim != 4 or front.shape[-1] != 3:
        raise ValueError("front/rear RGB arrays must both have shape [T, H, W, 3]")
    visible = np.zeros((front.shape[0],), dtype=np.float32)
    for index in range(front.shape[0]):
        count = target_pixel_count(front[index], target_label) + target_pixel_count(rear[index], target_label)
        visible[index] = 1.0 if count >= int(pixel_threshold) else 0.0
    return visible


def first_visual_target_visible_step(
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    target_label: str,
    *,
    pixel_threshold: int = TARGET_PIXEL_THRESHOLD,
) -> int:
    trace = visual_target_visible_trace(front_rgb, rear_rgb, target_label, pixel_threshold=pixel_threshold)
    matches = np.flatnonzero(trace > 0.5)
    return int(matches[0]) if matches.size else -1

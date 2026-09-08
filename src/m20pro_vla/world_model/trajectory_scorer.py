"""Lightweight learned scorer for action-chunk selection.

The scorer is not a full latent dynamics model. It is a compact learned
world-model surrogate that scores candidate body-command chunks from the
current multimodal latent and prefers chunks that are likely to make
progress, discover the target, and finish safely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


SCORER_HEAD_ORDER = (
    "score",
    "visible",
    "reach",
    "safe",
    "progress",
    "final_close",
    "visibility_keep",
    "smooth",
)
SCORE_HEAD_INDEX = 0
VISIBLE_HEAD_INDEX = 1
REACH_HEAD_INDEX = 2
SAFE_HEAD_INDEX = 3
PROGRESS_HEAD_INDEX = 4
FINAL_CLOSE_HEAD_INDEX = 5
VISIBILITY_KEEP_HEAD_INDEX = 6
SMOOTH_HEAD_INDEX = 7
SCORER_HEAD_COUNT = len(SCORER_HEAD_ORDER)
LEGACY_SCORER_HEAD_ORDER = ("score", "visible", "reach", "safe")


def visual_goal_distance_from_pixels(
    pixel_count: int | float,
    *,
    close_pixel_threshold: int | float,
) -> float:
    """Conservative target-distance hint from learner-only RGB target area.

    This is intentionally only a proposal hint, not a success signal.  The
    close-pixel threshold used by playback means "large enough to trust visual
    closeness"; it does not mean the robot is already at the final stop pose.
    A count near that threshold maps to roughly the task success radius, and
    the distance shrinks with inverse square-root image area.
    """
    count = max(1.0, float(pixel_count))
    close = max(1.0, float(close_pixel_threshold))
    ratio = max(1.0e-4, count / close)
    return float(np.clip(0.70 / np.sqrt(ratio), 0.28, 2.40))


def visual_goal_visible_prob_from_pixels(
    pixel_count: int | float,
    *,
    visible_pixel_threshold: int | float,
    close_pixel_threshold: int | float,
) -> float:
    """Soft confidence that target-color pixels are useful for proposals."""
    count = max(0.0, float(pixel_count))
    visible = max(1.0, float(visible_pixel_threshold))
    close = max(visible + 1.0, float(close_pixel_threshold))
    if count < visible:
        return 0.0
    ratio = (count - visible) / max(1.0, close - visible)
    return float(np.clip(0.55 + 0.45 * ratio, 0.55, 1.0))


def _clip_command(command: np.ndarray) -> np.ndarray:
    clipped = np.asarray(command, dtype=np.float32).copy()
    clipped[0] = float(np.clip(clipped[0], 0.0, 0.35))
    clipped[1] = float(np.clip(clipped[1], -0.10, 0.10))
    clipped[2] = float(np.clip(clipped[2], -0.15, 0.15))
    clipped[3] = float(np.clip(clipped[3], 0.0, 1.0))
    return clipped


def make_repeated_action_chunks(
    base_command: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Build a compact candidate set around one proposed body command.

    The candidates are still learner-only at inference time: the scorer sees
    the policy latent plus candidate body commands, not target XY or simulator
    state.  The set intentionally includes slower arcs and stop-tail variants
    so the world model can prefer recoverable motion over the nearest/fastest
    command.
    """
    command = _clip_command(base_command)
    fwd = float(command[0])
    lat = float(command[1])
    yaw = float(command[2])
    stop = float(command[3])

    def constant_chunk(variant: tuple[float, float, float, float]) -> np.ndarray:
        return np.repeat(np.asarray(variant, dtype=np.float32)[None, :], int(horizon), axis=0)

    def ramped_chunk(variant: tuple[float, float, float, float], *, start_scale: float, end_scale: float) -> np.ndarray:
        chunk = constant_chunk(variant)
        scales = np.linspace(float(start_scale), float(end_scale), int(horizon), dtype=np.float32)
        chunk[:, :3] *= scales[:, None]
        return chunk

    def stop_tail_chunk(variant: tuple[float, float, float, float]) -> np.ndarray:
        chunk = constant_chunk(variant)
        tail = max(2, int(horizon) // 3)
        fade = np.linspace(1.0, 0.0, tail, dtype=np.float32)
        chunk[-tail:, :3] *= fade[:, None]
        chunk[-tail:, 3] = 1.0
        return chunk

    variants = [
        (fwd, lat, yaw, 0.0),
        (float(np.clip(fwd * 1.15 + 0.015, 0.0, 0.35)), lat, yaw, 0.0),
        (float(np.clip(fwd * 0.75, 0.0, 0.35)), lat, yaw, 0.0),
        (float(np.clip(fwd * 0.50, 0.0, 0.35)), lat, yaw, 0.0),
        (fwd, lat, float(np.clip(yaw + 0.08, -0.15, 0.15)), 0.0),
        (fwd, lat, float(np.clip(yaw - 0.08, -0.15, 0.15)), 0.0),
        (float(np.clip(fwd * 0.65, 0.0, 0.35)), lat, float(np.clip(yaw + 0.15, -0.15, 0.15)), 0.0),
        (float(np.clip(fwd * 0.65, 0.0, 0.35)), lat, float(np.clip(yaw - 0.15, -0.15, 0.15)), 0.0),
        (float(np.clip(fwd * 0.65, 0.0, 0.35)), float(np.clip(lat + 0.06, -0.10, 0.10)), yaw, 0.0),
        (float(np.clip(fwd * 0.65, 0.0, 0.35)), float(np.clip(lat - 0.06, -0.10, 0.10)), yaw, 0.0),
        (0.08, 0.0, 0.15, 0.0),
        (0.08, 0.0, -0.15, 0.0),
        (0.0, 0.0, 0.15, 0.0),
        (0.0, 0.0, -0.15, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ]
    chunks = [constant_chunk(variant) for variant in variants]
    chunks.append(ramped_chunk((fwd, lat, yaw, 0.0), start_scale=0.35, end_scale=1.0))
    chunks.append(ramped_chunk((fwd, lat, yaw, 0.0), start_scale=1.0, end_scale=0.45))
    chunks.append(stop_tail_chunk((float(np.clip(fwd * 0.45, 0.0, 0.35)), lat, yaw, 0.0)))
    return np.stack(chunks, axis=0).astype(np.float32, copy=False)


def _constant_action_chunk(
    variant: tuple[float, float, float, float],
    *,
    horizon: int,
) -> np.ndarray:
    return np.repeat(np.asarray(variant, dtype=np.float32)[None, :], int(horizon), axis=0)


def _ramped_action_chunk(
    variant: tuple[float, float, float, float],
    *,
    horizon: int,
    start_scale: float,
    end_scale: float,
) -> np.ndarray:
    chunk = _constant_action_chunk(variant, horizon=horizon)
    scales = np.linspace(float(start_scale), float(end_scale), int(horizon), dtype=np.float32)
    chunk[:, :3] *= scales[:, None]
    return chunk


def _turn_then_drive_chunk(
    *,
    forward: float,
    yaw: float,
    horizon: int,
    turn_steps: int,
    drive_scale: float = 1.0,
) -> np.ndarray:
    horizon = int(horizon)
    turn_steps = int(np.clip(turn_steps, 1, max(1, horizon - 1)))
    chunk = np.zeros((horizon, 4), dtype=np.float32)
    chunk[:turn_steps, 2] = float(np.clip(yaw, -0.15, 0.15))
    chunk[turn_steps:, 0] = float(np.clip(forward * drive_scale, 0.0, 0.35))
    chunk[turn_steps:, 2] = float(np.clip(0.55 * yaw, -0.15, 0.15))
    return chunk


def _deduplicate_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for chunk in chunks:
        rounded = np.round(np.asarray(chunk, dtype=np.float32), 4)
        key = rounded.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(rounded.astype(np.float32, copy=False))
    if not unique:
        raise ValueError("candidate action chunk set is empty")
    return np.stack(unique, axis=0).astype(np.float32, copy=False)


def make_phase_action_chunks(
    base_command: np.ndarray,
    *,
    horizon: int,
    phase_prob: np.ndarray | list[float] | tuple[float, ...] | None = None,
    goal_bearing: float | None = None,
    goal_distance: float | None = None,
    goal_visible_prob: float | None = None,
) -> np.ndarray:
    """Build learner-only action proposals for world-model selection.

    The earlier proposal set was too local: if the VLA output was weak after
    discovery, the scorer could only choose weak variants.  This pool keeps the
    direct VLA-centered candidates but adds phase-conditioned search arcs and
    stronger approach motions.  It still does not use target XY, object ID,
    semantic masks, or simulator state.
    """
    command = _clip_command(base_command)
    fwd = float(command[0])
    lat = float(command[1])
    yaw = float(command[2])
    stop = float(command[3])
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    phase = np.asarray(phase_prob if phase_prob is not None else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), dtype=np.float32)
    if phase.shape[0] < 3 or not np.isfinite(phase).all():
        phase = np.asarray((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), dtype=np.float32)
    search_prob, approach_prob, stop_prob = (float(value) for value in phase[:3])
    goal_hint_active = goal_bearing is not None and np.isfinite(float(goal_bearing))

    chunks = [chunk for chunk in make_repeated_action_chunks(command, horizon=horizon)]

    # Search proposals: exploratory arcs and turn-then-drive primitives.
    # These are useful before the target is visible and for escaping obstacle
    # occlusion.  They deliberately include both yaw signs; visual-language
    # latent + world-model outcome heads decide which is plausible.
    search_active = search_prob >= 0.25 or (not goal_hint_active and approach_prob < 0.45)
    if search_active:
        search_forward = (0.08, 0.14, 0.20) if approach_prob < 0.45 else (0.08, 0.16)
        search_yaws = (-0.15, 0.15) if approach_prob >= 0.45 else (-0.15, -0.10, 0.10, 0.15)
        if abs(yaw) >= 0.035:
            search_yaws = tuple(dict.fromkeys((float(np.clip(yaw, -0.15, 0.15)), *search_yaws)))
        for speed in search_forward:
            for yaw_value in search_yaws:
                chunks.append(_constant_action_chunk((speed, 0.0, yaw_value, 0.0), horizon=horizon))
        for yaw_value in (-0.15, 0.15):
            chunks.append(_constant_action_chunk((0.0, 0.0, yaw_value, 0.0), horizon=horizon))
            chunks.append(_turn_then_drive_chunk(forward=0.18, yaw=yaw_value, horizon=horizon, turn_steps=max(2, horizon // 5)))
            if approach_prob < 0.45:
                chunks.append(_turn_then_drive_chunk(forward=0.22, yaw=yaw_value, horizon=horizon, turn_steps=max(3, horizon // 4)))

    # Approach proposals: when the policy phase head thinks the target is
    # visible/approachable, allow the scorer to choose stronger forward motion
    # than the direct policy head emits.  Direct yaw remains a first-class
    # anchor; symmetric yaw alternatives cover weak or noisy yaw heads.
    yaw_anchor = float(np.clip(yaw, -0.15, 0.15))
    approach_active = approach_prob >= 0.20 or goal_hint_active
    if approach_active:
        approach_multiplier = 1.10 if approach_prob >= search_prob else 0.85
        if goal_hint_active:
            approach_forward = (
                float(np.clip(max(fwd, 0.10) * approach_multiplier, 0.08, 0.22)),
                0.16,
                0.22,
            )
            approach_yaws = (
                yaw_anchor,
                float(np.clip(yaw_anchor + 0.05, -0.15, 0.15)),
                float(np.clip(yaw_anchor - 0.05, -0.15, 0.15)),
                0.0,
            )
            slow_tail_speeds = (0.08, 0.12)
        else:
            approach_forward = (
                float(np.clip(max(fwd, 0.10) * approach_multiplier, 0.08, 0.24)),
                0.14,
                0.20,
            )
            approach_yaws = (
                yaw_anchor,
                float(np.clip(yaw_anchor + 0.06, -0.15, 0.15)),
                float(np.clip(yaw_anchor - 0.06, -0.15, 0.15)),
                0.0,
            )
            slow_tail_speeds = (0.08,)
        for speed in approach_forward:
            for yaw_value in approach_yaws:
                chunks.append(_constant_action_chunk((speed, 0.0, yaw_value, 0.0), horizon=horizon))
        for speed in slow_tail_speeds:
            slow_tail = _constant_action_chunk((speed, 0.0, yaw_anchor, 0.0), horizon=horizon)
            tail = max(2, horizon // 4)
            slow_tail[-tail:, 0] = np.linspace(speed, 0.0, tail, dtype=np.float32)
            slow_tail[-tail:, 3] = 1.0
            chunks.append(slow_tail)

    if stop_prob >= 0.35 or stop >= 0.5:
        chunks.append(_constant_action_chunk((0.0, 0.0, 0.0, 1.0), horizon=horizon))

    # Goal-belief proposals: a learned latent world-model head can predict
    # relative bearing/distance from policy observations.  These proposals are
    # still learner-only at inference time; the labels used to train the belief
    # head are privileged offline supervision, not runtime inputs.
    if goal_hint_active:
        bearing = float(np.clip(float(goal_bearing), -np.pi, np.pi))
        distance = float(goal_distance) if goal_distance is not None and np.isfinite(float(goal_distance)) else 1.0
        visible_boost = float(goal_visible_prob) if goal_visible_prob is not None and np.isfinite(float(goal_visible_prob)) else approach_prob
        yaw_goal = float(np.clip(0.60 * bearing, -0.15, 0.15))
        if distance <= 0.45 and visible_boost >= 0.45:
            chunks.append(_constant_action_chunk((0.0, 0.0, 0.0, 1.0), horizon=horizon))
        if distance <= 0.45:
            speed_goal = 0.04 if abs(bearing) <= 0.20 else 0.0
        elif distance <= 0.75:
            speed_goal = 0.08 if abs(bearing) <= 0.35 else 0.0
        elif distance > 1.80:
            speed_goal = 0.24 if abs(bearing) <= 0.45 else (0.16 if abs(bearing) <= 0.75 else 0.06)
        elif distance > 1.20:
            speed_goal = 0.18 if abs(bearing) <= 0.55 else (0.10 if abs(bearing) <= 0.75 else 0.04)
        elif abs(bearing) > 0.75:
            speed_goal = 0.04
        elif abs(bearing) > 0.40:
            speed_goal = 0.10
        else:
            speed_goal = 0.18 if distance <= 1.50 else 0.22
        for scale in (0.70, 1.00, 1.25):
            speed = float(np.clip(speed_goal * scale, 0.0, 0.26))
            chunks.append(_constant_action_chunk((speed, 0.0, yaw_goal, 0.0), horizon=horizon))
            chunks.append(_constant_action_chunk((speed, 0.0, float(np.clip(yaw_goal + 0.04, -0.15, 0.15)), 0.0), horizon=horizon))
            chunks.append(_constant_action_chunk((speed, 0.0, float(np.clip(yaw_goal - 0.04, -0.15, 0.15)), 0.0), horizon=horizon))
        if distance > 0.90:
            if distance > 1.80:
                minimum_commit_speed = 0.24 if abs(bearing) <= 0.55 else 0.14
            elif distance > 1.20:
                minimum_commit_speed = 0.20 if abs(bearing) <= 0.55 else 0.10
            else:
                minimum_commit_speed = 0.18 if abs(bearing) <= 0.55 else 0.10
            committed_speed = float(np.clip(max(speed_goal, minimum_commit_speed) * 1.15, 0.0, 0.30))
            chunks.append(_constant_action_chunk((committed_speed, 0.0, yaw_goal, 0.0), horizon=horizon))
            chunks.append(
                _ramped_action_chunk(
                    (committed_speed, 0.0, yaw_goal, 0.0),
                    horizon=horizon,
                    start_scale=0.60,
                    end_scale=1.10,
                )
            )
            chunks.append(
                _ramped_action_chunk(
                    (committed_speed, 0.0, yaw_goal, 0.0),
                    horizon=horizon,
                    start_scale=1.10,
                    end_scale=0.55,
                )
            )
        if abs(bearing) > 0.35:
            chunks.append(
                _turn_then_drive_chunk(
                    forward=max(0.08, speed_goal),
                    yaw=yaw_goal,
                    horizon=horizon,
                    turn_steps=max(2, horizon // 4),
                )
            )
            if distance > 0.90:
                chunks.append(
                    _turn_then_drive_chunk(
                        forward=min(0.30, max(0.12, committed_speed)),
                        yaw=yaw_goal,
                        horizon=horizon,
                        turn_steps=max(2, horizon // 5),
                        drive_scale=1.10,
                    )
                )
        if distance <= 0.55 and visible_boost >= 0.35:
            tail = _constant_action_chunk((max(0.04, min(0.10, speed_goal)), 0.0, yaw_goal, 0.0), horizon=horizon)
            stop_tail = max(2, horizon // 3)
            tail[-stop_tail:, 0] = np.linspace(float(tail[-stop_tail, 0]), 0.0, stop_tail, dtype=np.float32)
            tail[-stop_tail:, 3] = 1.0
            chunks.append(tail)

    return _deduplicate_chunks(chunks)


def scorer_combined_score(head_output: torch.Tensor) -> torch.Tensor:
    """Map head logits to a single scalar preference score.

    Legacy checkpoints output ``[score, visible, reach, safe]``.  New outcome
    checkpoints add progress, terminal closeness, visibility retention, and
    smoothness.  The combined score is used only to rank candidate action
    chunks; stop permission is still handled by an explicit gate in playback.
    """
    if head_output.ndim != 2:
        raise ValueError("trajectory scorer head must have shape [B, heads]")
    head_dim = int(head_output.size(-1))
    if head_dim == len(LEGACY_SCORER_HEAD_ORDER):
        return (
            torch.sigmoid(head_output[:, SCORE_HEAD_INDEX])
            + 0.20 * torch.sigmoid(head_output[:, VISIBLE_HEAD_INDEX])
            + 0.55 * torch.sigmoid(head_output[:, REACH_HEAD_INDEX])
            + 0.20 * torch.sigmoid(head_output[:, SAFE_HEAD_INDEX])
        )
    if head_dim < SCORER_HEAD_COUNT:
        raise ValueError(
            f"trajectory scorer head must output either {len(LEGACY_SCORER_HEAD_ORDER)} legacy heads "
            f"or at least {SCORER_HEAD_COUNT} outcome heads; got {head_dim}"
        )
    probs = torch.sigmoid(head_output)
    return (
        0.30 * probs[:, SCORE_HEAD_INDEX]
        + 0.06 * probs[:, VISIBLE_HEAD_INDEX]
        + 0.48 * probs[:, REACH_HEAD_INDEX]
        + 0.28 * probs[:, SAFE_HEAD_INDEX]
        + 1.10 * probs[:, PROGRESS_HEAD_INDEX]
        + 1.25 * probs[:, FINAL_CLOSE_HEAD_INDEX]
        + 0.34 * probs[:, VISIBILITY_KEEP_HEAD_INDEX]
        + 0.18 * probs[:, SMOOTH_HEAD_INDEX]
        + 0.20 * probs[:, PROGRESS_HEAD_INDEX] * probs[:, FINAL_CLOSE_HEAD_INDEX]
        + 0.10 * probs[:, SAFE_HEAD_INDEX] * probs[:, SMOOTH_HEAD_INDEX]
    )


def visual_goal_alignment_prior(
    action_chunks: torch.Tensor,
    *,
    goal_bearing: float | None,
    goal_distance: float | None,
    goal_visible_prob: float | None = None,
) -> torch.Tensor:
    """Learner-only prior for ranking chunks when policy RGB has a weak target cue.

    This is deliberately not a success signal and does not use target XY.  It
    only says that, when the exact policy RGB contains target-color pixels, a
    candidate whose first body command turns toward that image bearing and keeps
    moving until the target is visually close should receive a small ranking
    preference.  The learned scorer heads still provide the main ranking.
    """
    if action_chunks.ndim != 3 or action_chunks.size(-1) < 4:
        raise ValueError("action_chunks must have shape [B, T, 4]")
    if goal_bearing is None or not np.isfinite(float(goal_bearing)):
        return torch.zeros((action_chunks.shape[0],), dtype=action_chunks.dtype, device=action_chunks.device)
    distance = float(goal_distance) if goal_distance is not None and np.isfinite(float(goal_distance)) else 1.0
    visible_prob = (
        float(goal_visible_prob)
        if goal_visible_prob is not None and np.isfinite(float(goal_visible_prob))
        else 0.55
    )
    visible_weight = float(np.clip(visible_prob, 0.0, 1.0))
    bearing = float(np.clip(float(goal_bearing), -np.pi, np.pi))
    yaw_goal = float(np.clip(0.60 * bearing, -0.15, 0.15))
    first = action_chunks[:, 0, :]
    forward = first[:, 0].clamp(0.0, 0.35)
    yaw = first[:, 2].clamp(-0.15, 0.15)
    stop_any = (action_chunks[:, :, 3].amax(dim=1) >= 0.5).to(action_chunks.dtype)

    yaw_error = torch.abs(yaw - yaw.new_tensor(yaw_goal))
    yaw_score = (1.0 - yaw_error / 0.15).clamp(0.0, 1.0)
    turn_required = min(1.0, abs(bearing) / 0.75)

    if distance <= 0.45 and visible_weight >= 0.50:
        desired_forward = 0.0
        stop_score = stop_any
    elif distance <= 0.75:
        desired_forward = 0.08 if abs(bearing) <= 0.35 else 0.0
        stop_score = 1.0 - stop_any
    elif distance <= 1.20:
        desired_forward = 0.12 if abs(bearing) <= 0.55 else 0.08
        stop_score = 1.0 - stop_any
    elif distance <= 1.80:
        desired_forward = 0.20 if abs(bearing) <= 0.45 else (0.12 if abs(bearing) <= 0.75 else 0.08)
        stop_score = 1.0 - stop_any
    elif abs(bearing) > 0.75:
        desired_forward = 0.04
        stop_score = 1.0 - stop_any
    elif abs(bearing) > 0.40:
        desired_forward = 0.10
        stop_score = 1.0 - stop_any
    else:
        desired_forward = 0.22 if distance <= 1.50 else 0.26
        stop_score = 1.0 - stop_any

    desired = forward.new_tensor(float(desired_forward))
    tolerance = max(0.05, float(desired_forward) * 0.75)
    forward_score = (1.0 - torch.abs(forward - desired) / tolerance).clamp(0.0, 1.0)
    # When the object is far to the side, a turn-in-place / very slow turn is
    # acceptable.  Without this term the prior over-prefers driving while the
    # target is still on the image edge.
    turn_score = yaw_score * (1.0 - forward / 0.25).clamp(0.0, 1.0)
    approach_score = (1.0 - turn_required) * forward_score + turn_required * turn_score
    prior = 0.50 * yaw_score + 0.34 * approach_score + 0.16 * stop_score
    return (visible_weight * prior).clamp(0.0, 1.0)


class M20TrajectoryScorer(nn.Module):
    """Compact learned score model for candidate action chunks."""

    def __init__(
        self,
        latent_dim: int = 192,
        horizon: int = 24,
        action_dim: int = 4,
        head_dim: int = SCORER_HEAD_COUNT,
        belief_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.head_dim = int(head_dim)
        self.belief_dim = int(belief_dim)
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.belief_dim < 0:
            raise ValueError("belief_dim must be non-negative")
        action_chunk_dim = self.horizon * self.action_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(action_chunk_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, self.head_dim),
        )
        self.belief_head = (
            nn.Sequential(
                nn.Linear(self.latent_dim, 96),
                nn.LayerNorm(96),
                nn.GELU(),
                nn.Linear(96, self.belief_dim),
            )
            if self.belief_dim > 0
            else None
        )

    def forward(self, latent: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.size(-1) != self.latent_dim:
            raise ValueError(f"latent must have shape [B, {self.latent_dim}]")
        if action_chunks.ndim != 3 or action_chunks.size(1) != self.horizon or action_chunks.size(2) != self.action_dim:
            raise ValueError(
                f"action_chunks must have shape [B, {self.horizon}, {self.action_dim}]"
            )
        action_flat = action_chunks.reshape(action_chunks.size(0), -1)
        fused = torch.cat((latent, self.action_encoder(action_flat)), dim=-1)
        return self.head(fused)

    def predict_belief(self, latent: torch.Tensor) -> torch.Tensor:
        if self.belief_head is None:
            raise RuntimeError("this trajectory scorer checkpoint has no goal-belief head")
        if latent.ndim != 2 or latent.size(-1) != self.latent_dim:
            raise ValueError(f"latent must have shape [B, {self.latent_dim}]")
        return self.belief_head(latent)

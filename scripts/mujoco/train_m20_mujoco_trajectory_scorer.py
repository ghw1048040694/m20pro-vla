#!/usr/bin/env python3
"""Train a lightweight world-model scorer for M20 VLA action chunks."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from m20pro_vla.low_level import M20LowLevelController, M20LowLevelControllerState
from m20pro_vla.policies import M20MuJoCoVLA, encode_text
from m20pro_vla.sim.mujoco import (
    ObjectSpec,
    ObstacleSpec,
    SceneLightSpec,
    TerrainSpec,
    build_scene,
    obstacle_blocks_segment,
)
from m20pro_vla.data import (
    DEFAULT_HISTORY_PIXEL_THRESHOLD,
    TARGET_PIXEL_THRESHOLD,
    build_visual_history_feature_trace,
    target_color_mask,
    target_pixel_count,
    visual_target_visible_trace,
)
from m20pro_vla.world_model import (
    M20TrajectoryScorer,
    make_phase_action_chunks,
    visual_goal_distance_from_pixels,
    visual_goal_visible_prob_from_pixels,
)
from m20pro_vla.world_model.trajectory_scorer import SCORER_HEAD_ORDER


TERRAIN_WEIGHT_MULTIPLIER = {
    "flat": 1.0,
    "slope": 1.20,
    "bumps": 1.35,
    "step": 1.50,
}
STANDOFF_REACH_RADIUS_M = 0.45
STANDOFF_CLOSE_DECAY_M = 0.55


@dataclass(frozen=True)
class ReplayState:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray | None
    qacc_warmstart: np.ndarray | None
    time: float
    controller_state: M20LowLevelControllerState


@dataclass(frozen=True)
class EpisodeReplayBank:
    model: mujoco.MjModel
    target_xy: np.ndarray
    obstacles: list[ObstacleSpec]
    xy_trace: np.ndarray
    visible_trace: np.ndarray
    height_trace: np.ndarray
    states: dict[int, ReplayState]


def sample_training_weight(meta: dict, step: int, horizon: int) -> float:
    weight = float(TERRAIN_WEIGHT_MULTIPLIER.get(str(meta.get("terrain_profile", "flat")), 1.0))
    success = bool(meta.get("success", False))
    first_visible = int(meta.get("target_first_visible_step", -1))
    reached_step = int(meta.get("target_reached_step", -1))
    if first_visible >= 0:
        if step < first_visible:
            weight *= 0.70
        else:
            weight *= 1.25
    else:
        weight *= 0.50
    if success and reached_step >= 0:
        remaining = max(reached_step - step, 0)
        weight *= 1.0 + 1.8 * math.exp(-remaining / max(1.0, float(horizon) * 2.0))
        if step >= max(0, reached_step - 64):
            weight *= 1.40
    if bool(meta.get("target_discovered_before_reach", False)):
        weight *= 1.08
    if not success:
        weight *= 0.85
    return float(np.clip(weight, 0.20, 2.50))


def score_target(meta: dict, step: int, horizon: int) -> float:
    success = bool(meta.get("success", False))
    first_visible = int(meta.get("target_first_visible_step", -1))
    reached_step = int(meta.get("target_reached_step", -1))
    if success and reached_step >= 0:
        remaining = max(reached_step - step, 0)
        return float(np.exp(-remaining / max(1.0, float(horizon) * 2.0)))
    if first_visible >= 0:
        return 0.55 if step >= first_visible else 0.25
    return 0.05


def _metadata_object_specs(meta: dict) -> list[ObjectSpec]:
    result = []
    for item in meta["objects"]:
        name = str(item["name"])
        kind = "cylinder" if "cylinder" in name else "box"
        size = tuple(item.get("size", (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)))
        result.append(
            ObjectSpec(
                name=name,
                label=str(item.get("label", name)),
                kind=kind,
                position=tuple(item["position"]),
                rgba=tuple(item.get("rgba", (0.8, 0.8, 0.8, 1.0))),
                size=size,
            )
        )
    return result


def _metadata_obstacle_specs(meta: dict) -> list[ObstacleSpec]:
    result = []
    for item in meta.get("obstacles", []):
        result.append(
            ObstacleSpec(
                name=str(item["name"]),
                kind=str(item.get("kind", "box")),
                position=tuple(item["position"]),
                size=tuple(item["size"]),
                rgba=tuple(item.get("rgba", (0.3, 0.33, 0.36, 1.0))),
            )
        )
    return result


def _metadata_scene_light(meta: dict) -> SceneLightSpec | None:
    item = meta.get("scene_light") or {}
    if not item:
        return None
    return SceneLightSpec(
        pos=tuple(item.get("pos", (0.0, 0.0, 5.0))),
        direction=tuple(item.get("direction", (0.0, 0.0, -1.0))),
        diffuse=tuple(item.get("diffuse", (0.85, 0.85, 0.85))),
        ambient=tuple(item.get("ambient", (0.0, 0.0, 0.0))),
        specular=tuple(item.get("specular", (0.0, 0.0, 0.0))),
        directional=bool(item.get("directional", True)),
    )


def _capture_replay_state(data: mujoco.MjData, controller: M20LowLevelController) -> ReplayState:
    return ReplayState(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        ctrl=data.ctrl.copy(),
        act=data.act.copy() if hasattr(data, "act") and data.act is not None else None,
        qacc_warmstart=(
            data.qacc_warmstart.copy()
            if hasattr(data, "qacc_warmstart") and data.qacc_warmstart is not None
            else None
        ),
        time=float(data.time),
        controller_state=controller.snapshot(),
    )


def _restore_replay_state(model: mujoco.MjModel, state: ReplayState) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    data.ctrl[:] = state.ctrl
    if state.act is not None and hasattr(data, "act") and data.act.size == state.act.size:
        data.act[:] = state.act
    if (
        state.qacc_warmstart is not None
        and hasattr(data, "qacc_warmstart")
        and data.qacc_warmstart.size == state.qacc_warmstart.size
    ):
        data.qacc_warmstart[:] = state.qacc_warmstart
    data.time = float(state.time)
    mujoco.mj_forward(model, data)
    return data


def _visible_from_xy(point_xy: np.ndarray, target_xy: np.ndarray, obstacles: list[ObstacleSpec]) -> float:
    return 0.0 if any(
        obstacle_blocks_segment(point_xy, target_xy, obstacle, padding=0.04)
        for obstacle in obstacles
    ) else 1.0


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_qpos(qpos: np.ndarray) -> float:
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    return float(np.arctan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
    ))


def belief_targets_from_state(bank: EpisodeReplayBank | None, step: int) -> tuple[float, float, float, float]:
    if bank is None or step not in bank.states:
        return 0.0, 1.0, 1.0, 0.0
    state = bank.states[int(step)]
    xy = np.asarray(state.qpos[:2], dtype=np.float64)
    delta = bank.target_xy - xy
    distance = float(np.linalg.norm(delta))
    bearing = _wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - _yaw_from_qpos(state.qpos))
    visible = _visible_from_xy(xy, bank.target_xy, bank.obstacles)
    return (
        float(math.sin(bearing)),
        float(math.cos(bearing)),
        float(np.clip(distance / 4.0, 0.0, 1.0)),
        float(visible),
    )


def _replay_episode_bank(
    meta: dict,
    arrays: dict[str, np.ndarray],
    *,
    capture_steps: set[int] | None = None,
) -> EpisodeReplayBank:
    """Replay recorded actions and optionally save simulator states."""
    tmpdir = Path(tempfile.mkdtemp(prefix="m20_traj_scorer_"))
    scene_path = tmpdir / "scene.xml"
    try:
        terrain_profile = str(meta.get("terrain_profile", "flat"))
        terrain = None if terrain_profile in {"", "flat"} else TerrainSpec(terrain_profile)
        build_scene(
            scene_path,
            _metadata_object_specs(meta),
            obstacles=_metadata_obstacle_specs(meta),
            task_object_collisions=bool(meta.get("task_object_collisions", True)),
            terrain=terrain,
            light=_metadata_scene_light(meta),
        )
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        controller = M20LowLevelController(model)
        controller.reset(data, float(meta.get("initial_yaw", 0.0)), tuple(meta.get("initial_xy", (0.0, 0.0))))
        for _ in range(int(meta.get("warmup_steps", 0))):
            controller.step(data, np.zeros(4, dtype=np.float64))
        target_xy = np.asarray(meta["target_xy_privileged_label_only"], dtype=np.float64)
        obstacles = _metadata_obstacle_specs(meta)
        xy_trace = np.zeros((arrays["action"].shape[0], 2), dtype=np.float32)
        visible_trace = np.zeros((arrays["action"].shape[0],), dtype=np.float32)
        height_trace = np.zeros((arrays["action"].shape[0],), dtype=np.float32)
        states: dict[int, ReplayState] = {}
        capture_steps = capture_steps or set()
        for step in range(arrays["action"].shape[0]):
            if step in capture_steps:
                states[step] = _capture_replay_state(data, controller)
            xy = np.asarray(data.qpos[:2], dtype=np.float64)
            xy_trace[step] = xy.astype(np.float32)
            height_trace[step] = float(data.qpos[2])
            visible_trace[step] = _visible_from_xy(xy, target_xy, obstacles)
            controller.step(data, arrays["action"][step].astype(np.float64))
        return EpisodeReplayBank(
            model=model,
            target_xy=target_xy.astype(np.float64, copy=False),
            obstacles=obstacles,
            xy_trace=xy_trace,
            visible_trace=visible_trace,
            height_trace=height_trace,
            states=states,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _replay_episode_dynamics(meta: dict, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay recorded actions to recover real base XY / visibility labels."""
    bank = _replay_episode_bank(meta, arrays)
    return bank.xy_trace, bank.visible_trace, bank.height_trace


def binary_labels(meta: dict, step: int, horizon: int) -> tuple[float, float, float]:
    first_visible = int(meta.get("target_first_visible_step", -1))
    reached_step = int(meta.get("target_reached_step", -1))
    success = bool(meta.get("success", False))
    min_height = float(meta.get("min_base_height", 0.0))
    visible = 1.0 if first_visible >= 0 and first_visible <= step + horizon else 0.0
    reach = 1.0 if success and reached_step >= 0 and reached_step <= step + horizon else 0.0
    safe = 1.0 if min_height >= 0.45 else 0.0
    return visible, reach, safe


def action_smoothness_target(chunk: np.ndarray, *, future_min_distance: float) -> float:
    """Soft label for stable, non-premature candidate motion.

    This is not a reward for standing still.  It penalizes high normalized
    effort and discontinuity, then explicitly marks stop-heavy chunks as bad
    when replay says the target is still not close enough.
    """
    if chunk.size == 0:
        return 0.0
    motion = np.asarray(chunk[:, :3], dtype=np.float32)
    scales = np.asarray([0.35, 0.10, 0.15], dtype=np.float32)
    normalized = motion / scales[None, :]
    effort = float(np.mean(np.abs(normalized)))
    if normalized.shape[0] > 1:
        jerk = float(np.mean(np.abs(np.diff(normalized, axis=0))))
    else:
        jerk = 0.0
    stop_fraction = float(np.mean(np.asarray(chunk[:, 3], dtype=np.float32) >= 0.5))
    smooth = math.exp(-0.18 * effort - 1.40 * jerk)
    if stop_fraction > 0.25 and future_min_distance > STANDOFF_REACH_RADIUS_M:
        smooth *= 0.15
    mean_forward = float(np.mean(np.abs(motion[:, 0])))
    mean_yaw = float(np.mean(np.abs(motion[:, 2])))
    if future_min_distance > STANDOFF_REACH_RADIUS_M + 0.12 and mean_forward < 0.035 and mean_yaw < 0.03:
        # A nearly static non-stop chunk can look "smooth" numerically, but it
        # is the wrong behavior when the robot is still well outside the final
        # standoff radius.  Penalize stagnation while still allowing deliberate
        # turn-in-place search motions.
        smooth *= 0.45
    elif future_min_distance > STANDOFF_REACH_RADIUS_M + 0.25 and mean_forward < 0.05 and mean_yaw < 0.08:
        smooth *= 0.70
    return float(np.clip(smooth, 0.0, 1.0))


def outcome_targets_from_trace(
    *,
    current_distance: float,
    current_visible: float,
    future_distances: np.ndarray,
    future_visible: np.ndarray,
    future_heights: np.ndarray,
    action_chunk: np.ndarray,
    finite: bool = True,
) -> dict[str, float]:
    future_distances = np.asarray(future_distances, dtype=np.float32)
    future_visible = np.asarray(future_visible, dtype=np.float32)
    future_heights = np.asarray(future_heights, dtype=np.float32)
    if future_distances.size:
        future_min_distance = float(future_distances.min())
        final_distance = float(future_distances[-1])
    else:
        future_min_distance = float(current_distance)
        final_distance = float(current_distance)
    current_excess = max(0.0, float(current_distance) - STANDOFF_REACH_RADIUS_M)
    future_excess = max(0.0, future_min_distance - STANDOFF_REACH_RADIUS_M)
    progress_delta = current_excess - future_excess
    progress_target = float(np.clip(progress_delta / 0.28, 0.0, 1.0))
    final_close_target = float(np.exp(-future_excess / STANDOFF_CLOSE_DECAY_M))
    visible_target = float(future_visible.max() > 0.5) if future_visible.size else 0.0
    if future_visible.size:
        if float(current_visible) > 0.5:
            visibility_keep_target = float(np.mean(future_visible))
        else:
            visibility_keep_target = float(
                0.70 * float(future_visible.max() > 0.5) + 0.30 * np.mean(future_visible)
            )
    else:
        visibility_keep_target = 0.0
    reach_target = float(future_min_distance <= STANDOFF_REACH_RADIUS_M)
    safe_target = float(bool(finite) and future_heights.size > 0 and float(future_heights.min()) >= 0.45)
    smooth_target = action_smoothness_target(action_chunk, future_min_distance=future_min_distance)
    no_progress_penalty = float(final_distance > float(current_distance) + 0.04)
    stall_penalty = float(progress_target < 0.10 and future_excess > 0.10)
    score_value = float(
        np.clip(
            0.40 * final_close_target
            + 0.48 * progress_target
            + 0.08 * visibility_keep_target
            + 0.03 * safe_target
            + 0.01 * smooth_target
            - 0.20 * no_progress_penalty
            - 0.10 * stall_penalty
            - 0.10 * float(not finite),
            0.0,
            1.0,
        )
    )
    return {
        "score": score_value,
        "visible": float(visible_target),
        "reach": float(reach_target),
        "safe": float(safe_target),
        "progress": float(progress_target),
        "final_close": float(final_close_target),
        "visibility_keep": float(visibility_keep_target),
        "smooth": float(smooth_target),
    }


def _rollout_candidate_targets(bank: EpisodeReplayBank, step: int, chunk: np.ndarray) -> dict[str, float] | None:
    state = bank.states.get(int(step))
    if state is None:
        return None
    data = _restore_replay_state(bank.model, state)
    controller = M20LowLevelController(bank.model)
    controller.restore(state.controller_state)
    current_xy = np.asarray(data.qpos[:2], dtype=np.float64)
    current_distance = float(np.linalg.norm(bank.target_xy - current_xy))
    current_visible = _visible_from_xy(current_xy, bank.target_xy, bank.obstacles)
    future_distances: list[float] = []
    future_visible: list[float] = []
    future_heights: list[float] = []
    finite = True
    for action in np.asarray(chunk, dtype=np.float32):
        diagnostics = controller.step(data, action.astype(np.float64))
        xy = np.asarray(data.qpos[:2], dtype=np.float64)
        finite = finite and bool(diagnostics.finite) and bool(np.isfinite(data.qpos).all()) and bool(np.isfinite(data.qvel).all())
        future_distances.append(float(np.linalg.norm(bank.target_xy - xy)))
        future_visible.append(_visible_from_xy(xy, bank.target_xy, bank.obstacles))
        future_heights.append(float(data.qpos[2]))
        if not finite:
            break
    return outcome_targets_from_trace(
        current_distance=current_distance,
        current_visible=current_visible,
        future_distances=np.asarray(future_distances, dtype=np.float32),
        future_visible=np.asarray(future_visible, dtype=np.float32),
        future_heights=np.asarray(future_heights, dtype=np.float32),
        action_chunk=chunk,
        finite=finite,
    )


def select_diverse_proposal_indices(candidates: np.ndarray, *, max_count: int, step: int) -> list[int]:
    if max_count <= 0 or candidates.size == 0:
        return []
    count = int(candidates.shape[0])
    first = candidates[:, 0, :]
    stop_any = candidates[:, :, 3].max(axis=1) >= 0.5
    speed = first[:, 0]
    yaw = first[:, 2]
    motion_norm = np.linalg.norm(first[:, :3], axis=1)
    order: list[int] = []

    def add(index: int | None) -> None:
        if index is None:
            return
        index = int(index)
        if 0 <= index < count and index not in order:
            order.append(index)

    add(0)
    non_stop = np.flatnonzero(~stop_any & (motion_norm > 1.0e-5))
    if non_stop.size:
        add(int(non_stop[np.argmax(speed[non_stop])]))
        straight = non_stop[np.argsort(np.abs(yaw[non_stop]))]
        if straight.size:
            add(int(straight[-1] if speed[int(straight[-1])] >= speed[int(straight[0])] else straight[0]))
            add(int(straight[0]))
        positive = non_stop[yaw[non_stop] > 0.04]
        negative = non_stop[yaw[non_stop] < -0.04]
        if positive.size:
            add(int(positive[np.argmax(speed[positive] + 0.25 * np.abs(yaw[positive]))]))
            add(int(positive[np.argmax(np.abs(yaw[positive]))]))
        if negative.size:
            add(int(negative[np.argmax(speed[negative] + 0.25 * np.abs(yaw[negative]))]))
            add(int(negative[np.argmax(np.abs(yaw[negative]))]))
    stop = np.flatnonzero(stop_any)
    if stop.size:
        add(int(stop[0]))
        add(int(stop[-1]))
    remaining = [index for index in range(count) if index not in order]
    if remaining:
        start = int(step) % len(remaining)
        rotated = remaining[start:] + remaining[:start]
        for index in rotated:
            add(index)
            if len(order) >= max_count:
                break
    return order[:max_count]


def visual_goal_hint_from_rgb(
    front: np.ndarray,
    rear: np.ndarray,
    target_label: str,
    *,
    visible_pixel_threshold: int,
    close_pixel_threshold: int,
) -> tuple[float | None, float | None, float | None, int, str]:
    """Return learner-only visual proposal hint from the exact policy RGB."""
    front_count = target_pixel_count(front, target_label)
    rear_count = target_pixel_count(rear, target_label)
    total_count = int(front_count + rear_count)
    if total_count < int(visible_pixel_threshold):
        return None, None, None, total_count, "none"
    source = "front" if front_count >= rear_count else "rear"
    image = front if source == "front" else rear
    mask = target_color_mask(image, target_label)
    if not mask.any():
        return None, None, None, total_count, source
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    del ys, height
    centroid_x = float(xs.mean())
    normalized_x = (centroid_x - 0.5 * (width - 1)) / max(1.0, 0.5 * width)
    bearing = float(np.clip(-0.75 * normalized_x, -0.75, 0.75))
    if source == "rear":
        bearing = float(np.clip(-np.sign(bearing if abs(bearing) > 1.0e-6 else 1.0) * 0.75, -0.75, 0.75))
    distance = visual_goal_distance_from_pixels(
        total_count,
        close_pixel_threshold=close_pixel_threshold,
    )
    visible_prob = visual_goal_visible_prob_from_pixels(
        total_count,
        visible_pixel_threshold=visible_pixel_threshold,
        close_pixel_threshold=close_pixel_threshold,
    )
    return bearing, distance, visible_prob, total_count, source


def split_paths_by_layout(paths: list[Path], *, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    metadata = {path: json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in paths}
    layout_groups: dict[int, list[Path]] = {}
    for path, meta in metadata.items():
        layout_id = meta.get("layout_id")
        if layout_id is not None:
            layout_groups.setdefault(int(layout_id), []).append(path)
    if layout_groups and len(layout_groups) >= 2 and sum(len(group) for group in layout_groups.values()) == len(paths):
        rng = random.Random(seed)
        layout_ids = sorted(layout_groups)
        rng.shuffle(layout_ids)
        val_layout_count = max(1, int(round(len(layout_ids) * val_fraction)))
        val_layouts = set(layout_ids[:val_layout_count])
        val_paths = [path for layout_id in layout_ids if layout_id in val_layouts for path in sorted(layout_groups[layout_id])]
        train_paths = [path for layout_id in layout_ids if layout_id not in val_layouts for path in sorted(layout_groups[layout_id])]
        return train_paths, val_paths
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


@dataclass(frozen=True)
class SampleRecord:
    latent: np.ndarray
    action_chunk: np.ndarray
    score_target: float
    visible_target: float
    reach_target: float
    safe_target: float
    progress_target: float
    final_close_target: float
    visibility_keep_target: float
    smooth_target: float
    belief_sin_target: float
    belief_cos_target: float
    belief_distance_target: float
    belief_visible_target: float
    sample_weight: float


def _make_rgb_batch(arrays: dict[str, np.ndarray], indices: list[int]) -> torch.Tensor:
    rgb = np.concatenate((arrays["front_rgb"][indices], arrays["rear_rgb"][indices]), axis=-1)
    rgb = rgb.transpose(0, 3, 1, 2)
    return torch.from_numpy(rgb.copy()).float().div_(255.0)


def _make_episode_policy_outputs(
    policy: M20MuJoCoVLA,
    arrays: dict[str, np.ndarray],
    meta: dict,
    step_indices: list[int],
    *,
    device: torch.device,
    history_pixel_threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latents = []
    actions = []
    phase_probs = []
    batch_size = 64
    language = torch.from_numpy(encode_text(meta["task_text"])).unsqueeze(0).to(device)
    executed_actions = arrays.get("executed_action")
    history_features = build_visual_history_feature_trace(
        arrays["front_rgb"],
        arrays["rear_rgb"],
        arrays["action"],
        str(meta["target_label"]),
        executed_actions=executed_actions,
        pixel_threshold=int(history_pixel_threshold),
    )
    with torch.no_grad():
        for start in range(0, len(step_indices), batch_size):
            chunk = step_indices[start : start + batch_size]
            rgb = _make_rgb_batch(arrays, chunk).to(device)
            lidar = torch.from_numpy(arrays["lidar"][chunk].copy()).float().div_(10.0).to(device)
            proprio = torch.from_numpy(arrays["proprio"][chunk].copy()).float().to(device)
            history = torch.from_numpy(history_features[chunk].copy()).float().to(device)
            language_batch = language.expand(len(chunk), -1)
            latent_tensor = policy.encode(rgb, lidar, proprio, language_batch, history=history)
            target_logits = policy.target_head(latent_tensor)
            phase_logits = policy.phase_head(latent_tensor)
            target_context = policy._target_context_from_logits(target_logits)
            action_tensor = policy.action_from_latent(
                latent_tensor,
                target_context,
                phase_logits=phase_logits,
                history=history,
            )
            latents.append(latent_tensor.detach().cpu().numpy())
            actions.append(action_tensor.detach().cpu().numpy())
            phase_probs.append(torch.softmax(phase_logits, dim=-1).detach().cpu().numpy())
    if not latents:
        return (
            np.zeros((0, 192), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    return (
        np.concatenate(latents, axis=0).astype(np.float32, copy=False),
        np.concatenate(actions, axis=0).astype(np.float32, copy=False),
        np.concatenate(phase_probs, axis=0).astype(np.float32, copy=False),
    )


def build_samples(
    dataset_root: Path,
    *,
    policy: M20MuJoCoVLA,
    horizon: int,
    stride: int,
    device: torch.device,
    val_fraction: float,
    seed: int,
    use_replay_labels: bool,
    proposal_labels_per_sample: int,
    proposal_sample_weight: float,
    train_goal_belief: bool,
    visual_target_pixel_threshold: int,
    visual_close_pixel_threshold: int,
    history_pixel_threshold: int,
) -> tuple[list[SampleRecord], list[SampleRecord], dict]:
    episode_paths = [path for path in sorted(dataset_root.glob("episode_*.npz")) if path.with_suffix(".json").is_file()]
    if len(episode_paths) < 4:
        raise RuntimeError(f"Need at least 4 episodes in {dataset_root}")
    train_paths, val_paths = split_paths_by_layout(episode_paths, val_fraction=val_fraction, seed=seed)
    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    for split_name, split_paths, split_bucket in (
        ("train", train_paths, train_records),
        ("val", val_paths, val_records),
    ):
        for episode_index, path in enumerate(split_paths, start=1):
            before_count = len(split_bucket)
            with np.load(path) as arrays:
                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                action_len = int(arrays["action"].shape[0])
                step_indices = list(range(0, max(1, action_len - horizon + 1), stride))
                if not step_indices:
                    continue
                latents, direct_actions, phase_probs = _make_episode_policy_outputs(
                    policy,
                    arrays,
                    meta,
                    step_indices,
                    device=device,
                    history_pixel_threshold=int(history_pixel_threshold),
                )
                replay_bank = None
                if use_replay_labels:
                    replay_bank = _replay_episode_bank(
                        meta,
                        arrays,
                        capture_steps=set(step_indices) if (int(proposal_labels_per_sample) > 0 or bool(train_goal_belief)) else None,
                    )
                    xy_trace = replay_bank.xy_trace
                    visible_trace = visual_target_visible_trace(
                        arrays["front_rgb"],
                        arrays["rear_rgb"],
                        meta["target_label"],
                    )
                    height_trace = replay_bank.height_trace
                    target_xy = np.asarray(meta["target_xy_privileged_label_only"], dtype=np.float32)
                    distance_trace = np.linalg.norm(xy_trace.astype(np.float32) - target_xy[None, :], axis=1).astype(np.float32)
                else:
                    xy_trace = None
                    visible_trace = None
                    height_trace = None
                    distance_trace = None
                for local_index, step in enumerate(step_indices):
                    chunk = arrays["action"][step : step + horizon]
                    if chunk.shape[0] < horizon:
                        pad = np.repeat(chunk[-1:, :], horizon - chunk.shape[0], axis=0) if chunk.shape[0] else np.zeros((horizon, 4), dtype=np.float32)
                        chunk = np.concatenate((chunk, pad), axis=0)
                    if (
                        use_replay_labels
                        and distance_trace is not None
                        and visible_trace is not None
                        and height_trace is not None
                    ):
                        end = min(distance_trace.shape[0], step + horizon)
                        future_slice = distance_trace[step:end]
                        current_distance = float(distance_trace[min(step, distance_trace.shape[0] - 1)])
                        visible_slice = visible_trace[step:end]
                        height_slice = height_trace[step:end]
                        targets = outcome_targets_from_trace(
                            current_distance=current_distance,
                            current_visible=float(visible_trace[min(step, visible_trace.shape[0] - 1)]),
                            future_distances=future_slice,
                            future_visible=visible_slice,
                            future_heights=height_slice,
                            action_chunk=chunk,
                            finite=True,
                        )
                        score_value = targets["score"]
                        visible_target = targets["visible"]
                        reach_target = targets["reach"]
                        safe_target = targets["safe"]
                        progress_target = targets["progress"]
                        final_close_target = targets["final_close"]
                        visibility_keep_target = targets["visibility_keep"]
                        smooth_target = targets["smooth"]
                    else:
                        visible_target, reach_target, safe_target = binary_labels(meta, step, horizon)
                        progress_target = float(score_target(meta, step, horizon))
                        final_close_target = float(reach_target)
                        visibility_keep_target = float(visible_target)
                        smooth_target = action_smoothness_target(
                            chunk,
                            future_min_distance=STANDOFF_REACH_RADIUS_M if reach_target else 1.0,
                        )
                        score_value = float(score_target(meta, step, horizon))
                    belief_sin, belief_cos, belief_distance, belief_visible = belief_targets_from_state(replay_bank, step)
                    if visible_trace is not None:
                        belief_visible = float(visible_trace[min(step, visible_trace.shape[0] - 1)])
                    record = SampleRecord(
                        latent=latents[local_index].astype(np.float32, copy=False),
                        action_chunk=chunk.astype(np.float32, copy=False),
                        score_target=score_value,
                        visible_target=float(visible_target),
                        reach_target=float(reach_target),
                        safe_target=float(safe_target),
                        progress_target=float(progress_target),
                        final_close_target=float(final_close_target),
                        visibility_keep_target=float(visibility_keep_target),
                        smooth_target=float(smooth_target),
                        belief_sin_target=float(belief_sin),
                        belief_cos_target=float(belief_cos),
                        belief_distance_target=float(belief_distance),
                        belief_visible_target=float(belief_visible),
                        sample_weight=float(sample_training_weight(meta, step, horizon)),
                    )
                    split_bucket.append(record)
                    if (
                        int(proposal_labels_per_sample) > 0
                        and use_replay_labels
                        and replay_bank is not None
                        and step in replay_bank.states
                    ):
                        visual_bearing, visual_distance, visual_visible_prob, _, _ = visual_goal_hint_from_rgb(
                            arrays["front_rgb"][step],
                            arrays["rear_rgb"][step],
                            str(meta["target_label"]),
                            visible_pixel_threshold=int(visual_target_pixel_threshold),
                            close_pixel_threshold=int(visual_close_pixel_threshold),
                        )
                        candidates = make_phase_action_chunks(
                            direct_actions[local_index],
                            horizon=horizon,
                            phase_prob=phase_probs[local_index],
                            goal_bearing=visual_bearing,
                            goal_distance=visual_distance,
                            goal_visible_prob=visual_visible_prob,
                        )
                        proposal_indices = select_diverse_proposal_indices(
                            candidates,
                            max_count=int(proposal_labels_per_sample),
                            step=step,
                        )
                        for proposal_index in proposal_indices:
                            proposal_chunk = candidates[proposal_index]
                            proposal_targets = _rollout_candidate_targets(replay_bank, step, proposal_chunk)
                            if proposal_targets is None:
                                continue
                            split_bucket.append(
                                SampleRecord(
                                    latent=latents[local_index].astype(np.float32, copy=False),
                                    action_chunk=proposal_chunk.astype(np.float32, copy=False),
                                    score_target=float(proposal_targets["score"]),
                                    visible_target=float(proposal_targets["visible"]),
                                    reach_target=float(proposal_targets["reach"]),
                                    safe_target=float(proposal_targets["safe"]),
                                    progress_target=float(proposal_targets["progress"]),
                                    final_close_target=float(proposal_targets["final_close"]),
                                    visibility_keep_target=float(proposal_targets["visibility_keep"]),
                                    smooth_target=float(proposal_targets["smooth"]),
                                    belief_sin_target=float(belief_sin),
                                    belief_cos_target=float(belief_cos),
                                    belief_distance_target=float(belief_distance),
                                    belief_visible_target=float(belief_visible),
                                    sample_weight=float(sample_training_weight(meta, step, horizon) * float(proposal_sample_weight)),
                                )
                            )
            added_count = len(split_bucket) - before_count
            print(
                f"[M20-WORLD-SCORER-BUILD] split={split_name} "
                f"episode={episode_index:03d}/{len(split_paths):03d} "
                f"path={path.name} added_samples={added_count} total_split_samples={len(split_bucket)}",
                flush=True,
            )
    info = {
        "episode_count": len(episode_paths),
        "train_episodes": len(train_paths),
        "val_episodes": len(val_paths),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
    }
    return train_records, val_records, info


class TrajectoryScorerDataset(Dataset):
    def __init__(self, records: list[SampleRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "latent": torch.from_numpy(record.latent),
            "action_chunk": torch.from_numpy(record.action_chunk),
            "score_target": torch.tensor(record.score_target, dtype=torch.float32),
            "visible_target": torch.tensor(record.visible_target, dtype=torch.float32),
            "reach_target": torch.tensor(record.reach_target, dtype=torch.float32),
            "safe_target": torch.tensor(record.safe_target, dtype=torch.float32),
            "progress_target": torch.tensor(record.progress_target, dtype=torch.float32),
            "final_close_target": torch.tensor(record.final_close_target, dtype=torch.float32),
            "visibility_keep_target": torch.tensor(record.visibility_keep_target, dtype=torch.float32),
            "smooth_target": torch.tensor(record.smooth_target, dtype=torch.float32),
            "belief_sin_target": torch.tensor(record.belief_sin_target, dtype=torch.float32),
            "belief_cos_target": torch.tensor(record.belief_cos_target, dtype=torch.float32),
            "belief_distance_target": torch.tensor(record.belief_distance_target, dtype=torch.float32),
            "belief_visible_target": torch.tensor(record.belief_visible_target, dtype=torch.float32),
            "sample_weight": torch.tensor(record.sample_weight, dtype=torch.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(".runtime/datasets/m20_search_balanced_gate_v7"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path(".runtime/checkpoints/m20_search_balanced_gate_v7_hidden_vla_v3_dense_stepboost/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/checkpoints/m20_mujoco_trajectory_outcome_v2"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional trajectory-scorer checkpoint used to warm-start the world model.",
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--val-fraction", type=float, default=0.30)
    parser.add_argument("--use-replay-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--proposal-labels-per-sample",
        type=int,
        default=0,
        help="Roll out this many learner-only phase-aware proposal chunks per sampled state and train scorer on their measured outcomes.",
    )
    parser.add_argument("--proposal-sample-weight", type=float, default=0.70)
    parser.add_argument("--train-goal-belief", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--goal-belief-loss-weight", type=float, default=0.35)
    parser.add_argument("--visual-target-pixel-threshold", type=int, default=5)
    parser.add_argument(
        "--history-pixel-threshold",
        type=int,
        default=DEFAULT_HISTORY_PIXEL_THRESHOLD,
        help="Target-color pixel threshold used to rebuild the same learner-only history features used by the VLA checkpoint.",
    )
    parser.add_argument(
        "--visual-close-pixel-threshold",
        type=int,
        default=None,
        help="RGB target-pixel count mapped to roughly success-radius scale; defaults to 2% of combined front+rear pixels.",
    )
    parser.add_argument("--visual-close-area-fraction", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.proposal_labels_per_sample < 0 or args.proposal_sample_weight < 0.0 or args.goal_belief_loss_weight < 0.0:
        raise ValueError("proposal labeling parameters must be non-negative")
    if args.visual_target_pixel_threshold <= 0:
        raise ValueError("--visual-target-pixel-threshold must be positive")
    if args.history_pixel_threshold <= 0:
        raise ValueError("--history-pixel-threshold must be positive")
    if args.visual_close_area_fraction <= 0.0:
        raise ValueError("--visual-close-area-fraction must be positive")

    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if args.init_checkpoint is not None and not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)

    first_episode = next(iter(sorted(args.dataset.glob("episode_*.npz"))), None)
    if first_episode is None:
        raise RuntimeError(f"No episode_*.npz files found in {args.dataset}")
    with np.load(first_episode) as first_arrays:
        height, width = first_arrays["front_rgb"].shape[1:3]
    if args.visual_close_pixel_threshold is None:
        visual_close_pixel_threshold = max(
            TARGET_PIXEL_THRESHOLD,
            int(round(float(args.visual_close_area_fraction) * float(width * height * 2))),
        )
    else:
        visual_close_pixel_threshold = int(args.visual_close_pixel_threshold)
    if visual_close_pixel_threshold < int(args.visual_target_pixel_threshold):
        raise ValueError("--visual-close-pixel-threshold must be >= --visual-target-pixel-threshold")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    policy = M20MuJoCoVLA.for_state_dict(base_ckpt["model_state_dict"]).to(device)
    policy.load_state_dict(base_ckpt["model_state_dict"], strict=False)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    train_records, val_records, build_info = build_samples(
        args.dataset,
        policy=policy,
        horizon=args.horizon,
        stride=args.stride,
        device=device,
        val_fraction=args.val_fraction,
        seed=args.seed,
        use_replay_labels=bool(args.use_replay_labels),
        proposal_labels_per_sample=int(args.proposal_labels_per_sample),
        proposal_sample_weight=float(args.proposal_sample_weight),
        train_goal_belief=bool(args.train_goal_belief),
        visual_target_pixel_threshold=int(args.visual_target_pixel_threshold),
        visual_close_pixel_threshold=int(visual_close_pixel_threshold),
        history_pixel_threshold=int(args.history_pixel_threshold),
    )
    if len(train_records) < 8 or len(val_records) < 4:
        raise RuntimeError(f"Not enough scorer samples: train={len(train_records)} val={len(val_records)}")

    train_loader = DataLoader(TrajectoryScorerDataset(train_records), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TrajectoryScorerDataset(val_records), batch_size=args.batch_size, shuffle=False, num_workers=0)

    belief_dim = 4 if bool(args.train_goal_belief) else 0
    scorer = M20TrajectoryScorer(latent_dim=192, horizon=args.horizon, action_dim=4, belief_dim=belief_dim).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mse = torch.nn.MSELoss(reduction="none")
    belief_loss_weight = float(args.goal_belief_loss_weight) if belief_dim > 0 else 0.0
    loss_weights = {
        "score": 0.90,
        "visible": 0.30,
        "reach": 0.95,
        "safe": 0.45,
        "progress": 1.25,
        "final_close": 1.20,
        "visibility_keep": 0.35,
        "smooth": 0.25,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "m20pro_mujoco_trajectory_outcome_model_v2",
        "dataset": str(args.dataset),
        "base_checkpoint": str(args.base_checkpoint),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else "",
        "device": str(device),
        "horizon": int(args.horizon),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "build_info": build_info,
        "belief_dim": int(belief_dim),
        "goal_belief": {
            "enabled": bool(args.train_goal_belief),
            "loss_weight": float(args.goal_belief_loss_weight),
            "target_order": ["sin_bearing", "cos_bearing", "distance_div_4", "visible"],
            "geometry_loss_contract": "bearing/distance loss is strongly weighted only when target is line-of-sight visible; occluded frames train visibility without privileged bearing memorization",
            "runtime_contract": "belief is predicted from learner-only multimodal latent; target_xy is offline label only",
        },
        "proposal_labeling": {
            "enabled": bool(int(args.proposal_labels_per_sample) > 0),
            "proposal_labels_per_sample": int(args.proposal_labels_per_sample),
            "proposal_sample_weight": float(args.proposal_sample_weight),
            "proposal_contract": "phase-aware learner-only action proposals are short MuJoCo rollouts from recorded states; target_xy is used only for offline labels",
        },
        "visual_goal_proposals": {
            "enabled": True,
            "visual_target_pixel_threshold": int(args.visual_target_pixel_threshold),
            "visual_close_pixel_threshold": int(visual_close_pixel_threshold),
            "visual_close_area_fraction": float(args.visual_close_area_fraction),
            "contract": "when target-color pixels are visible in policy RGB, candidate chunks include learner-only visual-bearing approach proposals; privileged target_xy is used only to label rollout outcomes offline",
        },
        "history_contract": {
            "enabled": True,
            "history_pixel_threshold": int(args.history_pixel_threshold),
            "contract": "world-model latent extraction rebuilds the same learner-only VisualHistoryTracker features used by VLA training/runtime; no target_xy/object_id/semantic mask",
        },
        "head_order": list(SCORER_HEAD_ORDER),
        "label_contract": {
            "score": "weighted soft utility emphasizing progress and final_close with backward/stall/unstable penalties",
            "visible": "target becomes line-of-sight visible within candidate horizon",
            "reach": "future min target distance <= 0.45m safe-standoff radius",
            "safe": "future min base height >= 0.45m",
            "progress": "soft clipped reduction of distance remaining outside the 0.45m standoff radius",
            "final_close": "exp(-max(0, future_min_distance - 0.45) / 0.55), saturated inside the safe-standoff radius",
            "visibility_keep": "future target visibility retention or discovery",
            "smooth": "low normalized effort/jerk with no premature stop and no far-from-goal stagnation",
        },
        "loss_weights": loss_weights,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.init_checkpoint is not None:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        init_config = init_ckpt.get("config", {})
        expected = {
            "horizon": int(args.horizon),
            "head_order": list(SCORER_HEAD_ORDER),
            "belief_dim": int(belief_dim),
        }
        observed = {
            "horizon": int(init_config.get("horizon", -1)),
            "head_order": list(init_config.get("head_order", [])),
            "belief_dim": int(init_config.get("belief_dim", -1)),
        }
        if observed != expected:
            raise RuntimeError(f"init trajectory-scorer checkpoint is incompatible: expected={expected} observed={observed}")
        missing, unexpected = scorer.load_state_dict(init_ckpt["model_state_dict"], strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "init_missing_keys": list(missing),
                    "init_unexpected_keys": list(unexpected),
                },
                indent=2,
            ),
            flush=True,
        )

    best = float("inf")
    history = []
    head_names = list(SCORER_HEAD_ORDER)
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        train_total = 0.0
        train_weight = 0.0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            sample_weight = batch.pop("sample_weight")
            pred = scorer(batch["latent"], batch["action_chunk"])
            per_sample_loss = torch.zeros_like(sample_weight)
            for head_index, head_name in enumerate(head_names):
                per_sample_loss = per_sample_loss + float(loss_weights[head_name]) * bce(
                    pred[:, head_index],
                    batch[f"{head_name}_target"],
                )
            if belief_dim > 0:
                belief_pred = scorer.predict_belief(batch["latent"])
                visible_mask = batch["belief_visible_target"].clamp(0.0, 1.0)
                geo_weight = 0.05 + 0.95 * visible_mask
                belief_loss = (
                    geo_weight
                    * (
                        mse(belief_pred[:, 0], batch["belief_sin_target"])
                        + mse(belief_pred[:, 1], batch["belief_cos_target"])
                        + mse(belief_pred[:, 2], batch["belief_distance_target"])
                    )
                    + 1.20 * bce(belief_pred[:, 3], batch["belief_visible_target"])
                )
                per_sample_loss = per_sample_loss + belief_loss_weight * belief_loss
            wsum = sample_weight.sum().clamp_min(1.0e-6)
            loss = (per_sample_loss * sample_weight).sum() / wsum
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * float(wsum.detach())
            train_weight += float(wsum.detach())

        scorer.eval()
        val_total = 0.0
        val_weight = 0.0
        val_correct = {head_name: 0 for head_name in head_names}
        belief_bearing_error = 0.0
        belief_distance_error = 0.0
        belief_visible_correct = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sample_weight = batch.pop("sample_weight")
                pred = scorer(batch["latent"], batch["action_chunk"])
                per_sample_loss = torch.zeros_like(sample_weight)
                for head_index, head_name in enumerate(head_names):
                    target = batch[f"{head_name}_target"]
                    per_sample_loss = per_sample_loss + float(loss_weights[head_name]) * bce(pred[:, head_index], target)
                    val_correct[head_name] += int(((torch.sigmoid(pred[:, head_index]) > 0.5) == (target > 0.5)).sum())
                if belief_dim > 0:
                    belief_pred = scorer.predict_belief(batch["latent"])
                    visible_mask = batch["belief_visible_target"].clamp(0.0, 1.0)
                    geo_weight = 0.05 + 0.95 * visible_mask
                    belief_loss = (
                        geo_weight
                        * (
                            mse(belief_pred[:, 0], batch["belief_sin_target"])
                            + mse(belief_pred[:, 1], batch["belief_cos_target"])
                            + mse(belief_pred[:, 2], batch["belief_distance_target"])
                        )
                        + 1.20 * bce(belief_pred[:, 3], batch["belief_visible_target"])
                    )
                    per_sample_loss = per_sample_loss + belief_loss_weight * belief_loss
                    pred_bearing = torch.atan2(belief_pred[:, 0].clamp(-1.0, 1.0), belief_pred[:, 1].clamp(-1.0, 1.0))
                    target_bearing = torch.atan2(batch["belief_sin_target"], batch["belief_cos_target"])
                    wrapped_error = torch.atan2(torch.sin(pred_bearing - target_bearing), torch.cos(pred_bearing - target_bearing)).abs()
                    belief_bearing_error += float(wrapped_error.sum())
                    belief_distance_error += float((belief_pred[:, 2].clamp(0.0, 1.0) - batch["belief_distance_target"]).abs().sum() * 4.0)
                    belief_visible_correct += int(((torch.sigmoid(belief_pred[:, 3]) > 0.5) == (batch["belief_visible_target"] > 0.5)).sum())
                wsum = sample_weight.sum().clamp_min(1.0e-6)
                loss = (per_sample_loss * sample_weight).sum() / wsum
                val_total += float(loss) * float(wsum.detach())
                val_weight += float(wsum.detach())
                count += int(batch["latent"].shape[0])
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_total / train_weight,
            "val_loss": val_total / val_weight,
            "lr": scheduler.get_last_lr()[0],
        }
        row.update({f"val_{head_name}_accuracy": val_correct[head_name] / count for head_name in head_names})
        if belief_dim > 0:
            row["val_belief_bearing_mae_rad"] = belief_bearing_error / count
            row["val_belief_distance_mae_m"] = belief_distance_error / count
            row["val_belief_visible_accuracy"] = belief_visible_correct / count
        history.append(row)
        print(
            f"[M20-WORLD-SCORER] epoch={epoch:03d}/{args.epochs} "
            f"train={row['train_loss']:.5f} val={row['val_loss']:.5f} "
            f"score_acc={row['val_score_accuracy']:.3f} "
            f"progress_acc={row['val_progress_accuracy']:.3f} "
            f"close_acc={row['val_final_close_accuracy']:.3f} "
            f"reach_acc={row['val_reach_accuracy']:.3f}"
            + (
                f" bearing_mae={row['val_belief_bearing_mae_rad']:.3f} dist_mae={row['val_belief_distance_mae_m']:.3f}"
                if belief_dim > 0
                else ""
            ),
            flush=True,
        )
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": scorer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best,
                },
                args.output_dir / "best.pt",
            )

    torch.save({"model_state_dict": scorer.state_dict(), "config": config, "epoch": args.epochs}, args.output_dir / "last.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output_dir / "best.pt"), "best_val_loss": best, **build_info}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a trained M20 MuJoCo VLA checkpoint in a learner-only episode."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from m20pro_vla.data.visibility import TARGET_PIXEL_THRESHOLD, target_color_mask, target_pixel_count
from m20pro_vla.data import HISTORY_FEATURE_LABELS, VisualHistoryTracker
from m20pro_vla.sim.mujoco import (
    ASSET,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECTS,
    ObstacleSpec,
    ObjectSpec,
    SceneLightSpec,
    TerrainSpec,
    build_scene,
    obstacle_blocks_segment,
    open_video,
    planar_lidar,
    proprioception,
)
from m20pro_vla.low_level import build_low_level_controller
from m20pro_vla.planning import SearchMPCConfig, SearchMPCPlanner
from m20pro_vla.policies import M20MuJoCoVLA, PHASE_LABELS, encode_text
from m20pro_vla.world_model import (
    M20TrajectoryScorer,
    make_phase_action_chunks,
    scorer_combined_score,
    visual_goal_alignment_prior,
    visual_goal_distance_from_pixels,
    visual_goal_visible_prob_from_pixels,
)
from m20pro_vla.world_model.trajectory_scorer import LEGACY_SCORER_HEAD_ORDER, SCORER_HEAD_ORDER


def infer_scorer_head_dim(checkpoint: dict) -> int:
    config_order = checkpoint.get("config", {}).get("head_order")
    if isinstance(config_order, (list, tuple)) and len(config_order) > 0:
        return int(len(config_order))
    state_dict = checkpoint.get("model_state_dict", {})
    for key in ("head.3.weight", "head.3.bias"):
        value = state_dict.get(key)
        if value is not None:
            return int(value.shape[0])
    return int(len(SCORER_HEAD_ORDER))


def infer_scorer_belief_dim(checkpoint: dict) -> int:
    config_value = checkpoint.get("config", {}).get("belief_dim")
    if config_value is not None:
        return int(config_value)
    state_dict = checkpoint.get("model_state_dict", {})
    for key in ("belief_head.3.weight", "belief_head.3.bias"):
        value = state_dict.get(key)
        if value is not None:
            return int(value.shape[0])
    return 0


def scorer_head_order_from_checkpoint(checkpoint: dict, head_dim: int) -> list[str]:
    config_order = checkpoint.get("config", {}).get("head_order")
    if isinstance(config_order, (list, tuple)) and len(config_order) == head_dim:
        return [str(item) for item in config_order]
    if head_dim == len(LEGACY_SCORER_HEAD_ORDER):
        return list(LEGACY_SCORER_HEAD_ORDER)
    if head_dim <= len(SCORER_HEAD_ORDER):
        return list(SCORER_HEAD_ORDER[:head_dim])
    return list(SCORER_HEAD_ORDER) + [f"extra_{index}" for index in range(len(SCORER_HEAD_ORDER), head_dim)]


def scorer_probability(head_output: torch.Tensor, head_order: list[str], name: str, *, default: float | None = None) -> float | None:
    if name not in head_order:
        return default
    index = head_order.index(name)
    if index >= int(head_output.shape[-1]):
        return default
    return float(torch.sigmoid(head_output[0, index]).item())


def command_from_goal_belief(
    *,
    bearing: float,
    distance: float,
    visible_prob: float,
) -> np.ndarray:
    bearing = float(np.clip(bearing, -math.pi, math.pi))
    distance = max(0.0, float(distance))
    visible_prob = float(np.clip(visible_prob, 0.0, 1.0))
    if visible_prob >= 0.50 and distance <= 0.45:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    if distance <= 0.45:
        forward = 0.04 if abs(bearing) <= 0.22 else 0.0
    elif distance <= 0.75:
        forward = 0.08 if abs(bearing) <= 0.35 else 0.0
    elif abs(bearing) > 0.80:
        forward = 0.04
    elif abs(bearing) > 0.45:
        forward = 0.10
    elif distance > 1.50:
        forward = 0.22
    else:
        forward = 0.16
    yaw_gain = 0.70 if visible_prob >= 0.35 else 0.55
    yaw = float(np.clip(yaw_gain * bearing, -0.15, 0.15))
    return np.asarray((forward, 0.0, yaw, 0.0), dtype=np.float32)


def target_rgb_observation(
    front: np.ndarray,
    rear: np.ndarray,
    target_label: str,
    *,
    pixel_threshold: int,
) -> dict[str, float | int | bool | str | None]:
    """Learner-only target-color observation from the exact policy RGB frames."""
    front_count = target_pixel_count(front, target_label)
    rear_count = target_pixel_count(rear, target_label)
    total_count = int(front_count + rear_count)
    source = "front" if front_count >= rear_count else "rear"
    image = front if source == "front" else rear
    mask = target_color_mask(image, target_label)
    bearing = None
    centroid_x = None
    centroid_y = None
    if mask.any():
        ys, xs = np.nonzero(mask)
        height, width = mask.shape
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        # Approximate horizontal bearing from image x.  Positive yaw means
        # turn left in the body-command contract; image-right is negative yaw.
        normalized_x = (centroid_x - 0.5 * (width - 1)) / max(1.0, 0.5 * width)
        bearing = float(np.clip(-0.75 * normalized_x, -0.75, 0.75))
        if source == "rear":
            bearing = float(np.clip(-np.sign(bearing if abs(bearing) > 1.0e-6 else 1.0) * 0.75, -0.75, 0.75))
    return {
        "visible": bool(total_count >= int(pixel_threshold)),
        "front_count": int(front_count),
        "rear_count": int(rear_count),
        "total_count": int(total_count),
        "source": source,
        "bearing": bearing,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area_fraction": float(total_count / max(1, front.shape[0] * front.shape[1] + rear.shape[0] * rear.shape[1])),
    }


def command_from_rgb_target_observation(observation: dict[str, float | int | bool | str | None]) -> np.ndarray:
    """Visual-servo body command from learner-only target pixels.

    This is used as a safety/approach guard, not as privileged supervision:
    the only inputs are the same low-resolution RGB frames fed to the policy
    and the target label implied by language.
    """
    total_count = int(observation.get("total_count") or 0)
    area_fraction = float(observation.get("area_fraction") or 0.0)
    bearing = observation.get("bearing")
    source = str(observation.get("source") or "front")
    if bearing is None:
        return np.asarray((0.10, 0.0, 0.10, 0.0), dtype=np.float32)
    yaw = float(np.clip(0.45 * float(bearing), -0.15, 0.15))
    if source == "rear":
        return np.asarray((0.0, 0.0, yaw if abs(yaw) > 0.04 else 0.12, 0.0), dtype=np.float32)
    if total_count >= 850 or area_fraction >= 0.110:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    if total_count >= 520 or area_fraction >= 0.067:
        forward = 0.035
    elif total_count >= 220 or area_fraction >= 0.028:
        forward = 0.075
    else:
        forward = 0.13
    if abs(yaw) > 0.11:
        forward *= 0.45
    elif abs(yaw) > 0.07:
        forward *= 0.70
    return np.asarray((forward, 0.0, yaw, 0.0), dtype=np.float32)


def command_from_visual_search_guard(step: int, lidar: np.ndarray) -> np.ndarray:
    """Learner-only obstacle-aware search primitive for pre-discovery frames."""
    scan = np.asarray(lidar, dtype=np.float32)
    # Use a deterministic coverage sweep instead of greedily choosing the
    # locally more open side.  The held-out hidden-search layouts place targets
    # on both sides of a central occluder; a greedy one-sided choice can keep
    # the target permanently out of both policy cameras.
    phase = (int(step) // 360) % 4
    scheduled_yaw = (-0.12, 0.14, 0.10, -0.14)[phase]
    if scan.shape[0] < 72 or not np.isfinite(scan).all():
        return np.asarray((0.12, 0.0, scheduled_yaw, 0.0), dtype=np.float32)
    front = float(np.min(scan[32:41]))
    left = float(np.mean(scan[48:62]))
    right = float(np.mean(scan[10:24]))
    if scheduled_yaw > 0.0 and left < 0.45 and right > left:
        scheduled_yaw = -0.14
    elif scheduled_yaw < 0.0 and right < 0.45 and left > right:
        scheduled_yaw = 0.14
    if front < 0.42:
        return np.asarray((0.0, 0.0, scheduled_yaw, 0.0), dtype=np.float32)
    if front < 0.85:
        return np.asarray((0.055, 0.0, scheduled_yaw, 0.0), dtype=np.float32)
    return np.asarray((0.16, 0.0, scheduled_yaw, 0.0), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-mode", choices=("direct", "trajectory_scorer", "belief_guided", "search_mpc", "phase_routed"), default="direct")
    parser.add_argument("--trajectory-scorer", type=Path, default=None)
    parser.add_argument(
        "--trajectory-scorer-mode",
        choices=("selector", "guarded_selector", "gate"),
        default="selector",
        help=(
            "Selector ranks chunks directly; guarded_selector keeps direct VLA unless the scorer predicts a clearly "
            "safer/closer/smoother alternative; gate keeps direct motion and uses the scorer only to veto premature stop."
        ),
    )
    parser.add_argument(
        "--trajectory-scorer-interval",
        type=int,
        default=1,
        help="Run learned trajectory scorer every N sim steps and reuse the last selected command in between.",
    )
    parser.add_argument("--stop-threshold", type=float, default=0.50)
    parser.add_argument("--stop-visible-threshold", type=float, default=0.50)
    parser.add_argument("--stop-reach-threshold", type=float, default=0.50)
    parser.add_argument("--stop-confirm", type=int, default=2)
    parser.add_argument("--phase-stop-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase-stop-threshold", type=float, default=0.55)
    parser.add_argument("--visual-target-pixel-threshold", type=int, default=5)
    parser.add_argument(
        "--visual-goal-hint-pixel-threshold",
        type=int,
        default=5,
        help=(
            "Low policy-RGB target-pixel threshold used only to generate/rank visual-goal action proposals. "
            "Discovery and success are still judged by --visual-target-pixel-threshold and close/stop thresholds."
        ),
    )
    parser.add_argument(
        "--visual-goal-hint-memory-steps",
        type=int,
        default=160,
        help=(
            "Keep using the last learner-only visual-goal bearing for this many steps after a weak target cue "
            "temporarily leaves the policy RGB. This supports occlusion recovery without target_xy."
        ),
    )
    parser.add_argument(
        "--visual-goal-hint-memory-min-pixels",
        type=int,
        default=20,
        help="Minimum target-color pixels required before weak visual-goal bearing is trusted after it disappears.",
    )
    parser.add_argument("--visual-stop-pixel-threshold", type=int, default=TARGET_PIXEL_THRESHOLD)
    parser.add_argument(
        "--visual-close-pixel-threshold",
        type=int,
        default=None,
        help=(
            "Policy-RGB target-pixel count required as close evidence. Defaults to a resolution-scaled "
            "2% of the combined front+rear policy RGB area."
        ),
    )
    parser.add_argument("--selector-direct-margin", type=float, default=0.02)
    parser.add_argument("--visual-discovery-stop-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-search-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--visual-overshoot-stop-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the target has reached close visual scale, force stop when target pixels drop from their peak "
            "and the learner already shows stop intent. Uses only policy RGB + learner outputs."
        ),
    )
    parser.add_argument("--motion-smoothing", type=float, default=0.35)
    parser.add_argument(
        "--search-mpc-route-planner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the route-planner branch inside search MPC when obstacles exist.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override the recorded horizon for evaluation only.")
    parser.add_argument(
        "--success-radius-override",
        type=float,
        default=None,
        help="Metrics-only success radius override. This is never passed to the policy observation.",
    )
    parser.add_argument("--policy-width", type=int, default=None)
    parser.add_argument("--policy-height", type=int, default=None)
    parser.add_argument("--width", type=int, default=640, help="Third-person demo video width; does not affect policy input.")
    parser.add_argument("--height", type=int, default=360, help="Third-person demo video height; does not affect policy input.")
    parser.add_argument("--demo-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-camera-smoothing", type=float, default=0.08)
    parser.add_argument("--policy-front-output", type=Path, default=None, help="Optional video of the exact front RGB stream fed to the policy.")
    parser.add_argument("--policy-rear-output", type=Path, default=None, help="Optional video of the exact rear RGB stream fed to the policy.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print one progress line every N sim steps; 0 disables.")
    return parser.parse_args()


def object_specs(metadata: dict) -> list[ObjectSpec]:
    catalog = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    result = []
    for item in metadata["objects"]:
        label, kind, rgba = catalog[item["name"]]
        size = tuple(item.get("size", (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)))
        result.append(
            ObjectSpec(
                item["name"],
                label,
                kind,
                tuple(item["position"]),
                tuple(item.get("rgba", rgba)),
                size,
            )
        )
    return result


def obstacle_specs(metadata: dict) -> list[ObstacleSpec]:
    result = []
    for item in metadata.get("obstacles", []):
        result.append(
            ObstacleSpec(
                name=str(item["name"]),
                kind=str(item.get("kind", "box")),
                position=tuple(item["position"]),
                size=tuple(item["size"]),
                rgba=tuple(item.get("rgba", (0.30, 0.33, 0.36, 1.0))),
            )
        )
    return result


def obstacle_clearance_xy(point_xy: np.ndarray, obstacle: ObstacleSpec) -> float:
    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(obstacle.position[:2], dtype=np.float64)
    half = np.asarray(obstacle.size[:2], dtype=np.float64)
    delta = np.maximum(np.abs(point - center) - half, 0.0)
    return float(np.hypot(delta[0], delta[1]))


def target_visible_from(
    point_xy: np.ndarray,
    target_xy: np.ndarray,
    obstacles: list[ObstacleSpec],
    padding: float = 0.04,
) -> bool:
    return not any(
        obstacle_blocks_segment(point_xy, target_xy, obstacle, padding=padding)
        for obstacle in obstacles
    )


def minimum_obstacle_clearance(point_xy: np.ndarray, obstacles: list[ObstacleSpec]) -> float | None:
    if not obstacles:
        return None
    return min(obstacle_clearance_xy(point_xy, obstacle) for obstacle in obstacles)


def scene_light_spec(metadata: dict) -> SceneLightSpec | None:
    item = metadata.get("scene_light") or {}
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


def run_episode(args: argparse.Namespace) -> dict:
    if not args.checkpoint.is_file() or not args.episode_json.is_file() or not ASSET.is_file():
        raise FileNotFoundError("checkpoint, episode JSON, or MuJoCo asset is missing")
    if not 0.0 < args.motion_smoothing <= 1.0 or not 0.0 < args.stop_threshold < 1.0 or args.stop_confirm <= 0:
        raise ValueError("invalid VLA replay thresholds")
    metadata = json.loads(args.episode_json.read_text(encoding="utf-8"))
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    metadata_success_radius = float(metadata.get("success_radius", 0.70))
    success_radius_override = getattr(args, "success_radius_override", None)
    success_radius = (
        metadata_success_radius
        if success_radius_override is None
        else float(success_radius_override)
    )
    if success_radius <= 0.0:
        raise ValueError("success radius must be positive")
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    checkpoint_resolution = checkpoint.get("config", {}).get("policy_input_resolution", {}) or {}
    policy_width_arg = getattr(args, "policy_width", None)
    policy_height_arg = getattr(args, "policy_height", None)
    policy_width = int(policy_width_arg or checkpoint_resolution.get("width", IMAGE_WIDTH))
    policy_height = int(policy_height_arg or checkpoint_resolution.get("height", IMAGE_HEIGHT))
    demo_width = int(getattr(args, "width", 640))
    demo_height = int(getattr(args, "height", 360))
    demo_video_enabled = bool(getattr(args, "demo_video", True))
    demo_camera_smoothing = float(getattr(args, "demo_camera_smoothing", 0.08))
    policy_front_output = getattr(args, "policy_front_output", None)
    policy_rear_output = getattr(args, "policy_rear_output", None)
    progress_interval = int(getattr(args, "progress_interval", 0) or 0)
    if min(policy_width, policy_height, demo_width, demo_height) <= 0:
        raise ValueError("policy and demo video resolutions must be positive")
    if not 0.0 < demo_camera_smoothing <= 1.0:
        raise ValueError("demo camera smoothing must be in (0, 1]")
    stop_visible_threshold = float(getattr(args, "stop_visible_threshold", 0.50))
    stop_reach_threshold = float(getattr(args, "stop_reach_threshold", 0.50))
    visual_target_pixel_threshold = int(getattr(args, "visual_target_pixel_threshold", TARGET_PIXEL_THRESHOLD))
    visual_goal_hint_pixel_threshold = int(getattr(args, "visual_goal_hint_pixel_threshold", 5))
    visual_goal_hint_memory_steps = int(getattr(args, "visual_goal_hint_memory_steps", 160))
    visual_goal_hint_memory_min_pixels = int(getattr(args, "visual_goal_hint_memory_min_pixels", 20))
    visual_stop_pixel_threshold = int(getattr(args, "visual_stop_pixel_threshold", TARGET_PIXEL_THRESHOLD))
    visual_close_pixel_threshold_arg = getattr(args, "visual_close_pixel_threshold", None)
    if visual_close_pixel_threshold_arg is None:
        visual_close_pixel_threshold = max(
            visual_stop_pixel_threshold,
            int(round(0.020 * float(policy_width * policy_height * 2))),
        )
    else:
        visual_close_pixel_threshold = int(visual_close_pixel_threshold_arg)
    selector_direct_margin = float(getattr(args, "selector_direct_margin", 0.02))
    trajectory_scorer_mode = str(getattr(args, "trajectory_scorer_mode", "selector"))
    trajectory_scorer_interval = int(getattr(args, "trajectory_scorer_interval", 1))
    if not 0.0 <= stop_visible_threshold <= 1.0 or not 0.0 <= stop_reach_threshold <= 1.0:
        raise ValueError("invalid trajectory scorer gate thresholds")
    if visual_target_pixel_threshold <= 0:
        raise ValueError("visual target pixel threshold must be positive")
    if visual_goal_hint_pixel_threshold <= 0:
        raise ValueError("visual goal hint pixel threshold must be positive")
    if visual_goal_hint_memory_steps < 0:
        raise ValueError("visual goal hint memory steps must be non-negative")
    if visual_goal_hint_memory_min_pixels < visual_goal_hint_pixel_threshold:
        raise ValueError("visual goal hint memory min pixels must be >= visual goal hint pixel threshold")
    if visual_stop_pixel_threshold < visual_target_pixel_threshold:
        raise ValueError("visual stop pixel threshold must be >= visual target pixel threshold")
    if visual_close_pixel_threshold < visual_stop_pixel_threshold:
        raise ValueError("visual close pixel threshold must be >= visual stop pixel threshold")
    if selector_direct_margin < 0.0:
        raise ValueError("selector direct margin must be non-negative")
    if trajectory_scorer_interval <= 0:
        raise ValueError("trajectory scorer interval must be positive")
    if progress_interval < 0:
        raise ValueError("progress interval must be non-negative")
    if args.trajectory_scorer is not None and not Path(args.trajectory_scorer).is_file():
        raise FileNotFoundError("trajectory scorer checkpoint path is invalid")
    if args.decision_mode == "belief_guided" and args.trajectory_scorer is None:
        raise ValueError("belief_guided decision mode requires --trajectory-scorer")
    policy = M20MuJoCoVLA.for_state_dict(state_dict).to(device)
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    checkpoint_has_phase = "phase_head.weight" in state_dict and "phase_action_bias.weight" in state_dict
    scorer = None
    scorer_horizon = None
    scorer_head_order = None
    scorer_belief_dim = 0
    if args.trajectory_scorer is not None:
        scorer_checkpoint = torch.load(Path(args.trajectory_scorer), map_location=device, weights_only=True)
        scorer_horizon = int(scorer_checkpoint.get("config", {}).get("horizon", 24))
        scorer_head_dim = infer_scorer_head_dim(scorer_checkpoint)
        scorer_belief_dim = infer_scorer_belief_dim(scorer_checkpoint)
        scorer_head_order = scorer_head_order_from_checkpoint(scorer_checkpoint, scorer_head_dim)
        scorer = M20TrajectoryScorer(
            latent_dim=int(scorer_checkpoint.get("config", {}).get("latent_dim", 192)),
            horizon=scorer_horizon,
            action_dim=int(scorer_checkpoint.get("config", {}).get("action_dim", 4)),
            head_dim=scorer_head_dim,
            belief_dim=scorer_belief_dim,
        ).to(device)
        scorer.load_state_dict(scorer_checkpoint["model_state_dict"])
        scorer.eval()
    obstacles = obstacle_specs(metadata)
    scene_path = args.output.with_suffix(".scene.xml")
    terrain_profile = str(metadata.get("terrain_profile", "flat"))
    terrain = None if terrain_profile in {"", "flat"} else TerrainSpec(terrain_profile)
    build_scene(
        scene_path,
        object_specs(metadata),
        obstacles=obstacles,
        task_object_collisions=bool(metadata.get("task_object_collisions", True)),
        terrain=terrain,
        light=scene_light_spec(metadata),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model)
    search_planner = None
    if args.decision_mode in {"search_mpc", "phase_routed"}:
        search_planner = SearchMPCPlanner(
            model,
            target_xy,
            obstacles,
            SearchMPCConfig(use_route_planner_when_obstacles=bool(getattr(args, "search_mpc_route_planner", False))),
        )
    initial_yaw = float(metadata.get("initial_yaw", math.atan2(target_xy[1], target_xy[0])))
    initial_xy = tuple(metadata.get("initial_xy", (0.0, 0.0)))
    controller.reset(data, initial_yaw, initial_xy)
    # Match the collection state distribution before asking the VLA for its
    # first action. The warmup is fixed PD standing, not a target controller.
    for _ in range(int(metadata["warmup_steps"])):
        controller.step(data, np.zeros(4, dtype=np.float64))
    # Keep the policy observation contract fixed to the training resolution.
    # Human-facing video is rendered by a separate high-resolution renderer so
    # changing demo video settings cannot silently change VLA inputs.
    policy_renderer = mujoco.Renderer(model, height=policy_height, width=policy_width)
    demo_renderer = mujoco.Renderer(model, height=demo_height, width=demo_width) if demo_video_enabled else None
    third_camera = mujoco.MjvCamera()
    third_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    third_camera.distance = 3.1
    third_camera.azimuth = 138.0
    third_camera.elevation = -24.0
    smoothed_demo_lookat = np.asarray(data.qpos[:3], dtype=np.float64).copy()
    smoothed_demo_lookat[2] = 0.50
    writer = open_video(args.output, demo_width, demo_height) if demo_video_enabled else None
    front_writer = open_video(Path(policy_front_output), policy_width, policy_height) if policy_front_output else None
    rear_writer = open_video(Path(policy_rear_output), policy_width, policy_height) if policy_rear_output else None
    metrics_path = args.metrics or args.output.with_suffix(".json")
    previous_motion = np.zeros(3, dtype=np.float64)
    previous_executed_action = np.zeros(4, dtype=np.float32)
    history_tracker = VisualHistoryTracker(pixel_threshold=visual_target_pixel_threshold)
    stop_votes = 0
    stop_latched = False
    stop_latched_step = None
    min_distance = float("inf")
    min_height = float(data.qpos[2])
    target_first_visible_step = -1
    target_visible_steps = 0
    visual_stop_ready_step = -1
    visual_close_ready_step = -1
    visual_close_peak_count = 0
    visual_close_peak_step = -1
    geometric_target_first_visible_step = -1
    geometric_target_visible_steps = 0
    target_rgb_pixel_counts = []
    visual_goal_hint_steps = 0
    visual_goal_hint_first_step = -1
    visual_goal_hint_pixel_counts: list[int] = []
    visual_goal_memory_steps = 0
    last_visual_goal_hint_step = -1
    last_visual_goal_bearing: float | None = None
    last_visual_goal_distance: float | None = None
    last_visual_goal_visible_prob: float | None = None
    last_visual_goal_pixel_count = 0
    target_reached_step = -1
    min_obstacle_clearance = minimum_obstacle_clearance(np.asarray(data.qpos[:2], dtype=np.float64), obstacles)
    executed = []
    probabilities = []
    selected_candidate_indices = []
    selected_candidate_scores = []
    selected_candidate_histogram: dict[str, int] = {}
    selected_candidate_count = None
    gate_visible_probs = []
    gate_reach_probs = []
    gate_safe_probs = []
    gate_progress_probs = []
    gate_final_close_probs = []
    gate_visibility_keep_probs = []
    gate_smooth_probs = []
    gate_stop_allowed = []
    selector_stop_evidences = []
    selector_stop_allowed = []
    selector_stop_candidate_counts = []
    selector_forced_stop_count = 0
    selector_forced_stop_steps = []
    guarded_selector_eligible_counts = []
    guarded_selector_override_count = 0
    guarded_selector_rejected_best_count = 0
    trajectory_scorer_call_count = 0
    trajectory_scorer_cached_count = 0
    cached_scorer_prediction: np.ndarray | None = None
    cached_scorer_stop_probability = 0.0
    phase_stop_probs = []
    phase_predictions = []
    belief_bearings = []
    belief_distances = []
    belief_visible_probs = []
    visual_search_guard_count = 0
    visual_servo_guard_count = 0
    visual_goal_hint_count = 0
    visual_goal_hint_bearings: list[float] = []
    visual_goal_hint_distances: list[float] = []
    visual_stop_block_count = 0
    visual_overshoot_stop_guard_count = 0
    visual_overshoot_stop_guard_step = -1
    search_mpc_planner_info: dict[str, float | list[float]] | None = None
    phase_routed_search_steps = 0
    phase_routed_policy_steps = 0
    try:
        evaluation_steps = int(args.steps or metadata["steps"])
        for step in range(evaluation_steps):
            if progress_interval and step % progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "progress": "m20_vla_episode",
                            "step": int(step),
                            "steps": int(evaluation_steps),
                            "target": str(metadata.get("target_label", "")),
                            "stop_latched": bool(stop_latched),
                            "min_distance": None if not np.isfinite(min_distance) else float(min_distance),
                            "min_height": None if not np.isfinite(min_height) else float(min_height),
                        }
                    ),
                    flush=True,
                )
            base_xy_before_action = np.asarray(data.qpos[:2], dtype=np.float64)
            geometric_target_visible_now = target_visible_from(base_xy_before_action, target_xy, obstacles)
            if geometric_target_visible_now:
                geometric_target_visible_steps += 1
                if geometric_target_first_visible_step < 0:
                    geometric_target_first_visible_step = step
            policy_renderer.update_scene(data, camera="front_rgb")
            front = policy_renderer.render().copy()
            if front_writer is not None:
                front_writer.stdin.write(front.tobytes())
            policy_renderer.update_scene(data, camera="rear_rgb")
            rear = policy_renderer.render().copy()
            if rear_writer is not None:
                rear_writer.stdin.write(rear.tobytes())
            target_rgb = target_rgb_observation(
                front,
                rear,
                str(metadata["target_label"]),
                pixel_threshold=visual_target_pixel_threshold,
            )
            target_rgb_pixel_counts.append(int(target_rgb["total_count"]))
            visual_goal_hint_now = (
                target_rgb.get("bearing") is not None
                and int(target_rgb["total_count"]) >= visual_goal_hint_pixel_threshold
            )
            if visual_goal_hint_now:
                visual_goal_hint_steps += 1
                visual_goal_hint_pixel_counts.append(int(target_rgb["total_count"]))
                if visual_goal_hint_first_step < 0:
                    visual_goal_hint_first_step = step
            current_visual_goal_bearing = None
            current_visual_goal_distance = None
            current_visual_goal_visible_prob = None
            if visual_goal_hint_now:
                current_visual_goal_bearing = float(target_rgb["bearing"])
                current_visual_goal_distance = visual_goal_distance_from_pixels(
                    int(target_rgb["total_count"]),
                    close_pixel_threshold=visual_close_pixel_threshold,
                )
                current_visual_goal_visible_prob = visual_goal_visible_prob_from_pixels(
                    int(target_rgb["total_count"]),
                    visible_pixel_threshold=visual_goal_hint_pixel_threshold,
                    close_pixel_threshold=visual_close_pixel_threshold,
                )
                last_visual_goal_hint_step = step
                last_visual_goal_bearing = current_visual_goal_bearing
                last_visual_goal_distance = current_visual_goal_distance
                last_visual_goal_visible_prob = current_visual_goal_visible_prob
                last_visual_goal_pixel_count = int(target_rgb["total_count"])
            visual_goal_memory_active = (
                not visual_goal_hint_now
                and visual_goal_hint_memory_steps > 0
                and last_visual_goal_hint_step >= 0
                and step - last_visual_goal_hint_step <= visual_goal_hint_memory_steps
                and last_visual_goal_bearing is not None
                and last_visual_goal_pixel_count >= visual_goal_hint_memory_min_pixels
            )
            if visual_goal_memory_active:
                visual_goal_memory_steps += 1
            target_visible_now = bool(target_rgb["visible"])
            if target_visible_now:
                target_visible_steps += 1
                if target_first_visible_step < 0:
                    target_first_visible_step = step
            if int(target_rgb["total_count"]) >= visual_stop_pixel_threshold and visual_stop_ready_step < 0:
                visual_stop_ready_step = step
            if int(target_rgb["total_count"]) >= visual_close_pixel_threshold and visual_close_ready_step < 0:
                visual_close_ready_step = step
            if int(target_rgb["total_count"]) > visual_close_peak_count:
                visual_close_peak_count = int(target_rgb["total_count"])
                visual_close_peak_step = step
            visual_target_discovered = target_first_visible_step >= 0
            visual_stop_ready_latched = visual_stop_ready_step >= 0
            visual_close_ready_latched = visual_close_ready_step >= 0
            lidar_values = planar_lidar(model, data)
            rgb = np.concatenate((front, rear), axis=-1).transpose(2, 0, 1)
            history_feature = history_tracker.observe(
                front,
                rear,
                str(metadata["target_label"]),
                previous_action=previous_executed_action,
            )
            batch = {
                "rgb": torch.from_numpy(rgb.copy()).unsqueeze(0).to(device).float().div_(255.0),
                "lidar": torch.from_numpy(lidar_values).unsqueeze(0).to(device).float().div_(10.0),
                "proprio": torch.from_numpy(proprioception(model, data)).unsqueeze(0).to(device).float(),
                "history": torch.from_numpy(history_feature.copy()).unsqueeze(0).to(device).float(),
                "language": torch.from_numpy(encode_text(metadata["task_text"])).unsqueeze(0).to(device),
            }
            with torch.no_grad():
                phase_stop_prob = 0.0
                scorer_ran_this_step = False
                use_search_planner = args.decision_mode == "search_mpc" or (
                    args.decision_mode == "phase_routed" and not visual_target_discovered
                )
                if use_search_planner:
                    if args.decision_mode == "phase_routed":
                        phase_routed_search_steps += 1
                    assert search_planner is not None
                    prediction, planner_info = search_planner.recommend(data, controller)
                    search_mpc_planner_info = planner_info
                    stop_probability = 0.99 if float(prediction[3]) >= 0.5 else 0.0
                else:
                    if args.decision_mode == "phase_routed":
                        phase_routed_policy_steps += 1
                    latent = policy.encode(
                        batch["rgb"],
                        batch["lidar"],
                        batch["proprio"],
                        batch["language"],
                        history=batch["history"],
                    )
                    target_logits = policy.target_head(latent)
                    phase_logits = policy.phase_head(latent)
                    phase_prob = torch.softmax(phase_logits, dim=-1)
                    phase_stop_prob = float(phase_prob[0, 2].item())
                    phase_predictions.append(int(phase_prob.argmax(dim=-1).item()))
                    phase_stop_probs.append(phase_stop_prob)
                    target_context = policy._target_context_from_logits(target_logits)
                    direct_prediction = policy.action_from_latent(
                        latent,
                        target_context,
                        phase_logits=phase_logits,
                        history=batch["history"],
                    )[0].detach().cpu().numpy()
                    prediction = direct_prediction.copy()
                    stop_probability = float(1.0 / (1.0 + np.exp(-direct_prediction[3])))
                    belief_bearing = None
                    belief_distance = None
                    belief_visible_prob = None
                    if scorer is not None and scorer_horizon is not None:
                        if scorer_belief_dim >= 4:
                            belief_raw = scorer.predict_belief(latent)[0].detach().cpu().numpy()
                            belief_sin = float(np.clip(belief_raw[0], -1.0, 1.0))
                            belief_cos = float(np.clip(belief_raw[1], -1.0, 1.0))
                            norm = float(np.hypot(belief_sin, belief_cos))
                            if norm >= 1.0e-6:
                                belief_sin /= norm
                                belief_cos /= norm
                            belief_bearing = float(np.arctan2(belief_sin, belief_cos))
                            belief_distance = float(np.clip(belief_raw[2], 0.0, 1.0) * 4.0)
                            belief_visible_prob = float(1.0 / (1.0 + np.exp(-belief_raw[3])))
                            belief_bearings.append(belief_bearing)
                            belief_distances.append(belief_distance)
                            belief_visible_probs.append(belief_visible_prob)
                        proposal_bearing = belief_bearing
                        proposal_distance = belief_distance
                        proposal_visible_prob = belief_visible_prob
                        visual_hint_for_proposal = bool(visual_goal_hint_now or visual_goal_memory_active)
                        if visual_hint_for_proposal:
                            if visual_goal_hint_now:
                                visual_bearing = float(current_visual_goal_bearing)
                                visual_distance = float(current_visual_goal_distance)
                                visual_visible_prob = float(current_visual_goal_visible_prob)
                            else:
                                visual_bearing = float(last_visual_goal_bearing)
                                visual_distance = float(last_visual_goal_distance or 2.4)
                                age = max(0, step - int(last_visual_goal_hint_step))
                                decay = max(0.0, 1.0 - float(age) / max(1.0, float(visual_goal_hint_memory_steps)))
                                visual_visible_prob = max(
                                    0.20,
                                    float(last_visual_goal_visible_prob or 0.55) * decay,
                                )
                            proposal_bearing = visual_bearing
                            proposal_distance = visual_distance
                            proposal_visible_prob = max(
                                visual_visible_prob,
                                0.0 if belief_visible_prob is None else float(belief_visible_prob),
                            )
                            visual_goal_hint_count += 1
                            visual_goal_hint_bearings.append(visual_bearing)
                            visual_goal_hint_distances.append(visual_distance)
                        scorer_cache_allowed = (
                            args.decision_mode == "trajectory_scorer"
                            and trajectory_scorer_mode in {"selector", "guarded_selector"}
                            and trajectory_scorer_interval > 1
                        )
                        scorer_due = (
                            not scorer_cache_allowed
                            or cached_scorer_prediction is None
                            or step % trajectory_scorer_interval == 0
                            or target_first_visible_step == step
                            or visual_close_ready_step == step
                        )
                        if scorer_cache_allowed and not scorer_due and cached_scorer_prediction is not None:
                            prediction = cached_scorer_prediction.copy()
                            stop_probability = float(cached_scorer_stop_probability)
                            trajectory_scorer_cached_count += 1
                        elif args.decision_mode == "belief_guided":
                            if belief_bearing is None or belief_distance is None or belief_visible_prob is None:
                                raise RuntimeError("belief_guided decision mode requires a trajectory scorer checkpoint with belief_dim >= 4")
                            prediction = command_from_goal_belief(
                                bearing=belief_bearing,
                                distance=belief_distance,
                                visible_prob=belief_visible_prob,
                            )
                            stop_probability = 0.99 if float(prediction[3]) >= 0.5 else 0.0
                        else:
                            scorer_ran_this_step = True
                            trajectory_scorer_call_count += 1
                            candidate_chunks = make_phase_action_chunks(
                                prediction,
                                horizon=scorer_horizon,
                                phase_prob=phase_prob[0].detach().cpu().numpy(),
                                goal_bearing=proposal_bearing,
                                goal_distance=proposal_distance,
                                goal_visible_prob=proposal_visible_prob,
                            )
                            candidate_tensor = torch.from_numpy(candidate_chunks).to(device)
                            selected_candidate_count = int(candidate_tensor.shape[0])
                            latent_batch = latent.expand(candidate_tensor.shape[0], -1)
                            candidate_scores_raw = scorer(latent_batch, candidate_tensor)
                            candidate_scores = scorer_combined_score(candidate_scores_raw)
                            if visual_hint_for_proposal and proposal_bearing is not None:
                                alignment_prior = visual_goal_alignment_prior(
                                    candidate_tensor,
                                    goal_bearing=proposal_bearing,
                                    goal_distance=proposal_distance,
                                    goal_visible_prob=proposal_visible_prob,
                                )
                                # Current-frame target pixels are more reliable
                                # than stale memory.  Give them enough weight to
                                # stop the selector from driving past a newly
                                # visible edge target; stale memory remains a
                                # weak hint and still needs safe/progress heads.
                                prior_weight = 0.22 if visual_goal_hint_now else 0.18
                                candidate_scores = candidate_scores + prior_weight * alignment_prior
                            if trajectory_scorer_mode in {"selector", "guarded_selector"}:
                                # Use the scorer to rank chunks, but do not let the weak direct
                                # action stop logit suppress every stop candidate.  In practice
                                # the phase head and the learned belief head become reliable
                                # stop evidence earlier than the direct scalar action head.
                                active_head_order = scorer_head_order or list(LEGACY_SCORER_HEAD_ORDER)
                                candidate_probabilities = torch.sigmoid(candidate_scores_raw)

                                def head_vector(name: str, fallback: float | torch.Tensor) -> torch.Tensor:
                                    if name in active_head_order:
                                        index = active_head_order.index(name)
                                        if index < int(candidate_probabilities.shape[-1]):
                                            return candidate_probabilities[:, index]
                                    if isinstance(fallback, torch.Tensor):
                                        return fallback
                                    return torch.full_like(candidate_scores, float(fallback))

                                stop_candidate_mask = candidate_tensor[:, 0, 3] >= 0.5
                                future_stop_candidate_mask = candidate_tensor[:, :, 3].amax(dim=1) >= 0.5
                                candidate_index = torch.arange(candidate_tensor.shape[0], device=device)
                                direct_candidate_mask = candidate_index == 0
                                non_direct_candidate_mask = ~direct_candidate_mask
                                stationary_nonstop_mask = (
                                    torch.linalg.norm(candidate_tensor[:, 0, :3], dim=1) < 1.0e-5
                                ) & (~stop_candidate_mask)
                                stop_gate_threshold = max(0.35, 0.70 * args.stop_threshold)
                                visual_stop_ready = (
                                    target_visible_now
                                    and int(target_rgb["total_count"]) >= visual_stop_pixel_threshold
                                )
                                visual_close_now = (
                                    target_visible_now
                                    and int(target_rgb["total_count"]) >= visual_close_pixel_threshold
                                )
                                can_stop_after_visual = (
                                    not bool(args.visual_discovery_stop_gate)
                                    or visual_close_ready_latched
                                )
                                belief_close_visible = (
                                    belief_distance is not None
                                    and belief_visible_prob is not None
                                    and belief_distance <= 0.45
                                    and belief_visible_prob >= stop_visible_threshold
                                )
                                phase_stop_evidence = (
                                    phase_stop_prob
                                    if (
                                        can_stop_after_visual
                                        and checkpoint_has_phase
                                        and phase_stop_prob >= float(args.phase_stop_threshold)
                                    )
                                    else 0.0
                                )
                                belief_stop_evidence = (
                                    float(belief_visible_prob)
                                    if can_stop_after_visual and belief_close_visible
                                    else 0.0
                                )
                                stop_evidence = max(stop_probability, phase_stop_evidence, belief_stop_evidence)
                                allow_future_stop = stop_evidence >= stop_gate_threshold and visual_close_now
                                selector_stop_evidences.append(float(stop_evidence))
                                selector_stop_allowed.append(bool(allow_future_stop))
                                selector_stop_candidate_counts.append(int(future_stop_candidate_mask.sum().item()))
                                selection_scores = candidate_scores.clone()
                                if not allow_future_stop:
                                    selection_scores = selection_scores.masked_fill(
                                        future_stop_candidate_mask & non_direct_candidate_mask,
                                        -1.0e9,
                                    )
                                elif (
                                    can_stop_after_visual
                                    and checkpoint_has_phase
                                    and phase_stop_prob >= float(args.phase_stop_threshold)
                                    and belief_close_visible
                                    and visual_close_now
                                ):
                                    # Once both learner-only heads agree that the target is close
                                    # and visible, bias the selector toward stopping instead of
                                    # overshooting.  The world model still chooses among candidates.
                                    selection_scores = selection_scores + stop_candidate_mask.float() * 0.24
                                    selection_scores = selection_scores + (
                                        future_stop_candidate_mask & (~stop_candidate_mask)
                                    ).float() * 0.08
                                selection_scores = selection_scores.masked_fill(
                                    stationary_nonstop_mask & non_direct_candidate_mask,
                                    -1.0e9,
                                )
                                direct_candidate_score = float(selection_scores[0].item())
                                best_index = int(selection_scores.argmax(dim=0).item())
                                best_score = float(selection_scores[best_index].item())
                                guarded_margin = selector_direct_margin
                                if trajectory_scorer_mode == "guarded_selector":
                                    visible_probs = head_vector("visible", 0.0)
                                    reach_probs = head_vector("reach", 0.0)
                                    safe_probs = head_vector("safe", 1.0)
                                    progress_probs = head_vector("progress", 0.0)
                                    final_close_probs = head_vector("final_close", reach_probs)
                                    visibility_keep_probs = head_vector("visibility_keep", visible_probs)
                                    smooth_probs = head_vector("smooth", safe_probs)
                                    goal_probs = torch.maximum(reach_probs, final_close_probs)
                                    direct_goal = goal_probs[0]
                                    direct_progress = progress_probs[0]
                                    direct_visible = visible_probs[0]
                                    direct_visibility_keep = visibility_keep_probs[0]
                                    visual_goal_active_now = (
                                        visual_goal_hint_now or visual_goal_memory_active
                                    )
                                    if visual_goal_active_now:
                                        safe_floor = max(0.58, float(safe_probs[0].item()) - 0.10)
                                        smooth_floor = max(0.42, float(smooth_probs[0].item()) - 0.12)
                                    else:
                                        safe_floor = max(0.66, float(safe_probs[0].item()) - 0.03)
                                        smooth_floor = max(0.50, float(smooth_probs[0].item()) - 0.05)
                                    safe_enough = safe_probs >= safe_floor
                                    smooth_enough = smooth_probs >= smooth_floor
                                    if visual_target_discovered:
                                        goal_improves = (
                                            (goal_probs >= direct_goal + 0.08)
                                            | (
                                                (progress_probs >= direct_progress + 0.12)
                                                & (goal_probs >= direct_goal - 0.05)
                                            )
                                            | (
                                                (visibility_keep_probs >= direct_visibility_keep + 0.15)
                                                & (progress_probs >= direct_progress + 0.05)
                                            )
                                            | (
                                                visual_goal_active_now
                                                & (goal_probs >= direct_goal - 0.08)
                                                & (progress_probs >= direct_progress - 0.04)
                                                & (visibility_keep_probs >= direct_visibility_keep - 0.12)
                                            )
                                        )
                                    else:
                                        goal_improves = (
                                            (visible_probs >= direct_visible + 0.15)
                                            & (progress_probs >= direct_progress - 0.03)
                                        ) | (
                                            (progress_probs >= direct_progress + 0.15)
                                            & (visible_probs >= direct_visible - 0.05)
                                        )
                                    stop_candidate_ready = (
                                        allow_future_stop
                                        & stop_candidate_mask
                                        & (goal_probs >= torch.maximum(direct_goal - 0.02, torch.tensor(0.55, device=device)))
                                    )
                                    guarded_eligible = (
                                        non_direct_candidate_mask
                                        & safe_enough
                                        & smooth_enough
                                        & (goal_improves | stop_candidate_ready)
                                        & ((~future_stop_candidate_mask) | allow_future_stop)
                                    )
                                    guarded_selector_eligible_counts.append(int(guarded_eligible.sum().item()))
                                    raw_best_index = best_index
                                    selection_scores = selection_scores.masked_fill(
                                        (~guarded_eligible) & non_direct_candidate_mask,
                                        -1.0e9,
                                    )
                                    guarded_margin = max(
                                        selector_direct_margin,
                                        0.01 if visual_target_discovered and visual_goal_active_now else (0.04 if visual_target_discovered else 0.12),
                                    )
                                    best_index = int(selection_scores.argmax(dim=0).item())
                                    best_score = float(selection_scores[best_index].item())
                                    if raw_best_index != 0 and best_index == 0:
                                        guarded_selector_rejected_best_count += 1
                                selected_index = (
                                    best_index
                                    if (best_index != 0 and best_score > direct_candidate_score + guarded_margin)
                                    else 0
                                )
                                if trajectory_scorer_mode == "guarded_selector" and selected_index != 0:
                                    guarded_selector_override_count += 1
                                selected_candidate_indices.append(selected_index)
                                selected_candidate_histogram[str(selected_index)] = selected_candidate_histogram.get(str(selected_index), 0) + 1
                                selected_candidate_scores.append(float(selection_scores[selected_index].item()))
                                prediction = candidate_chunks[selected_index, 0].copy()
                                if selected_index != 0:
                                    if float(prediction[3]) >= 0.5:
                                        stop_probability = 0.99
                                    else:
                                        stop_probability = min(stop_probability, 0.05)
                                strong_learner_stop = (
                                    can_stop_after_visual
                                    and checkpoint_has_phase
                                    and phase_stop_prob >= float(args.phase_stop_threshold)
                                    and visual_close_now
                                    and belief_close_visible
                                )
                                if strong_learner_stop:
                                    prediction = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
                                    stop_probability = 0.99
                                    selector_forced_stop_count += 1
                                    if len(selector_forced_stop_steps) < 20:
                                        selector_forced_stop_steps.append(int(step))
                            else:
                                active_head_order = scorer_head_order or list(LEGACY_SCORER_HEAD_ORDER)
                                visible_value = scorer_probability(candidate_scores_raw, active_head_order, "visible", default=0.0)
                                reach_value = scorer_probability(candidate_scores_raw, active_head_order, "reach", default=0.0)
                                safe_value = scorer_probability(candidate_scores_raw, active_head_order, "safe", default=1.0)
                                gate_visible_prob = 0.0 if visible_value is None else float(visible_value)
                                gate_reach_prob = 0.0 if reach_value is None else float(reach_value)
                                gate_safe_prob = 1.0 if safe_value is None else float(safe_value)
                                gate_progress_prob = scorer_probability(candidate_scores_raw, active_head_order, "progress", default=None)
                                gate_final_close_prob = scorer_probability(candidate_scores_raw, active_head_order, "final_close", default=None)
                                gate_visibility_keep_prob = scorer_probability(candidate_scores_raw, active_head_order, "visibility_keep", default=None)
                                gate_smooth_prob = scorer_probability(candidate_scores_raw, active_head_order, "smooth", default=None)
                                gate_closeness_prob = max(
                                    gate_reach_prob,
                                    gate_final_close_prob if gate_final_close_prob is not None else gate_reach_prob,
                                )
                                gate_allow_stop = (
                                    stop_probability >= args.stop_threshold
                                    and gate_visible_prob >= stop_visible_threshold
                                    and gate_closeness_prob >= stop_reach_threshold
                                    and gate_safe_prob >= 0.40
                                )
                                gate_visible_probs.append(gate_visible_prob)
                                gate_reach_probs.append(gate_reach_prob)
                                gate_safe_probs.append(gate_safe_prob)
                                if gate_progress_prob is not None:
                                    gate_progress_probs.append(float(gate_progress_prob))
                                if gate_final_close_prob is not None:
                                    gate_final_close_probs.append(float(gate_final_close_prob))
                                if gate_visibility_keep_prob is not None:
                                    gate_visibility_keep_probs.append(float(gate_visibility_keep_prob))
                                if gate_smooth_prob is not None:
                                    gate_smooth_probs.append(float(gate_smooth_prob))
                                gate_stop_allowed.append(gate_allow_stop)
                                if not gate_allow_stop:
                                    stop_probability = 0.0
                        if (
                            args.decision_mode == "trajectory_scorer"
                            and trajectory_scorer_mode in {"selector", "guarded_selector"}
                            and scorer_ran_this_step
                        ):
                            cached_scorer_prediction = prediction.copy()
                            cached_scorer_stop_probability = float(stop_probability)
                    if bool(args.phase_stop_gate):
                        if not checkpoint_has_phase or phase_stop_prob < float(args.phase_stop_threshold):
                            stop_probability = 0.0
                if bool(args.visual_search_guard) and args.decision_mode in {"direct", "trajectory_scorer", "belief_guided"}:
                    if not visual_target_discovered and not visual_goal_hint_now and not visual_goal_memory_active:
                        prediction = command_from_visual_search_guard(step, lidar_values)
                        stop_probability = 0.0
                        visual_search_guard_count += 1
                runtime_visual_stop_ready = (
                    visual_close_ready_latched
                )
                if bool(args.visual_discovery_stop_gate) and not runtime_visual_stop_ready:
                    if stop_probability >= args.stop_threshold or float(prediction[3]) >= 0.5:
                        prediction = prediction.copy()
                        prediction[3] = 0.0
                        stop_probability = 0.0
                        visual_stop_block_count += 1
                if (
                    bool(getattr(args, "visual_overshoot_stop_guard", True))
                    and visual_close_ready_latched
                    and not stop_latched
                    and visual_close_peak_step >= 0
                    and step >= visual_close_peak_step + 6
                    and visual_close_peak_count >= visual_close_pixel_threshold
                ):
                    current_target_count = int(target_rgb["total_count"])
                    dropped_from_peak = current_target_count <= max(
                        visual_stop_pixel_threshold,
                        int(round(0.72 * float(visual_close_peak_count))),
                    )
                    learner_stop_intent = (
                        stop_probability >= max(0.22, 0.45 * float(args.stop_threshold))
                        or phase_stop_prob >= max(0.22, 0.45 * float(args.phase_stop_threshold))
                    )
                    if dropped_from_peak and learner_stop_intent:
                        prediction = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
                        stop_probability = 0.99
                        visual_overshoot_stop_guard_count += 1
                        if visual_overshoot_stop_guard_step < 0:
                            visual_overshoot_stop_guard_step = step
                probabilities.append(stop_probability)
            stop_votes = stop_votes + 1 if stop_probability >= args.stop_threshold else 0
            if stop_votes >= args.stop_confirm and not stop_latched:
                stop_latched = True
                stop_latched_step = step
            if stop_latched:
                motion = np.zeros(3, dtype=np.float64)
                previous_motion = motion
            else:
                motion = args.motion_smoothing * prediction[:3] + (1.0 - args.motion_smoothing) * previous_motion
                previous_motion = motion
            action = np.array((motion[0], motion[1], motion[2], 1.0 if stop_latched else 0.0), dtype=np.float64)
            previous_executed_action = action.astype(np.float32, copy=True)
            # target_xy below is metrics-only. It is never added to the policy batch.
            controller.step(data, action)
            current_distance = float(np.linalg.norm(target_xy - data.qpos[:2]))
            min_distance = min(min_distance, current_distance)
            if current_distance <= success_radius and target_reached_step < 0:
                target_reached_step = step
            min_height = min(min_height, float(data.qpos[2]))
            clearance = minimum_obstacle_clearance(np.asarray(data.qpos[:2], dtype=np.float64), obstacles)
            if clearance is not None:
                min_obstacle_clearance = (
                    clearance
                    if min_obstacle_clearance is None
                    else min(min_obstacle_clearance, clearance)
                )
            executed.append(action.tolist())
            target_demo_lookat = np.asarray(data.qpos[:3], dtype=np.float64).copy()
            # Stabilize the human-facing camera: follow x/y smoothly and keep
            # the vertical target near nominal chassis height so gait/body
            # oscillation does not turn into camera shake.
            target_demo_lookat[2] = 0.50
            smoothed_demo_lookat = (
                (1.0 - demo_camera_smoothing) * smoothed_demo_lookat
                + demo_camera_smoothing * target_demo_lookat
            )
            if demo_renderer is not None and writer is not None:
                third_camera.lookat[:] = smoothed_demo_lookat
                demo_renderer.update_scene(data, camera=third_camera)
                writer.stdin.write(demo_renderer.render().tobytes())
    finally:
        if writer is not None:
            writer.stdin.close()
        if front_writer is not None:
            front_writer.stdin.close()
        if rear_writer is not None:
            rear_writer.stdin.close()
        if writer is not None and writer.wait() != 0:
            raise RuntimeError("learner-only video encoding failed")
        if front_writer is not None and front_writer.wait() != 0:
            raise RuntimeError("policy front video encoding failed")
        if rear_writer is not None and rear_writer.wait() != 0:
            raise RuntimeError("policy rear video encoding failed")
        policy_renderer.close()
        if demo_renderer is not None:
            demo_renderer.close()
        scene_path.unlink(missing_ok=True)
    final_distance = float(np.linalg.norm(target_xy - data.qpos[:2]))
    success = bool(
        stop_latched
        and target_first_visible_step >= 0
        and final_distance <= success_radius
        and min_height >= 0.45
    )
    report = {
        "schema": "m20pro_mujoco_vla_learner_only_v1",
        "checkpoint": str(args.checkpoint),
        "episode": str(args.episode_json),
        "task_text": metadata["task_text"],
        "target_label": str(metadata.get("target_label", "")),
        "initial_yaw": initial_yaw,
        "initial_xy": list(initial_xy),
        "terrain_profile": terrain_profile,
        "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "visual_history_16", "language"],
        "policy_input_resolution": {"width": policy_width, "height": policy_height},
        "demo_video_resolution": {"width": demo_width, "height": demo_height},
        "demo_camera": {
            "mode": "smoothed_third_person",
            "smoothing": demo_camera_smoothing,
            "azimuth": third_camera.azimuth,
            "elevation": third_camera.elevation,
            "distance": third_camera.distance,
        },
        "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
        "search_required": bool(metadata.get("search_required", False)),
        "initial_target_visible": bool(metadata.get("initial_target_visible", True)),
        "target_visibility_contract": "policy_rgb_color_pixels",
        "history_contract": "short memory from policy RGB target pixels + previous executed body command; no target_xy/object_id/semantic mask",
        "history_feature_labels": list(HISTORY_FEATURE_LABELS),
        "visual_target_pixel_threshold": visual_target_pixel_threshold,
        "visual_goal_hint_pixel_threshold": visual_goal_hint_pixel_threshold,
        "visual_goal_hint_memory_steps_config": visual_goal_hint_memory_steps,
        "visual_goal_hint_memory_min_pixels": visual_goal_hint_memory_min_pixels,
        "visual_stop_pixel_threshold": visual_stop_pixel_threshold,
        "visual_close_pixel_threshold": visual_close_pixel_threshold,
        "selector_direct_margin": selector_direct_margin,
        "visual_stop_ready_step": visual_stop_ready_step,
        "visual_close_ready_step": visual_close_ready_step,
        "visual_close_peak_count": int(visual_close_peak_count),
        "visual_close_peak_step": int(visual_close_peak_step),
        "target_first_visible_step": target_first_visible_step,
        "target_visible_steps": target_visible_steps,
        "target_visibility_fraction": float(target_visible_steps / evaluation_steps) if evaluation_steps > 0 else 0.0,
        "target_discovered": target_first_visible_step >= 0,
        "visual_goal_hint_first_step": int(visual_goal_hint_first_step),
        "visual_goal_hint_steps": int(visual_goal_hint_steps),
        "visual_goal_memory_steps": int(visual_goal_memory_steps),
        "last_visual_goal_hint_step": int(last_visual_goal_hint_step),
        "last_visual_goal_pixel_count": int(last_visual_goal_pixel_count),
        "visual_goal_hint_pixel_count_max": max(visual_goal_hint_pixel_counts) if visual_goal_hint_pixel_counts else None,
        "visual_goal_hint_pixel_count_mean": (
            float(np.mean(visual_goal_hint_pixel_counts)) if visual_goal_hint_pixel_counts else None
        ),
        "target_rgb_pixel_count_min": min(target_rgb_pixel_counts) if target_rgb_pixel_counts else None,
        "target_rgb_pixel_count_max": max(target_rgb_pixel_counts) if target_rgb_pixel_counts else None,
        "target_rgb_pixel_count_mean": float(np.mean(target_rgb_pixel_counts)) if target_rgb_pixel_counts else None,
        "geometric_target_first_visible_step": geometric_target_first_visible_step,
        "geometric_target_visible_steps": geometric_target_visible_steps,
        "geometric_target_visibility_fraction": (
            float(geometric_target_visible_steps / evaluation_steps) if evaluation_steps > 0 else 0.0
        ),
        "target_reached_step": target_reached_step,
        "target_discovered_before_reach": (
            target_first_visible_step >= 0
            and (target_reached_step < 0 or target_first_visible_step <= target_reached_step)
        ),
        "steps": evaluation_steps,
        "warmup_steps": int(metadata["warmup_steps"]),
        "stop_threshold": args.stop_threshold,
        "stop_confirm": args.stop_confirm,
        "trajectory_scorer_mode": trajectory_scorer_mode,
        "stop_latched": stop_latched,
        "stop_latched_step": stop_latched_step,
        "stop_probability_min": min(probabilities),
        "stop_probability_max": max(probabilities),
        "decision_mode": args.decision_mode,
        "phase_routed_visible_switch": bool(args.decision_mode == "phase_routed"),
        "phase_routed_switch_condition": "target_first_visible_step_latched" if args.decision_mode == "phase_routed" else None,
        "phase_routed_search_steps": int(phase_routed_search_steps),
        "phase_routed_policy_steps": int(phase_routed_policy_steps),
        "trajectory_scorer": str(args.trajectory_scorer) if args.trajectory_scorer else "",
        "trajectory_scorer_head_order": scorer_head_order if args.trajectory_scorer else None,
        "trajectory_scorer_belief_dim": scorer_belief_dim if args.trajectory_scorer else None,
        "trajectory_scorer_interval": int(trajectory_scorer_interval),
        "trajectory_scorer_call_count": int(trajectory_scorer_call_count),
        "trajectory_scorer_cached_count": int(trajectory_scorer_cached_count),
        "phase_labels": list(PHASE_LABELS),
        "phase_stop_gate": bool(args.phase_stop_gate),
        "visual_discovery_stop_gate": bool(args.visual_discovery_stop_gate),
        "visual_search_guard": bool(args.visual_search_guard),
        "visual_overshoot_stop_guard": bool(getattr(args, "visual_overshoot_stop_guard", True)),
        "visual_search_guard_count": int(visual_search_guard_count),
        "visual_servo_guard_count": int(visual_servo_guard_count),
        "visual_goal_hint_count": int(visual_goal_hint_count),
        "visual_goal_hint_bearing_mean_rad": (
            float(np.mean(visual_goal_hint_bearings)) if visual_goal_hint_bearings else None
        ),
        "visual_goal_hint_distance_mean_m": (
            float(np.mean(visual_goal_hint_distances)) if visual_goal_hint_distances else None
        ),
        "visual_goal_hint_distance_min_m": min(visual_goal_hint_distances) if visual_goal_hint_distances else None,
        "visual_goal_hint_distance_max_m": max(visual_goal_hint_distances) if visual_goal_hint_distances else None,
        "visual_stop_block_count": int(visual_stop_block_count),
        "visual_overshoot_stop_guard_count": int(visual_overshoot_stop_guard_count),
        "visual_overshoot_stop_guard_step": int(visual_overshoot_stop_guard_step),
        "checkpoint_has_phase": bool(checkpoint_has_phase),
        "policy_load_missing_keys": list(missing_keys),
        "policy_load_unexpected_keys": list(unexpected_keys),
        "search_mpc_planner": bool(args.decision_mode in {"search_mpc", "phase_routed"}),
        "search_mpc_planner_info": search_mpc_planner_info,
        "trajectory_scorer_selected_candidate_mean": (
            float(np.mean(selected_candidate_indices)) if selected_candidate_indices else None
        ),
        "trajectory_scorer_candidate_count": selected_candidate_count,
        "trajectory_scorer_selected_candidate_histogram": selected_candidate_histogram,
        "trajectory_scorer_selected_score_mean": (
            float(np.mean(selected_candidate_scores)) if selected_candidate_scores else None
        ),
        "trajectory_scorer_gate_visible_prob_min": min(gate_visible_probs) if gate_visible_probs else None,
        "trajectory_scorer_gate_visible_prob_max": max(gate_visible_probs) if gate_visible_probs else None,
        "trajectory_scorer_gate_reach_prob_min": min(gate_reach_probs) if gate_reach_probs else None,
        "trajectory_scorer_gate_reach_prob_max": max(gate_reach_probs) if gate_reach_probs else None,
        "trajectory_scorer_gate_safe_prob_min": min(gate_safe_probs) if gate_safe_probs else None,
        "trajectory_scorer_gate_safe_prob_max": max(gate_safe_probs) if gate_safe_probs else None,
        "trajectory_scorer_gate_progress_prob_min": min(gate_progress_probs) if gate_progress_probs else None,
        "trajectory_scorer_gate_progress_prob_max": max(gate_progress_probs) if gate_progress_probs else None,
        "trajectory_scorer_gate_final_close_prob_min": min(gate_final_close_probs) if gate_final_close_probs else None,
        "trajectory_scorer_gate_final_close_prob_max": max(gate_final_close_probs) if gate_final_close_probs else None,
        "trajectory_scorer_gate_visibility_keep_prob_min": min(gate_visibility_keep_probs) if gate_visibility_keep_probs else None,
        "trajectory_scorer_gate_visibility_keep_prob_max": max(gate_visibility_keep_probs) if gate_visibility_keep_probs else None,
        "trajectory_scorer_gate_smooth_prob_min": min(gate_smooth_probs) if gate_smooth_probs else None,
        "trajectory_scorer_gate_smooth_prob_max": max(gate_smooth_probs) if gate_smooth_probs else None,
        "trajectory_scorer_gate_stop_allowed_count": int(sum(gate_stop_allowed)),
        "trajectory_scorer_gate_stop_blocked_count": int(len(gate_stop_allowed) - sum(gate_stop_allowed)),
        "trajectory_scorer_selector_stop_evidence_min": min(selector_stop_evidences) if selector_stop_evidences else None,
        "trajectory_scorer_selector_stop_evidence_max": max(selector_stop_evidences) if selector_stop_evidences else None,
        "trajectory_scorer_selector_stop_allowed_count": int(sum(selector_stop_allowed)),
        "trajectory_scorer_selector_stop_blocked_count": int(len(selector_stop_allowed) - sum(selector_stop_allowed)),
        "trajectory_scorer_selector_stop_candidate_count_max": (
            max(selector_stop_candidate_counts) if selector_stop_candidate_counts else None
        ),
        "trajectory_scorer_selector_forced_stop_count": int(selector_forced_stop_count),
        "trajectory_scorer_selector_forced_stop_steps_first20": selector_forced_stop_steps,
        "trajectory_scorer_guarded_selector_eligible_count_mean": (
            float(np.mean(guarded_selector_eligible_counts)) if guarded_selector_eligible_counts else None
        ),
        "trajectory_scorer_guarded_selector_eligible_count_max": (
            max(guarded_selector_eligible_counts) if guarded_selector_eligible_counts else None
        ),
        "trajectory_scorer_guarded_selector_override_count": int(guarded_selector_override_count),
        "trajectory_scorer_guarded_selector_rejected_best_count": int(guarded_selector_rejected_best_count),
        "phase_stop_prob_min": min(phase_stop_probs) if phase_stop_probs else None,
        "phase_stop_prob_max": max(phase_stop_probs) if phase_stop_probs else None,
        "goal_belief_bearing_mean_rad": float(np.mean(belief_bearings)) if belief_bearings else None,
        "goal_belief_bearing_min_rad": min(belief_bearings) if belief_bearings else None,
        "goal_belief_bearing_max_rad": max(belief_bearings) if belief_bearings else None,
        "goal_belief_distance_mean_m": float(np.mean(belief_distances)) if belief_distances else None,
        "goal_belief_distance_min_m": min(belief_distances) if belief_distances else None,
        "goal_belief_distance_max_m": max(belief_distances) if belief_distances else None,
        "goal_belief_visible_prob_min": min(belief_visible_probs) if belief_visible_probs else None,
        "goal_belief_visible_prob_max": max(belief_visible_probs) if belief_visible_probs else None,
        "phase_prediction_counts": {label: int(sum(1 for value in phase_predictions if value == index)) for index, label in enumerate(PHASE_LABELS)},
        "min_target_distance": min_distance,
        "final_target_distance": final_distance,
        "success_radius": success_radius,
        "metadata_success_radius": metadata_success_radius,
        "success_radius_override": None if success_radius_override is None else float(success_radius_override),
        "final_base_xy": data.qpos[:2].tolist(),
        "min_base_height": min_height,
        "min_obstacle_clearance": min_obstacle_clearance,
        "success": success,
        "false_stop_before_discovery": bool(
            stop_latched_step is not None
            and (target_first_visible_step < 0 or stop_latched_step < target_first_visible_step)
        ),
        "video": str(args.output) if demo_video_enabled else "",
        "policy_front_video": str(policy_front_output) if policy_front_output else "",
        "policy_rear_video": str(policy_rear_output) if policy_rear_output else "",
        "executed_action_mean": np.asarray(executed, dtype=np.float64).mean(axis=0).tolist(),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run_episode(parse_args())


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

#!/usr/bin/env python3
"""Run a trained SmolVLA checkpoint in closed-loop M20 MuJoCo simulation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action

from m20pro_vla.data.lerobot_adapter import m20_smolvla_state
from m20pro_vla.data.failure_recovery import capture_state, collect_continuations
from m20pro_vla.data.visibility import target_pixel_count
from m20pro_vla.eval.acceptance import (
    CONTACT_TOLERANT_CORE_CRITERIA,
    CRITERION_NAMES,
    apply_evaluation_config,
    build_criteria,
    classify_failure,
    count_successes_by_contact_tolerance,
    resolve_policy_step_budget,
    terminate_after_stop,
)
from m20pro_vla.low_level import M20LowLevelController, build_low_level_controller
from m20pro_vla.low_level.shield import (
    DEFAULT_SAFETY_STOP_DISTANCE_M,
    DEFAULT_SUCCESS_RADIUS_M,
    lidar_safety_shield,
    validate_stop_geometry,
)
from m20pro_vla.sim.mujoco import ASSET, build_scene, open_video, planar_lidar, proprioception
from play_m20_mujoco_vla import (
    minimum_obstacle_clearance,
    object_specs,
    obstacle_specs,
    scene_light_spec,
)

ROBOT_FOOTPRINT_RADIUS_M = 0.375
WORKSPACE = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=None,
        help="Fleet mode: run every selected episode in this dataset directory and write one summary.",
    )
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--episode-ids", default=None, help="Comma-separated episode ids for fleet mode.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=WORKSPACE / "configs/experiment.json",
        help="Single experiment config holding the closed-loop acceptance thresholds.",
    )
    parser.add_argument(
        "--write-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the third-view and policy-camera videos. Disable for fleet sweeps.",
    )
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--front-output", type=Path, default=None)
    parser.add_argument("--rear-output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--policy-steps",
        type=int,
        default=None,
        help=(
            "Fallback per-episode policy-step budget. Defaults to "
            "smolvla_evaluation.policy_steps in the acceptance config. Ignored when "
            "--match-teacher-budget is on and the episode JSON carries 'steps'."
        ),
    )
    parser.add_argument(
        "--match-teacher-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Give each episode the policy-step budget its teacher demonstration "
            "actually used, read from the episode JSON 'steps' field."
        ),
    )
    parser.add_argument(
        "--post-stop-hold-steps",
        type=int,
        default=None,
        help=(
            "Settle window kept after a latched stop, in policy steps; the episode "
            "ends once it expires. Negative restores the old run-to-budget behaviour. "
            "Defaults to smolvla_evaluation.post_stop_hold_steps in the acceptance config."
        ),
    )
    parser.add_argument("--sim-steps-per-action", type=int, default=2)
    parser.add_argument("--action-replan-steps", type=int, default=10)
    parser.add_argument("--safety-shield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--safety-stop-distance",
        type=float,
        default=None,
        help="LiDAR safety-shield stop range. Defaults to smolvla_evaluation.safety_stop_distance in the acceptance config.",
    )
    parser.add_argument("--safety-slow-distance", type=float, default=1.25)
    parser.add_argument("--visual-stop-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visual-stop-min-pixels",
        type=int,
        default=None,
        help="Defaults to smolvla_evaluation.visual_stop_min_pixels in the acceptance config.",
    )
    parser.add_argument("--visual-stop-memory-steps", type=int, default=100)
    parser.add_argument("--visual-stop-memory-min-pixels", type=int, default=20)
    parser.add_argument(
        "--motion-smoothing",
        type=float,
        default=None,
        help="Defaults to smolvla_evaluation.motion_smoothing in the acceptance config.",
    )
    parser.add_argument(
        "--max-forward-delta",
        type=float,
        default=None,
        help="Defaults to smolvla_evaluation.max_forward_delta in the acceptance config.",
    )
    parser.add_argument(
        "--max-yaw-delta",
        type=float,
        default=None,
        help="Defaults to smolvla_evaluation.max_yaw_delta in the acceptance config.",
    )
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--stop-confirm", type=int, default=3)
    parser.add_argument(
        "--success-radius",
        type=float,
        default=None,
        help="Success radius in metres. Defaults to smolvla_evaluation.success_radius in the acceptance config.",
    )
    parser.add_argument(
        "--obstacle-contact-tolerance-steps",
        type=int,
        default=None,
        help=(
            "Contact steps tolerated before no_obstacle_contact fails. Defaults to "
            "smolvla_evaluation.obstacle_contact_tolerance_steps, else 0 (the historical "
            "zero-tolerance protocol). The strict score is always reported alongside the "
            "tolerance ladder, so raising this never hides the strict number."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--recovery-output-dir", type=Path, default=None)
    parser.add_argument("--recovery-max-episodes", type=int, default=2)
    parser.add_argument("--recovery-max-steps", type=int, default=650)
    return parser.parse_args()


# Episode-level evaluation geometry. It lives in the experiment config so the
# success radius and the safety stop cannot drift apart. An earlier revision
# hard-coded a 0.62 m safety stop against a 0.45 m success radius, so the LiDAR
# shield latched a stop 0.17 m *outside* the radius the episode was graded on:
# 3 of 13 episodes stopped 2.6-5.1 cm short of the ring and were scored as
# failures even though the policy had already arrived.
# ``validate_stop_geometry`` now enforces the invariant at run time.


def executable_action(raw_action: np.ndarray, stop_threshold: float) -> np.ndarray:
    """Clamp model output to the validated M20 body-command envelope."""
    raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError(f"SmolVLA returned an invalid action: {raw_action!r}")
    stop = float(raw[3] >= stop_threshold)
    if stop:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return np.asarray(
        (
            np.clip(raw[0], 0.0, 0.35),
            0.0,
            np.clip(raw[2], -0.15, 0.15),
            0.0,
        ),
        dtype=np.float64,
    )


def smooth_body_command(
    previous: np.ndarray,
    desired: np.ndarray,
    *,
    alpha: float,
    max_forward_delta: float,
    max_yaw_delta: float,
) -> np.ndarray:
    """Low-pass body commands while preserving an immediate confirmed stop."""
    previous = np.asarray(previous, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    if desired[3] > 0.5:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    blended = previous + alpha * (desired - previous)
    blended[0] = previous[0] + np.clip(blended[0] - previous[0], -max_forward_delta, max_forward_delta)
    blended[1] = 0.0
    blended[2] = previous[2] + np.clip(blended[2] - previous[2], -max_yaw_delta, max_yaw_delta)
    blended[3] = 0.0
    return blended


def contacting_obstacles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obstacle_geom_ids: set[int],
) -> set[str]:
    """Return obstacle geom names in physical contact with the robot."""
    names: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in obstacle_geom_ids or geom2 in obstacle_geom_ids:
            obstacle_id = geom1 if geom1 in obstacle_geom_ids else geom2
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_id)
            names.add(name or str(obstacle_id))
    return names


def load_policy(checkpoint: Path, device: torch.device, action_replan_steps: int):
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = str(device)
    config.pretrained_path = str(checkpoint)
    config.n_action_steps = int(action_replan_steps)
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    ).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def run_episode(args: argparse.Namespace, policy_bundle: tuple | None = None) -> dict:
    if args.episode_json is None or args.output is None:
        raise ValueError("--episode-json and --output are required unless fleet mode supplies them")
    if not args.checkpoint.is_dir() or not args.episode_json.is_file() or not ASSET.is_file():
        raise FileNotFoundError("checkpoint, episode JSON, or MuJoCo asset is missing")
    if args.policy_steps <= 0 or args.sim_steps_per_action <= 0 or args.stop_confirm <= 0:
        raise ValueError("step and stop-confirm values must be positive")
    if args.action_replan_steps <= 0:
        raise ValueError("action-replan-steps must be positive")
    validate_stop_geometry(
        success_radius=args.success_radius,
        stop_distance=args.safety_stop_distance,
        slow_distance=args.safety_slow_distance,
    )
    if not 0.0 < args.motion_smoothing <= 1.0:
        raise ValueError("motion-smoothing must be in (0, 1]")
    if args.max_forward_delta <= 0.0 or args.max_yaw_delta <= 0.0:
        raise ValueError("command delta limits must be positive")
    if args.visual_stop_min_pixels <= 0:
        raise ValueError("visual-stop-min-pixels must be positive")
    if args.visual_stop_memory_steps < 0 or args.visual_stop_memory_min_pixels <= 0:
        raise ValueError("visual stop memory settings are invalid")
    if args.recovery_output_dir is not None and (args.recovery_max_episodes <= 0 or args.recovery_max_steps <= 20):
        raise ValueError("recovery episode and step limits must be positive")

    metadata = json.loads(args.episode_json.read_text(encoding="utf-8"))
    policy_step_budget, policy_step_budget_source = resolve_policy_step_budget(args, metadata)
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    obstacles = obstacle_specs(metadata)
    scene_path = args.output.with_suffix(".scene.xml")
    build_scene(
        scene_path,
        object_specs(metadata),
        obstacles=obstacles,
        task_object_collisions=bool(metadata.get("task_object_collisions", False)),
        light=scene_light_spec(metadata),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    obstacle_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obstacle.name}_geom")
        for obstacle in obstacles
    }
    obstacle_geom_ids.discard(-1)
    controller = build_low_level_controller(model)
    controller.reset(
        data,
        float(metadata.get("initial_yaw", 0.0)),
        tuple(metadata.get("initial_xy", (0.0, 0.0))),
    )
    for _ in range(int(metadata.get("warmup_steps", 35))):
        controller.step(data, np.zeros(4, dtype=np.float64))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy, preprocessor, postprocessor = (
        policy_bundle if policy_bundle is not None else load_policy(args.checkpoint, device, args.action_replan_steps)
    )
    policy.reset()

    policy_width, policy_height = 160, 96
    policy_renderer = mujoco.Renderer(model, height=policy_height, width=policy_width)
    demo_renderer = (
        mujoco.Renderer(model, height=args.height, width=args.width) if args.write_videos else None
    )
    third_camera = mujoco.MjvCamera()
    third_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    third_camera.distance = 3.1
    third_camera.azimuth = 138.0
    third_camera.elevation = -24.0
    smoothed_lookat = np.asarray(data.qpos[:3], dtype=np.float64).copy()
    smoothed_lookat[2] = 0.5

    writer = open_video(args.output, args.width, args.height, fps=25) if args.write_videos else None
    front_writer = open_video(args.front_output, policy_width, policy_height, fps=25) if args.front_output else None
    rear_writer = open_video(args.rear_output, policy_width, policy_height, fps=25) if args.rear_output else None
    min_distance = float("inf")
    min_height = float(data.qpos[2])
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    min_clearance = minimum_obstacle_clearance(np.asarray(data.qpos[:2]), obstacles)
    target_first_visible_step = -1
    target_reached_step = -1
    stop_votes = 0
    stop_latched = False
    stop_step = -1
    # Steps actually simulated. Equals ``policy_step_budget`` unless the episode
    # ends early on a latched stop; logged so the size of the unattended tail the
    # old run-to-budget behaviour produced stays visible.
    graded_steps = 0
    raw_actions: list[list[float]] = []
    executed_actions: list[list[float]] = []
    shield_intervention_count = 0
    shield_emergency_count = 0
    minimum_front_lidar = float("inf")
    previous_command = np.zeros(4, dtype=np.float64)
    visual_stop_block_count = 0
    target_pixel_peak = 0
    target_pixel_peak_step = -1
    target_last_visible_step = -1
    target_last_confident_step = -1
    obstacle_contact_step_count = 0
    obstacle_first_contact_step = -1
    obstacle_contact_names: set[str] = set()
    base_heights: list[float] = []
    vertical_velocities: list[float] = []
    roll_rates: list[float] = []
    pitch_rates: list[float] = []
    wheel_targets: list[list[float]] = []
    recovery_active_steps = 0
    forward_blocked_steps = 0
    recovery_states: list[dict] = []
    try:
        for step in range(policy_step_budget):
            if args.recovery_output_dir is not None and step % 25 == 0 and not controller.safety_recovery_active:
                w, x, y, z = (float(value) for value in data.qpos[3:7])
                roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
                if (
                    float(data.qpos[2]) >= 0.45
                    and max(abs(math.degrees(roll)), abs(math.degrees(pitch))) <= 8.0
                    and not contacting_obstacles(model, data, obstacle_geom_ids)
                ):
                    recovery_states.append(capture_state(
                        data, controller, step, target_xy,
                        minimum_obstacle_clearance(np.asarray(data.qpos[:2]), obstacles),
                    ))
            policy_renderer.update_scene(data, camera="front_rgb")
            front = policy_renderer.render().copy()
            policy_renderer.update_scene(data, camera="rear_rgb")
            rear = policy_renderer.render().copy()
            visible_pixels = target_pixel_count(front, str(metadata["target_label"])) + target_pixel_count(
                rear, str(metadata["target_label"])
            )
            if visible_pixels > target_pixel_peak:
                target_pixel_peak = int(visible_pixels)
                target_pixel_peak_step = step
            if visible_pixels > 0:
                target_last_visible_step = step
            if visible_pixels >= args.visual_stop_memory_min_pixels:
                target_last_confident_step = step
            lidar = planar_lidar(model, data)
            state = m20_smolvla_state(proprioception(model, data), lidar)
            action_tensor = predict_action(
                {
                    "observation.images.front": front,
                    "observation.images.rear": rear,
                    "observation.state": state,
                },
                policy,
                device,
                preprocessor,
                postprocessor,
                bool(policy.config.use_amp),
                task=str(metadata["task_text"]),
                robot_type="m20pro",
            )
            raw = action_tensor.detach().cpu().numpy()[0]
            desired = executable_action(raw, args.stop_threshold)
            if stop_latched:
                desired = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
            else:
                stop_votes = stop_votes + 1 if desired[3] > 0.5 else 0
                if desired[3] > 0.5:
                    visual_stop_evidence = (
                        visible_pixels >= args.visual_stop_min_pixels
                        or (
                            target_last_confident_step >= 0
                            and step - target_last_confident_step <= args.visual_stop_memory_steps
                        )
                    )
                    if args.visual_stop_gate and not visual_stop_evidence:
                        visual_stop_block_count += 1
                        stop_votes = 0
                        desired = previous_command.copy()
                    elif stop_votes < args.stop_confirm:
                        desired = previous_command.copy()
                    else:
                        stop_latched = True
                        stop_step = step
            command = smooth_body_command(
                previous_command,
                desired,
                alpha=args.motion_smoothing,
                max_forward_delta=args.max_forward_delta,
                max_yaw_delta=args.max_yaw_delta,
            )
            front_lidar = float(np.min(lidar[32:41]))
            minimum_front_lidar = min(minimum_front_lidar, front_lidar)
            if args.safety_shield:
                command, shield_reason = lidar_safety_shield(
                    command,
                    lidar,
                    stop_distance=args.safety_stop_distance,
                    slow_distance=args.safety_slow_distance,
                )
                if shield_reason != "none":
                    shield_intervention_count += 1
                if shield_reason in {"emergency_stop", "invalid_lidar"}:
                    shield_emergency_count += 1
            previous_command = command.copy()
            raw_actions.append(raw.astype(float).tolist())
            executed_actions.append(command.astype(float).tolist())

            for _ in range(args.sim_steps_per_action):
                diagnostics = controller.step(data, command)
                recovery_active_steps += int(diagnostics.safety_recovery_active)
                forward_blocked_steps += int(
                    command[0] > 0.1 and float(np.linalg.norm(diagnostics.wheel_target)) < 1.0e-3
                )
                base_heights.append(float(data.qpos[2]))
                vertical_velocities.append(float(data.qvel[2]))
                roll_rates.append(float(data.qvel[3]))
                pitch_rates.append(float(data.qvel[4]))
                wheel_targets.append(diagnostics.wheel_target.astype(float).tolist())

            contacts = contacting_obstacles(model, data, obstacle_geom_ids)
            if contacts:
                obstacle_contact_step_count += 1
                obstacle_contact_names.update(contacts)
                if obstacle_first_contact_step < 0:
                    obstacle_first_contact_step = step

            distance = float(np.linalg.norm(np.asarray(data.qpos[:2]) - target_xy))
            min_distance = min(min_distance, distance)
            min_height = min(min_height, float(data.qpos[2]))
            w, x, y, z = (float(value) for value in data.qpos[3:7])
            roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
            max_abs_roll_deg = max(max_abs_roll_deg, abs(math.degrees(roll)))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(math.degrees(pitch)))
            clearance = minimum_obstacle_clearance(np.asarray(data.qpos[:2]), obstacles)
            if clearance is not None:
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            if visible_pixels > 0 and target_first_visible_step < 0:
                target_first_visible_step = step
            if distance <= args.success_radius and target_reached_step < 0:
                target_reached_step = step

            if front_writer is not None:
                front_writer.stdin.write(front.tobytes())
            if rear_writer is not None:
                rear_writer.stdin.write(rear.tobytes())
            if writer is not None:
                lookat = np.asarray(data.qpos[:3], dtype=np.float64).copy()
                lookat[2] = 0.5
                smoothed_lookat = 0.92 * smoothed_lookat + 0.08 * lookat
                third_camera.lookat[:] = smoothed_lookat
                demo_renderer.update_scene(data, camera=third_camera)
                writer.stdin.write(demo_renderer.render().copy().tobytes())
            if args.progress_interval and step % args.progress_interval == 0:
                print(json.dumps({"step": step, "distance": distance, "action": command.tolist()}), flush=True)
            # The latched stop is a decision, not a checkpoint: end the episode on
            # it (after a short settle window) exactly as the teacher episode ends,
            # instead of standing on a zero-velocity command until the budget runs
            # out. Every extra step can only hurt the whole-episode attitude and
            # contact criteria.
            if terminate_after_stop(
                stop_latched=stop_latched,
                stop_step=stop_step,
                step=step,
                hold_steps=args.post_stop_hold_steps,
            ):
                graded_steps = step + 1
                break
        else:
            graded_steps = policy_step_budget
    finally:
        for process in (writer, front_writer, rear_writer):
            if process is not None:
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("video encoding failed")
        policy_renderer.close()
        if demo_renderer is not None:
            demo_renderer.close()

    height_values = np.asarray(base_heights, dtype=np.float64)
    vertical_velocity_values = np.asarray(vertical_velocities, dtype=np.float64)
    roll_rate_values = np.asarray(roll_rates, dtype=np.float64)
    pitch_rate_values = np.asarray(pitch_rates, dtype=np.float64)
    wheel_target_values = np.asarray(wheel_targets, dtype=np.float64)
    wheel_target_delta = np.diff(wheel_target_values, axis=0)
    stable = bool(min_height >= 0.45 and max(max_abs_roll_deg, max_abs_pitch_deg) <= 8.0)
    contact_tolerance = max(
        0, int(getattr(args, "obstacle_contact_tolerance_steps", 0) or 0)
    )
    criteria = build_criteria(
        target_first_visible_step=target_first_visible_step,
        target_reached_step=target_reached_step,
        stop_step=stop_step,
        stable=stable,
        obstacle_contact_step_count=obstacle_contact_step_count,
        obstacle_contact_tolerance_steps=contact_tolerance,
    )
    report = {
        "schema": "m20pro_smolvla_closed_loop_v1",
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "episode_json": str(args.episode_json),
        "task_text": str(metadata["task_text"]),
        "target_label": str(metadata["target_label"]),
        "policy_steps": len(executed_actions),
        "policy_step_budget": int(policy_step_budget),
        "policy_step_budget_source": policy_step_budget_source,
        "policy_steps_executed": len(executed_actions),
        "graded_steps": int(graded_steps),
        "post_stop_hold_steps": int(args.post_stop_hold_steps),
        "post_stop_steps": int(max(0, len(executed_actions) - stop_step - 1)) if stop_step >= 0 else 0,
        "episode_ended_on_stop": bool(stop_step >= 0 and len(executed_actions) < policy_step_budget),
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "failure_mode": classify_failure(criteria, stop_step),
        "success_radius": float(args.success_radius),
        "sim_steps_per_action": int(args.sim_steps_per_action),
        "action_replan_steps": int(args.action_replan_steps),
        "safety_shield": bool(args.safety_shield),
        "safety_stop_distance": float(args.safety_stop_distance),
        "safety_slow_distance": float(args.safety_slow_distance),
        "visual_stop_gate": bool(args.visual_stop_gate),
        "visual_stop_min_pixels": int(args.visual_stop_min_pixels),
        "visual_stop_memory_steps": int(args.visual_stop_memory_steps),
        "visual_stop_memory_min_pixels": int(args.visual_stop_memory_min_pixels),
        "motion_smoothing": float(args.motion_smoothing),
        "max_forward_delta": float(args.max_forward_delta),
        "max_yaw_delta": float(args.max_yaw_delta),
        "policy_input": ["observation.images.front", "observation.images.rear", "observation.state", "task"],
        "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
        "success": bool(
            target_reached_step >= 0
            and stop_step >= 0
            and stable
            and obstacle_contact_step_count <= contact_tolerance
        ),
        "target_reached_step": int(target_reached_step),
        "target_first_visible_step": int(target_first_visible_step),
        "stop_step": int(stop_step),
        "min_target_distance": float(min_distance),
        "min_base_height": float(min_height),
        "max_abs_roll_deg": float(max_abs_roll_deg),
        "max_abs_pitch_deg": float(max_abs_pitch_deg),
        "base_height_range_m": float(np.ptp(height_values)),
        "base_height_std_m": float(np.std(height_values)),
        "vertical_velocity_rms_mps": float(np.sqrt(np.mean(np.square(vertical_velocity_values)))),
        "vertical_velocity_max_abs_mps": float(np.max(np.abs(vertical_velocity_values))),
        "roll_rate_rms_radps": float(np.sqrt(np.mean(np.square(roll_rate_values)))),
        "roll_rate_max_abs_radps": float(np.max(np.abs(roll_rate_values))),
        "pitch_rate_rms_radps": float(np.sqrt(np.mean(np.square(pitch_rate_values)))),
        "pitch_rate_max_abs_radps": float(np.max(np.abs(pitch_rate_values))),
        "wheel_target_delta_rms": float(np.sqrt(np.mean(np.square(wheel_target_delta)))),
        "wheel_target_delta_max_abs": float(np.max(np.abs(wheel_target_delta))),
        "recovery_active_control_steps": recovery_active_steps,
        "forward_command_without_wheel_target_steps": forward_blocked_steps,
        "min_obstacle_clearance": None if min_clearance is None else float(min_clearance),
        "min_footprint_clearance": None if min_clearance is None else float(min_clearance - ROBOT_FOOTPRINT_RADIUS_M),
        "minimum_front_lidar": float(minimum_front_lidar),
        "shield_intervention_count": int(shield_intervention_count),
        "shield_emergency_count": int(shield_emergency_count),
        "visual_stop_block_count": int(visual_stop_block_count),
        "target_pixel_peak": int(target_pixel_peak),
        "target_pixel_peak_step": int(target_pixel_peak_step),
        "target_last_visible_step": int(target_last_visible_step),
        "target_last_confident_step": int(target_last_confident_step),
        "obstacle_contact_step_count": int(obstacle_contact_step_count),
        "obstacle_contact_tolerance_steps": int(contact_tolerance),
        "obstacle_first_contact_step": int(obstacle_first_contact_step),
        "obstacle_contact_names": sorted(obstacle_contact_names),
        "stable": stable,
        "raw_action_min": np.min(np.asarray(raw_actions), axis=0).tolist(),
        "raw_action_max": np.max(np.asarray(raw_actions), axis=0).tolist(),
        "executed_forward_delta_max": float(np.max(np.abs(np.diff(np.asarray(executed_actions)[:, 0])))),
        "executed_yaw_delta_max": float(np.max(np.abs(np.diff(np.asarray(executed_actions)[:, 2])))),
        "final_xy": np.asarray(data.qpos[:2], dtype=float).tolist(),
        "demo_video": str(args.output),
        "front_video": "" if args.front_output is None else str(args.front_output),
        "rear_video": "" if args.rear_output is None else str(args.rear_output),
    }
    if args.recovery_output_dir is not None and not report["success"]:
        report["recovery_episodes"] = collect_continuations(
            model=model,
            states=recovery_states,
            metadata=metadata,
            obstacles=obstacles,
            source_episode_json=args.episode_json,
            checkpoint=args.checkpoint,
            output_dir=args.recovery_output_dir,
            max_episodes=args.recovery_max_episodes,
            max_steps=args.recovery_max_steps,
            success_radius=args.success_radius,
        )
    metrics_path = args.metrics or args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def episode_paths(args: argparse.Namespace) -> list[Path]:
    """Select the fleet episodes: explicit ids, else spread across the directory."""
    if not args.episodes_dir.is_dir():
        raise NotADirectoryError(f"Missing episodes directory: {args.episodes_dir}")
    if args.episode_ids:
        paths = []
        for item in (chunk.strip() for chunk in str(args.episode_ids).split(",")):
            if not item:
                continue
            candidate = args.episodes_dir / f"episode_{item}.json"
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing episode JSON: {candidate}")
            paths.append(candidate)
        if not paths:
            raise ValueError("--episode-ids did not name any episode")
        return paths
    available = sorted(
        args.episodes_dir.glob("episode_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if len(available) < args.episode_count:
        raise ValueError(f"requested {args.episode_count} episodes but only {len(available)} exist")
    stride = len(available) / float(args.episode_count)
    return [available[int(index * stride)] for index in range(args.episode_count)]


def judge_episode(report: dict, clearance_threshold: float) -> dict:
    """Apply the project's hidden-search wording to one closed-loop episode."""
    discovered = int(report["target_first_visible_step"]) >= 0
    reached = int(report["target_reached_step"]) >= 0
    stopped = int(report["stop_step"]) >= 0
    discovered_before_reach = discovered and (
        not reached or report["target_first_visible_step"] <= report["target_reached_step"]
    )
    false_stop_before_discovery = stopped and (
        not discovered or report["stop_step"] < report["target_first_visible_step"]
    )
    clearance = report.get("min_obstacle_clearance")
    clearance_ok = clearance is not None and float(clearance) >= clearance_threshold
    return {
        **report,
        "discovered": discovered,
        "false_stop_before_discovery": false_stop_before_discovery,
        "strict_pass": bool(
            report.get("success")
            and discovered
            and discovered_before_reach
            and not false_stop_before_discovery
            and clearance_ok
        ),
    }


def summarize_fleet(reports: list[dict], acceptance: dict) -> dict:
    """Aggregate learner-only closed-loop episodes and judge them against the
    single experiment config, because a single episode cannot decide promotion."""
    total = len(reports)
    if total == 0:
        raise ValueError("no episodes were evaluated")
    clearance_threshold = float(acceptance.get("min_obstacle_clearance_m", 0.1))
    judged = [judge_episode(report, clearance_threshold) for report in reports]
    successes = [item for item in judged if item.get("success")]
    discovered = [item for item in judged if item["discovered"]]
    strict = [item for item in judged if item["strict_pass"]]
    false_stops = [item for item in judged if item["false_stop_before_discovery"]]
    clearances = [
        float(item["min_obstacle_clearance"])
        for item in judged
        if item.get("min_obstacle_clearance") is not None
    ]
    contacts = [int(item["obstacle_contact_step_count"]) for item in judged]

    # ``success`` is a conjunction whose contact clause is zero-tolerance by
    # design, and on a doorway-constrained scene that single clause can decide
    # most of the score. Report the same conjunction at a ladder of tolerances
    # beside the strict count, plus how many episodes reached, stopped, stayed
    # upright and were blocked *only* by contact - that split is what separates
    # "the learner could not navigate" from "the learner navigated and grazed".
    tolerance_counts = count_successes_by_contact_tolerance(judged)
    arrived_only_blocked_by_contact = sum(
        1
        for item in judged
        if all(item.get("criteria", {}).get(name) for name in CONTACT_TOLERANT_CORE_CRITERIA)
        and int(item["obstacle_contact_step_count"]) > 0
    )

    # Report the five success criteria separately. ``success`` is a conjunction,
    # so a single pass/fail hides whether an episode failed on geometry, on the
    # stop decision, on attitude, or on collisions - four different fixes.
    criterion_counts = {
        name: sum(1 for item in judged if item.get("criteria", {}).get(name))
        for name in CRITERION_NAMES
    }
    failure_modes = Counter(item.get("failure_mode", "unknown") for item in judged)

    def mean(key: str) -> float | None:
        values = [float(item[key]) for item in judged if item.get(key) is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "schema": "m20pro_smolvla_closed_loop_fleet_v1",
        "checkpoint": judged[0]["checkpoint"],
        "seed": judged[0]["seed"],
        "policy_input": judged[0]["policy_input"],
        "prohibited_policy_input": judged[0]["prohibited_policy_input"],
        "success_definition": (
            f"reached the {judged[0]['success_radius']:.2f} m radius + latched stop + "
            "stable attitude + zero MuJoCo obstacle contact"
        ),
        "in_distribution_note": (
            "This panel alone does not establish held-out generalization. Verify exact scene "
            "overlap with the training dataset and the initialization policy's training history."
        ),
        "episode_count": total,
        "success_count": len(successes),
        "discovery_count": len(discovered),
        "strict_pass_count": len(strict),
        "false_stop_count": len(false_stops),
        "zero_obstacle_contact_episode_count": sum(1 for value in contacts if value == 0),
        "obstacle_contact_tolerance_steps": int(
            judged[0].get("obstacle_contact_tolerance_steps", 0)
        ),
        "success_count_by_obstacle_contact_tolerance": tolerance_counts,
        "arrived_only_blocked_by_contact_count": arrived_only_blocked_by_contact,
        "criterion_pass_count": criterion_counts,
        "criterion_pass_rate": {name: count / total for name, count in criterion_counts.items()},
        "failure_mode_counts": dict(failure_modes),
        "arrived_without_stop_count": failure_modes.get("arrived_without_stop", 0),
        "premature_stop_count": failure_modes.get("premature_stop", 0),
        "success_rate": len(successes) / total,
        "discovery_rate": len(discovered) / total,
        "strict_pass_rate": len(strict) / total,
        "false_stop_rate": len(false_stops) / total,
        "min_obstacle_clearance_min_m": min(clearances) if clearances else None,
        "jitter_mean": {
            "base_height_range_m": mean("base_height_range_m"),
            "vertical_velocity_rms_mps": mean("vertical_velocity_rms_mps"),
            "roll_rate_rms_radps": mean("roll_rate_rms_radps"),
            "pitch_rate_rms_radps": mean("pitch_rate_rms_radps"),
            "wheel_target_delta_rms": mean("wheel_target_delta_rms"),
            "max_abs_pitch_deg": mean("max_abs_pitch_deg"),
            "min_base_height_m": mean("min_base_height"),
        },
        "acceptance_thresholds": acceptance,
        "episodes": judged,
    }
    checks = {
        "enough_episodes": total >= int(acceptance.get("min_episodes", 1)),
        "success_rate": summary["success_rate"] >= float(acceptance.get("min_success_rate", 1.0)),
        "discovery_rate": summary["discovery_rate"] >= float(acceptance.get("min_discovery_rate", 1.0)),
        "strict_pass_rate": summary["strict_pass_rate"] >= float(acceptance.get("min_strict_pass_rate", 1.0)),
        "false_stop_rate": summary["false_stop_rate"] <= float(acceptance.get("max_false_stop_rate", 0.0)),
        "obstacle_clearance": (
            summary["min_obstacle_clearance_min_m"] is not None
            and summary["min_obstacle_clearance_min_m"] >= clearance_threshold
        ),
    }
    summary["acceptance_checks"] = checks
    summary["learner_only_closed_loop_passed"] = bool(all(checks.values()))
    return summary


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.acceptance_config.is_file():
        config = json.loads(args.acceptance_config.read_text(encoding="utf-8"))
    apply_evaluation_config(args, config)
    if args.episodes_dir is None:
        print(json.dumps(run_episode(args), indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in fleet mode")
    paths = episode_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    acceptance: dict = config.get("closed_loop_acceptance", {}) or {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_policy(args.checkpoint, device, args.action_replan_steps)
    reports: list[dict] = []
    for path in paths:
        stem = path.with_suffix("").name
        episode_args = argparse.Namespace(**vars(args))
        episode_args.episode_json = path
        episode_args.output = args.output_dir / f"{stem}.demo.mp4"
        episode_args.metrics = args.output_dir / f"{stem}.metrics.json"
        episode_args.front_output = args.output_dir / f"{stem}.front.mp4" if args.write_videos else None
        episode_args.rear_output = args.output_dir / f"{stem}.rear.mp4" if args.write_videos else None
        report = run_episode(episode_args, policy_bundle=bundle)
        reports.append(report)
        print(json.dumps({
            "episode": stem,
            "success": report["success"],
            "failure_mode": report["failure_mode"],
            "failed_criteria": report["failed_criteria"],
            "stable": report["stable"],
            "min_target_distance": round(float(report["min_target_distance"]), 3),
            "reached_step": report["target_reached_step"],
            "stop_step": report["stop_step"],
            "budget": report["policy_step_budget"],
            "obstacle_contact_steps": report["obstacle_contact_step_count"],
            "max_abs_pitch_deg": round(float(report["max_abs_pitch_deg"]), 2),
        }), flush=True)
    summary = summarize_fleet(reports, acceptance)
    summary_path = args.summary or (args.output_dir / "fleet_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

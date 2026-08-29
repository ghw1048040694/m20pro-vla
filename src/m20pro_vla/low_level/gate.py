"""Deterministic regression gate for the reusable M20 controller."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .controller import PHYSICS_STEPS, M20LowLevelController
from m20pro_vla.sim.mujoco import ASSET, build_scene


def _attitude(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation = data.xmat[base_id].reshape(3, 3)
    return (
        math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
        math.atan2(float(-rotation[2, 0]), float(np.hypot(rotation[2, 1], rotation[2, 2]))),
    )


def _yaw_angle(data: mujoco.MjData) -> float:
    q = data.qpos[3:7]
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def _wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def run_low_level_gate(
    *,
    warmup_steps: int = 100,
    forward_steps: int = 220,
    stop_steps: int = 100,
    turn_steps: int = 180,
    turn_command: float = 0.10,
) -> dict:
    """Run stance, forward, stop, and turn phases against the canonical controller."""
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the MuJoCo asset first: {ASSET}")
    if min(warmup_steps, forward_steps, stop_steps, turn_steps) <= 0:
        raise ValueError("all phases must be positive")
    if not np.isfinite(turn_command) or abs(turn_command) > 0.15:
        raise ValueError("turn_command must be finite and within the body-command yaw range")
    scene_path = ASSET.with_name("m20_low_level_gate.scene.xml")
    build_scene(scene_path, [])
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        controller = M20LowLevelController(model)
        controller.reset(data, 0.0)
        phases = (
            ("warmup", warmup_steps, np.array((0.0, 0.0, 0.0, 0.0))),
            ("forward", forward_steps, np.array((0.35, 0.0, 0.0, 0.0))),
            ("stop", stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
            ("turn", turn_steps, np.array((0.0, 0.0, turn_command, 0.0))),
            ("final_stop", stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
        )
        phase_stats: dict[str, dict] = {}
        all_finite = True
        initial_xy = data.qpos[:2].copy()
        initial_yaw = _yaw_angle(data)
        for phase_name, count, action in phases:
            start = data.qpos[:2].copy()
            phase_yaw = _yaw_angle(data)
            heights: list[float] = []
            rolls: list[float] = []
            pitches: list[float] = []
            speeds: list[float] = []
            for _ in range(count):
                controller.step(data, action)
                roll, pitch = _attitude(model, data)
                heights.append(float(data.qpos[2]))
                rolls.append(abs(roll))
                pitches.append(abs(pitch))
                speeds.append(float(np.linalg.norm(data.qvel[:2])))
                all_finite = all_finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            phase_stats[phase_name] = {
                "steps": count,
                "displacement_xy_m": float(np.linalg.norm(data.qpos[:2] - start)),
                "min_base_height_m": min(heights),
                "max_abs_roll_deg": math.degrees(max(rolls)),
                "max_abs_pitch_deg": math.degrees(max(pitches)),
                "max_horizontal_speed_mps": max(speeds),
                "yaw_change_deg": math.degrees(_wrapped_angle(_yaw_angle(data) - phase_yaw)),
                "final_xy": data.qpos[:2].tolist(),
            }
        final_yaw = _yaw_angle(data)
        forward = phase_stats["forward"]
        stop = phase_stats["stop"]
        turn = phase_stats["turn"]
        report = {
            "schema": "m20pro_low_level_stance_velocity_gate_v2",
            "controller": "M20LowLevelController_feedback_stance_velocity",
            "action_contract": ["forward", "lateral", "yaw", "stop"],
            "physics_steps_per_control": PHYSICS_STEPS,
            "all_finite": all_finite,
            "forward_displacement_m": forward["displacement_xy_m"],
            "forward_min_height_m": forward["min_base_height_m"],
            "forward_max_tilt_deg": max(forward["max_abs_roll_deg"], forward["max_abs_pitch_deg"]),
            "stop_drift_m": stop["displacement_xy_m"],
            "turn_displacement_m": turn["displacement_xy_m"],
            "yaw_change_deg": math.degrees(_wrapped_angle(final_yaw - initial_yaw)),
            "turn_command": turn_command,
            "turn_yaw_change_deg": turn["yaw_change_deg"],
            "final_stop_yaw_change_deg": phase_stats["final_stop"]["yaw_change_deg"],
            "gates": {
                "finite_state": all_finite,
                "forward_height": forward["min_base_height_m"] >= 0.45,
                "forward_tilt": forward["max_abs_roll_deg"] <= 8.0 and forward["max_abs_pitch_deg"] <= 8.0,
                "stop_drift": stop["displacement_xy_m"] <= 0.25,
                "turn_yaw": abs(turn["yaw_change_deg"]) >= 10.0,
                "turn_height": turn["min_base_height_m"] >= 0.45,
                "turn_tilt": turn["max_abs_roll_deg"] <= 8.0 and turn["max_abs_pitch_deg"] <= 8.0,
                "turn_exit_height": phase_stats["final_stop"]["min_base_height_m"] >= 0.45,
            },
            "phase_stats": phase_stats,
            "initial_xy": initial_xy.tolist(),
            "final_xy": data.qpos[:2].tolist(),
        }
        report["eligible_for_flat_vla_execution"] = bool(all(report["gates"].values()))
        return report
    finally:
        scene_path.unlink(missing_ok=True)


__all__ = ["run_low_level_gate"]

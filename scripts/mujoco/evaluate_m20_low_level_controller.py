#!/usr/bin/env python3
"""Run deterministic M20 low-level stance, drive, stop, and turn gates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.sim.mujoco import ASSET, build_scene
from m20pro_vla.low_level import M20LowLevelController
from m20pro_vla.low_level.controller import PHYSICS_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/low_level/m20_stance_velocity_gate.json"))
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--forward-steps", type=int, default=220)
    parser.add_argument("--stop-steps", type=int, default=100)
    parser.add_argument("--turn-steps", type=int, default=180)
    parser.add_argument("--turn-command", type=float, default=0.10, help="Signed yaw body command for this gate")
    return parser.parse_args()


def attitude(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation = data.xmat[base_id].reshape(3, 3)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(float(-rotation[2, 0]), float(np.hypot(rotation[2, 1], rotation[2, 2])))
    return roll, pitch


def yaw_angle(data: mujoco.MjData) -> float:
    q = data.qpos[3:7]
    return math.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
    )


def wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main() -> None:
    args = parse_args()
    if not ASSET.is_file():
        raise FileNotFoundError(ASSET)
    if min(args.warmup_steps, args.forward_steps, args.stop_steps, args.turn_steps) <= 0:
        raise ValueError("all phases must be positive")
    if not np.isfinite(args.turn_command) or abs(args.turn_command) > 0.15:
        raise ValueError("--turn-command must be finite and within the body-command yaw range")
    scene_path = args.output.with_suffix(".scene.xml")
    build_scene(scene_path, [])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = M20LowLevelController(model)
    controller.reset(data, 0.0)
    phases = (
        ("warmup", args.warmup_steps, np.array((0.0, 0.0, 0.0, 0.0))),
        ("forward", args.forward_steps, np.array((0.35, 0.0, 0.0, 0.0))),
        ("stop", args.stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
        ("turn", args.turn_steps, np.array((0.0, 0.0, args.turn_command, 0.0))),
        ("final_stop", args.stop_steps, np.array((0.0, 0.0, 0.0, 1.0))),
    )
    phase_stats = {}
    all_finite = True
    initial_xy = data.qpos[:2].copy()
    initial_yaw = yaw_angle(data)
    for phase_name, count, action in phases:
        start = data.qpos[:2].copy()
        phase_yaw = yaw_angle(data)
        heights = []
        rolls = []
        pitches = []
        speeds = []
        for _ in range(count):
            controller.step(data, action)
            roll, pitch = attitude(model, data)
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
            "yaw_change_deg": math.degrees(wrapped_angle(yaw_angle(data) - phase_yaw)),
            "final_xy": data.qpos[:2].tolist(),
        }
    final_yaw = yaw_angle(data)
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
        "yaw_change_deg": math.degrees(wrapped_angle(final_yaw - initial_yaw)),
        "turn_command": args.turn_command,
        "turn_yaw_change_deg": phase_stats["turn"]["yaw_change_deg"],
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    scene_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

#!/usr/bin/env python3
"""Measure the target's apparent size in the policy camera as a function of range.

The closed loop grades a stop on ``visual_stop_min_pixels`` (80 px of a 160x96
frame), but the training labels never used that criterion: the collector marks a
stop from the privileged base-to-target distance and marks "visible" from a
geometric line of sight. This probe answers the question that decides whether the
stop is learnable at all - how many colour-mask pixels does the target actually
cover at the range where the teacher stops?

The stance is settled once with the low-level controller and then frozen: only
the free joint is re-placed per sample, so the reported range is the true range.
An earlier revision re-settled at every sample and the standing controller walked
the base up to 1.07 m off the intended spot, which silently corrupted the sweep.

Usage:
  python scripts/mujoco/probe_target_pixels.py --episode-json <episode_6000.json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m20pro_vla.data.visibility import TARGET_PIXEL_THRESHOLD, target_pixel_count
from m20pro_vla.low_level import build_low_level_controller
from m20pro_vla.sim.mujoco import build_scene, planar_lidar
from play_m20_mujoco_vla import object_specs, obstacle_specs, scene_light_spec

POLICY_WIDTH, POLICY_HEIGHT = 160, 96
BASE_HEIGHT_M = 0.50
STANCE_SETTLE_STEPS = 240

# The camera is carried by the robot, so a zeroed joint pose is not a camera
# pose. The analytic backend is used deliberately: it needs no onnxruntime and
# settles the body at the same standing attitude the v5 policy produces.
STANCE_BACKEND = "analytic"


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=np.float64)


def settle_stance(model: mujoco.MjModel) -> np.ndarray:
    """Return a settled upright qpos: leg joints commanded to a steady stance."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[2] = BASE_HEIGHT_M
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    controller = build_low_level_controller(model, backend=STANCE_BACKEND)
    controller.reset(data, 0.0, (float(data.qpos[0]), float(data.qpos[1])))
    for _ in range(STANCE_SETTLE_STEPS):
        controller.step(data, np.zeros(4, dtype=np.float64))
        mujoco.mj_step(model, data)
    return data.qpos.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/target_pixel_probe.json"))
    parser.add_argument(
        "--ranges",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.25,1.50,2.00,2.50,3.00",
    )
    parser.add_argument("--yaw-offsets-deg", default="0")
    parser.add_argument("--gate-pixels", type=int, default=80)
    parser.add_argument(
        "--strip-obstacles",
        action="store_true",
        help=(
            "Rebuild the scene with no walls. The planar LiDAR only scans geom "
            "group 1, so this isolates whether it can see the target at all "
            "instead of reporting the nearest wall."
        ),
    )
    args = parser.parse_args()

    metadata = json.loads(args.episode_json.read_text(encoding="utf-8"))
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    target_label = str(metadata["target_label"])
    obstacles = obstacle_specs(metadata)

    scene_obstacles = [] if args.strip_obstacles else obstacles
    scene_path = Path("/tmp/probe_target_pixels.scene.xml")
    build_scene(
        scene_path,
        object_specs(metadata),
        obstacles=scene_obstacles,
        task_object_collisions=bool(metadata.get("task_object_collisions", False)),
        light=scene_light_spec(metadata),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    stance = settle_stance(model)
    stance_z = float(stance[2])
    joint_pose = stance[7:].copy()

    renderer = mujoco.Renderer(model, height=POLICY_HEIGHT, width=POLICY_WIDTH)
    ranges = [float(value) for value in args.ranges.split(",") if value]
    yaw_offsets = [math.radians(float(value)) for value in args.yaw_offsets_deg.split(",") if value]
    heading = math.atan2(float(target_xy[1]), float(target_xy[0]))

    rows = []
    try:
        for distance in ranges:
            for yaw_offset in yaw_offsets:
                yaw = heading + yaw_offset
                base_xy = target_xy - distance * np.asarray((math.cos(yaw), math.sin(yaw)))
                data.qpos[:] = 0.0
                data.qpos[0:2] = base_xy
                data.qpos[2] = stance_z
                data.qpos[3:7] = yaw_to_quat(yaw)
                data.qpos[7:] = joint_pose
                mujoco.mj_forward(model, data)
                true_distance = float(np.linalg.norm(np.asarray(data.qpos[:2]) - target_xy))
                renderer.update_scene(data, camera="front_rgb")
                front_px = target_pixel_count(renderer.render().copy(), target_label)
                renderer.update_scene(data, camera="rear_rgb")
                rear_px = target_pixel_count(renderer.render().copy(), target_label)
                scan = planar_lidar(model, data)
                fan = scan[32:41]
                rows.append(
                    {
                        "requested_distance_m": round(distance, 3),
                        "true_distance_m": round(true_distance, 4),
                        "yaw_offset_deg": round(math.degrees(yaw_offset), 1),
                        "front_pixels": front_px,
                        "rear_pixels": rear_px,
                        "total_pixels": front_px + rear_px,
                        "passes_gate": bool(front_px >= args.gate_pixels),
                        "lidar_forward_fan_min": round(float(fan.min()), 4),
                        "lidar_forward_fan_max": round(float(fan.max()), 4),
                        "lidar_scan_min": round(float(scan.min()), 4),
                    }
                )
    finally:
        renderer.close()

    print(f"stance height {stance_z:.3f} m, {len(joint_pose)} joint dof frozen")
    print(f"obstacles in scene: {len(scene_obstacles)} (stripped: {args.strip_obstacles})\n")
    print(f"{'dist':>6} {'true':>7} {'front':>7} {'total':>6} {'gate':>6} {'fan_min':>8} {'scan_min':>9}")
    for row in rows:
        print(
            f"{row['requested_distance_m']:6.2f} {row['true_distance_m']:7.3f} "
            f"{row['front_pixels']:7d} {row['total_pixels']:6d} "
            f"{str(row['passes_gate']):>6} {row['lidar_forward_fan_min']:8.3f} "
            f"{row['lidar_scan_min']:9.3f}"
        )

    facing = [row for row in rows if row["yaw_offset_deg"] == 0.0]
    print("\n--- facing the target ---")
    for row in facing:
        print(
            f"  {row['true_distance_m']:.2f} m -> {row['front_pixels']} px "
            f"(gate {args.gate_pixels}: {row['passes_gate']}), "
            f"lidar fan {row['lidar_forward_fan_min']:.3f} m"
        )

    payload = {
        "episode_json": str(args.episode_json),
        "target_label": target_label,
        "target_xy": target_xy.tolist(),
        "stance_height_m": stance_z,
        "target_pixel_threshold": int(TARGET_PIXEL_THRESHOLD),
        "gate_pixels": int(args.gate_pixels),
        "measurement": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

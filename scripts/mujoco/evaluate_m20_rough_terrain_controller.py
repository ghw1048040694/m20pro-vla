#!/usr/bin/env python3
"""Run the versioned M20 low-level rough-terrain and recovery gates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from m20pro_vla.low_level import M20LowLevelController
from m20pro_vla.low_level.controller import PHYSICS_STEPS
from m20pro_vla.sim.mujoco import ASSET, TerrainSpec, build_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/low_level/m20_rough_terrain_gate_v1.json"))
    parser.add_argument("--steps", type=int, default=360, help="Forward control steps per terrain profile")
    parser.add_argument("--forward", type=float, default=0.28)
    parser.add_argument("--disturbance-force", type=float, default=35.0)
    return parser.parse_args()


def attitude(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation = data.xmat[base_id].reshape(3, 3)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(float(-rotation[2, 0]), float(np.hypot(rotation[2, 1], rotation[2, 2])))
    return roll, pitch


def terrain_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        )
        if any(name.startswith("terrain_") for name in names):
            count += 1
    return count


def run_profile(name: str, steps: int, forward: float, disturbance_force: float) -> dict[str, object]:
    output_dir = Path(".runtime/low_level")
    scene_path = output_dir / f"m20_rough_{name}.scene.xml"
    build_scene(scene_path, [], terrain=None if name == "disturbance" else TerrainSpec(name))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = M20LowLevelController(model)
    controller.reset(data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for _ in range(60):
        controller.step(data, (0.0, 0.0, 0.0, 0.0))

    start = data.qpos[:3].copy()
    heights: list[float] = []
    tilts: list[float] = []
    speeds: list[float] = []
    contacts = 0
    lift_steps = 0
    safety_recovery_steps = 0
    finite = True
    recovery_window = range(steps // 3, steps // 3 + 8) if name == "disturbance" else range(0)
    for index in range(steps):
        data.xfrc_applied[base_id, :] = 0.0
        if name == "disturbance" and index in recovery_window:
            # A short lateral shove and roll impulse, representative of a contact.
            data.xfrc_applied[base_id, 1] = disturbance_force
            data.xfrc_applied[base_id, 3] = 0.35 * disturbance_force
        diagnostics = controller.step(data, (forward, 0.0, 0.0, 0.0))
        lift_steps += int(diagnostics.terrain_lift_active)
        safety_recovery_steps += int(diagnostics.safety_recovery_active)
        roll, pitch = attitude(model, data)
        heights.append(float(data.qpos[2]))
        tilts.append(max(abs(roll), abs(pitch)))
        speeds.append(float(np.linalg.norm(data.qvel[:2])))
        contacts += terrain_contacts(model, data)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    data.xfrc_applied[base_id, :] = 0.0
    for _ in range(80):
        controller.step(data, (0.0, 0.0, 0.0, 1.0))
    roll, pitch = attitude(model, data)
    scene_path.unlink(missing_ok=True)
    return {
        "profile": name,
        "terrain_schema": "m20_rough_terrain_v1",
        "steps": steps,
        "command": [forward, 0.0, 0.0, 0.0],
        "displacement_m": float(np.linalg.norm(data.qpos[:2] - start[:2])),
        "forward_displacement_m": float(data.qpos[0] - start[0]),
        "min_base_height_m": min(heights),
        "max_tilt_deg": math.degrees(max(tilts)),
        "final_tilt_deg": math.degrees(max(abs(roll), abs(pitch))),
        "max_horizontal_speed_mps": max(speeds),
        "terrain_contact_samples": int(contacts),
        "terrain_lift_steps": lift_steps,
        "safety_recovery_steps": safety_recovery_steps,
        "finite_state": finite,
        "post_brake_speed_mps": float(np.linalg.norm(data.qvel[:2])),
        "gates": {
            "finite_state": finite,
            "progress": float(data.qpos[0] - start[0]) >= 1.0,
            "height": min(heights) >= 0.42,
            "tilt": max(tilts) <= math.radians(12.0),
            "recovered": float(np.linalg.norm(data.qvel[:2])) <= 0.25 and max(abs(roll), abs(pitch)) <= math.radians(8.0),
            "terrain_contact_seen": contacts > 0 if name in {"slope", "bumps", "step"} else True,
        },
    }


def main() -> None:
    args = parse_args()
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the MuJoCo asset first: {ASSET}")
    if args.steps <= 0 or not np.isfinite(args.forward) or args.forward <= 0.0:
        raise ValueError("steps and forward must be positive")
    profiles = ["flat", "slope", "bumps", "step", "disturbance"]
    results = [run_profile(name, args.steps, args.forward, args.disturbance_force) for name in profiles]
    report = {
        "schema": "m20pro_low_level_rough_terrain_gate_v1",
        "controller": "M20LowLevelController_feedback_stance_velocity",
        "contract": "m20_low_level_v1",
        "physics_steps_per_control": PHYSICS_STEPS,
        "profiles": results,
        "eligible_for_rough_terrain_vla_execution": bool(all(all(item["gates"].values()) for item in results)),
        "note": "This gate validates the reusable execution layer; it is not a VLA success result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

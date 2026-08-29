#!/usr/bin/env python3
"""Validate the official M20 low-level locomotion adapter in MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from m20_mujoco_low_level import M20OfficialLocomotion
from m20pro_vla.sim.scene import RUNTIME_ROOT, TargetSpec, materialize_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_ROOT / "smoke" / "m20_low_level_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")
    command = np.array([args.forward, args.lateral, args.yaw], dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = materialize_scene(args.output_dir / "m20_visible_target.xml", TargetSpec())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    locomotion = M20OfficialLocomotion(model)
    locomotion.reset(data)

    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        renderer.update_scene(data, camera="front_rgb")
        Image.fromarray(renderer.render(), mode="RGB").save(args.output_dir / "initial_front_rgb.png")
    finally:
        renderer.close()

    total_steps = round(args.seconds / locomotion.policy_period)
    initial_position = data.qpos[:3].copy()
    heights = []
    for _ in range(total_steps):
        locomotion.step(data, command)
        heights.append(float(data.qpos[2]))
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    if not finite:
        raise RuntimeError("Low-level locomotion produced a non-finite MuJoCo state")

    metrics = {
        "schema": "m20pro_mujoco_low_level_smoke_v1",
        "scope": "Official low-level locomotion adapter only; not a VLA navigation result.",
        "command": command.tolist(),
        "seconds": args.seconds,
        "policy_hz": 1.0 / locomotion.policy_period,
        "physics_hz": 1.0 / model.opt.timestep,
        "initial_base_xyz": initial_position.tolist(),
        "final_base_xyz": data.qpos[:3].tolist(),
        "xy_displacement_m": float(np.linalg.norm(data.qpos[:2] - initial_position[:2])),
        "min_base_z": min(heights),
        "final_base_z": float(data.qpos[2]),
        "final_linear_speed_mps": float(np.linalg.norm(data.qvel[:3])),
        "finite_state": finite,
        "initial_front_rgb": str(args.output_dir / "initial_front_rgb.png"),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

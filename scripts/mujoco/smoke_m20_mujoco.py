#!/usr/bin/env python3
"""Validate the official M20 MJCF asset, physics state, and RGB observation path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from m20pro_vla.sim.scene import RUNTIME_ROOT, TargetSpec, materialize_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME_ROOT / "smoke" / "m20_mujoco_v1")
    return parser.parse_args()


def write_png(path: Path, pixels: np.ndarray) -> None:
    _, _, channels = pixels.shape
    if channels != 3 or pixels.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {pixels.shape} {pixels.dtype}")
    Image.fromarray(pixels, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("steps, width, and height must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = materialize_scene(args.output_dir / "m20_visible_target.xml", TargetSpec())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    initial_qpos = data.qpos.copy()
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("M20 physics generated a non-finite state")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        renderer.update_scene(data, camera="front_rgb")
        rgb = renderer.render()
    finally:
        renderer.close()
    png_path = args.output_dir / "front_rgb.png"
    write_png(png_path, rgb)

    green = (rgb[..., 1] > 150) & (rgb[..., 1] > rgb[..., 0] * 1.4) & (rgb[..., 1] > rgb[..., 2] * 1.2)
    metrics = {
        "schema": "m20pro_mujoco_asset_smoke_v1",
        "simulator": {"name": "MuJoCo", "version": mujoco.__version__},
        "asset": {
            "model": "AI-DA-STC/M20-autonomy-sim M20.xml",
            "scene_path": str(scene_path),
            "joint_count": model.njnt,
            "actuator_count": model.nu,
            "body_count": model.nbody,
        },
        "physics": {
            "steps": args.steps,
            "finite_state": True,
            "initial_base_z": float(initial_qpos[2]),
            "final_base_z": float(data.qpos[2]),
        },
        "observation": {
            "camera": "front_rgb",
            "rgb_shape": list(rgb.shape),
            "green_target_pixel_fraction": float(green.mean()),
            "png": str(png_path),
        },
        "scope": "Asset/physics/observation smoke test only; not a learned navigation result.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

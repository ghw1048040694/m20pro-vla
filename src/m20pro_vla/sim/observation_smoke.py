"""Package-level MuJoCo observation contract smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .mujoco import ObjectSpec, build_scene, planar_lidar, proprioception


def _colour_pixel_counts(image: np.ndarray) -> dict[str, int]:
    red, green, blue = (image[..., index].astype(np.int16) for index in range(3))
    return {
        "red": int(np.count_nonzero((red > 90) & (red > green * 3 // 2) & (red > blue * 3 // 2))),
        "green": int(np.count_nonzero((green > 70) & (green > red * 3 // 2) & (green > blue * 3 // 2))),
        "yellow": int(np.count_nonzero((red > 90) & (green > 80) & (blue * 2 < red))),
    }


def run_observation_smoke(output_dir: Path, *, width: int = 320, height: int = 180) -> dict:
    """Render RGB, LiDAR, and proprioception without privileged target state."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = [
        ObjectSpec("red_cube", "red cube", "box", (2.8, 0.0), (0.90, 0.08, 0.05, 1.0), (0.18, 0.18, 0.18)),
        ObjectSpec("green_cylinder", "green cylinder", "cylinder", (3.2, 0.7), (0.05, 0.75, 0.12, 1.0), (0.15, 0.20)),
        ObjectSpec("yellow_box", "yellow box", "box", (-2.6, -0.3), (0.95, 0.72, 0.05, 1.0), (0.20, 0.13, 0.16)),
    ]
    scene_path = build_scene(output_dir / "scene.xml", objects, task_object_collisions=False)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    images: dict[str, np.ndarray] = {}
    try:
        for camera in ("front_rgb", "rear_rgb"):
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            images[camera] = image
            Image.fromarray(image, mode="RGB").save(output_dir / f"{camera}.png")
    finally:
        renderer.close()
    lidar = planar_lidar(model, data)
    state = proprioception(model, data)
    np.save(output_dir / "lidar.npy", lidar)
    np.save(output_dir / "state.npy", state)
    counts = {name: _colour_pixel_counts(image) for name, image in images.items()}
    if counts["front_rgb"]["red"] == 0 or counts["front_rgb"]["green"] == 0:
        raise RuntimeError(f"Expected front RGB targets are absent: {counts}")
    report = {
        "schema": "m20pro_vla_observation_smoke_v1",
        "simulator": {"name": "MuJoCo", "version": mujoco.__version__},
        "asset": str(scene_path),
        "policy_input": {
            "front_rgb": {"shape": list(images["front_rgb"].shape), "path": "front_rgb.png"},
            "rear_rgb": {"shape": list(images["rear_rgb"].shape), "path": "rear_rgb.png"},
            "lidar": {"shape": list(lidar.shape), "path": "lidar.npy"},
            "state": {"shape": list(state.shape), "path": "state.npy"},
            "language": "Go to the red cube.",
        },
        "prohibited_policy_input": ["target_world_position", "target_geometry_id", "semantic_mask", "simulator_object_pose"],
        "colour_pixel_counts": counts,
        "lidar_min_m": float(lidar.min()),
        "lidar_max_m": float(lidar.max()),
    }
    (output_dir / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["run_observation_smoke"]

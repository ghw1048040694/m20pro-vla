#!/usr/bin/env python3
"""Render a target-visible M20 observation without exposing target truth to policy input."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
ASSET = WORKSPACE / ".runtime/mujoco_assets/M20_floating_actuated.xml"
DEFAULT_OUTPUT = WORKSPACE / ".runtime/mujoco_observation_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    return parser.parse_args()


def append_scene(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    base = worldbody.find("./body[@name='base_link']") if worldbody is not None else None
    if worldbody is None or base is None:
        raise ValueError("Floating M20 asset is missing worldbody/base_link")
    worldbody.insert(0, ET.Element("light", {
        "name": "overhead", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true", "diffuse": "0.9 0.9 0.9",
    }))
    for name, geom_type, pos, size, rgba in (
        ("red_cube", "box", "2.8 0.0 0.18", "0.18 0.18 0.18", "0.9 0.08 0.06 1"),
        ("green_cylinder", "cylinder", "3.2 0.7 0.20", "0.15 0.20", "0.05 0.75 0.12 1"),
        ("yellow_box", "box", "-2.6 -0.3 0.16", "0.20 0.13 0.16", "0.95 0.72 0.05 1"),
    ):
        worldbody.insert(1, ET.Element("geom", {
            "name": name, "type": geom_type, "pos": pos, "size": size, "rgba": rgba,
        }))
    for name, pos, size in (
        ("front_boundary", "5 0 1", "0.08 3 1"),
        ("rear_boundary", "-4 0 1", "0.08 3 1"),
        ("left_boundary", "0 3 1", "5 0.08 1"),
        ("right_boundary", "0 -3 1", "5 0.08 1"),
    ):
        worldbody.insert(1, ET.Element("geom", {
            "name": name, "type": "box", "pos": pos, "size": size, "rgba": "0.34 0.37 0.42 1",
        }))

    base.append(ET.Element("camera", {
        "name": "front_rgb", "pos": "0.42 0 0.15", "xyaxes": "0 1 0 -0.174 0 -0.985", "fovy": "65",
    }))
    base.append(ET.Element("camera", {
        "name": "rear_rgb", "pos": "-0.42 0 0.15", "xyaxes": "0 -1 0 0.174 0 -0.985", "fovy": "65",
    }))


def colour_pixel_counts(image: np.ndarray) -> dict[str, int]:
    red, green, blue = (image[..., index].astype(np.int16) for index in range(3))
    return {
        "red": int(np.count_nonzero((red > 90) & (red > green * 3 // 2) & (red > blue * 3 // 2))),
        "green": int(np.count_nonzero((green > 70) & (green > red * 3 // 2) & (green > blue * 3 // 2))),
        "yellow": int(np.count_nonzero((red > 90) & (green > 80) & (blue * 2 < red))),
    }


def planar_lidar(model: mujoco.MjModel, data: mujoco.MjData, ray_count: int = 72, max_range: float = 10.0) -> np.ndarray:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_pos = data.xpos[base_id].copy() + np.array((0.0, 0.0, 0.18))
    base_rot = data.xmat[base_id].reshape(3, 3)
    distances = np.empty(ray_count, dtype=np.float32)
    geomid = np.empty(1, dtype=np.int32)
    for index, heading in enumerate(np.linspace(-np.pi, np.pi, ray_count, endpoint=False)):
        direction = base_rot @ np.array((np.cos(heading), np.sin(heading), 0.0))
        distance = mujoco.mj_ray(model, data, base_pos, direction, None, 1, base_id, geomid)
        distances[index] = max_range if distance < 0.0 else min(float(distance), max_range)
    return distances


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Image width and height must be positive")
    asset = args.asset.resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"Build the floating M20 asset first: {asset}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(asset)
    append_scene(tree.getroot())
    scene_path = output_dir / "m20_visible_objects.xml"
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    images: dict[str, np.ndarray] = {}
    for camera in ("front_rgb", "rear_rgb"):
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
        images[camera] = image
        Image.fromarray(image).save(output_dir / f"{camera}.png")
    renderer.close()

    lidar = planar_lidar(model, data)
    state = np.concatenate((data.qpos, data.qvel)).astype(np.float32)
    np.save(output_dir / "lidar.npy", lidar)
    np.save(output_dir / "state.npy", state)
    counts = {name: colour_pixel_counts(image) for name, image in images.items()}
    if counts["front_rgb"]["red"] == 0 or counts["front_rgb"]["green"] == 0 or counts["rear_rgb"]["yellow"] == 0:
        raise RuntimeError(f"Expected visible objects are absent from RGB observations: {counts}")

    report = {
        "schema": "m20pro_mujoco_observation_smoke_v1",
        "asset": str(asset),
        "scene": str(scene_path),
        "instruction": "Go to the red cube.",
        "policy_input": {
            "front_rgb": {"shape": list(images["front_rgb"].shape), "path": "front_rgb.png"},
            "rear_rgb": {"shape": list(images["rear_rgb"].shape), "path": "rear_rgb.png"},
            "lidar": {"shape": list(lidar.shape), "path": "lidar.npy", "max_range_m": 10.0},
            "state": {"shape": list(state.shape), "path": "state.npy", "fields": ["qpos", "qvel"]},
            "language": "Go to the red cube.",
        },
        "prohibited_policy_input": ["target_world_position", "target_geometry_id", "semantic_mask", "simulator_object_pose"],
        "colour_pixel_counts": counts,
        "lidar_min_m": float(lidar.min()),
        "lidar_max_m": float(lidar.max()),
        "note": "This is an observation-contract smoke test, not a navigation result.",
    }
    (output_dir / "observation_contract.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

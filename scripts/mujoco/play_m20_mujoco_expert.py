#!/usr/bin/env python3
"""Replay one MuJoCo VLA collection expert episode from a third-person view."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.sim.mujoco import (
    ASSET, IMAGE_HEIGHT, IMAGE_WIDTH, OBJECTS,
    ObjectSpec, build_scene, open_video,
)
from m20pro_vla.low_level import M20LowLevelController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-json", type=Path,
        default=Path(".runtime/datasets/m20_mujoco_vla_v1/episode_0000.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(".runtime/videos/m20_mujoco_vla_v1/episode_0000_third_person.mp4"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def object_specs(metadata: dict) -> list[ObjectSpec]:
    catalog = {name: (label, kind, rgba) for name, label, kind, rgba in OBJECTS}
    result = []
    for item in metadata["objects"]:
        label, kind, rgba = catalog[item["name"]]
        size = (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)
        result.append(ObjectSpec(item["name"], label, kind, tuple(item["position"]), rgba, size))
    return result


def main() -> None:
    args = parse_args()
    if not args.episode_json.is_file():
        raise FileNotFoundError(args.episode_json)
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the asset first: {ASSET}")
    metadata = json.loads(args.episode_json.read_text(encoding="utf-8"))
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    scene_path = args.output.with_suffix(".scene.xml")
    build_scene(scene_path, object_specs(metadata))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = M20LowLevelController(model)
    controller.reset(data, float(math.atan2(target_xy[1], target_xy[0])))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.1
    camera.azimuth = 138.0
    camera.elevation = -24.0
    writer = open_video(args.output, args.width, args.height)
    reached = False
    try:
        for step in range(int(metadata["warmup_steps"]) + int(metadata["steps"])):
            base_xy = data.qpos[:2].copy()
            base_yaw = math.atan2(
                2.0 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]),
                1.0 - 2.0 * (data.qpos[5] ** 2 + data.qpos[6] ** 2),
            )
            delta = target_xy - base_xy
            distance = float(np.linalg.norm(delta))
            bearing = (math.atan2(float(delta[1]), float(delta[0])) - base_yaw + math.pi) % (2.0 * math.pi) - math.pi
            if reached or distance <= 0.65:
                reached = True
                action = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
            else:
                speed = 0.35 if distance > 1.0 else 0.16
                action = np.array((speed, 0.0, float(np.clip(0.35 * bearing, -0.12, 0.12)), 0.0), dtype=np.float64)
            controller.step(data, action)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            writer.stdin.write(renderer.render().tobytes())
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("third-person ffmpeg encoder failed")
        renderer.close()
        scene_path.unlink(missing_ok=True)
    print(json.dumps({"episode": str(args.episode_json), "video": str(args.output), "target": metadata["task_text"], "expert_only": True}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

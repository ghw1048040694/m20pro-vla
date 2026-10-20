"""Acceptance check: the teacher drives the robot through a structured scene.

Before :mod:`m20pro_vla.planning.global_planner` existed the only route the
teacher knew was a three-point detour around a single screen wall. That cannot
express "leave the start pose, thread the doorway, then turn towards an object
inside the room", which is exactly what the S2/S3 curricula need.

This script closes the loop on the real MuJoCo robot with the real low-level
controller and reports whether the teacher actually arrives, so the new
capability is *measured* rather than assumed:

* the planned route is solved on the privileged grid and drawn on a plan view;
* the executed trajectory is overlaid on the same view;
* the arrival distance, step count and stop latch are written to JSON.

Usage::

    python scripts/mujoco/preview_m20_global_route.py --scene s2 --video
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from m20pro_vla.low_level import build_low_level_controller
from m20pro_vla.planning import (
    GlobalPlanner,
    GlobalPlannerConfig,
    SearchMPCConfig,
    SearchMPCPlanner,
    yaw_from_quaternion_wxyz,
)
from m20pro_vla.sim.corridor import (
    CorridorEpisode,
    all_walls as corridor_all_walls,
    default_episode as corridor_default_episode,
    sample_episode as sample_corridor_episode,
)
from m20pro_vla.sim.mujoco import ObstacleSpec, build_scene, open_video
from m20pro_vla.sim.rooms import (
    RoomEpisode,
    default_episode as room_default_episode,
    room_obstacles,
    sample_episode as sample_room_episode,
)

WORKSPACE = Path(__file__).resolve().parents[2]

OBJECT_RGBA = {
    "red_cube": (229, 57, 53, 255),
    "green_cylinder": (67, 160, 71, 255),
    "yellow_box": (253, 216, 53, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("s2", "s3"), default="s2")
    parser.add_argument(
        "--episode",
        choices=("default", "sampled"),
        default="default",
        help="Use the hand-checked episode or a rejection-sampled one.",
    )
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--seed-index", type=int, default=0, help="How many sampled draws to skip.")
    parser.add_argument("--target-label", default="red cube")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--success-radius", type=float, default=0.95)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE / ".runtime/plan_preview")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also render a third-person video of the run.",
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Record one execution trace row every N policy steps (0 disables).",
    )
    return parser.parse_args()


def _build_episode(args: argparse.Namespace) -> tuple[RoomEpisode | CorridorEpisode, list[ObstacleSpec]]:
    if args.episode == "default":
        episode = room_default_episode() if args.scene == "s2" else corridor_default_episode()
    else:
        sampler = sample_room_episode if args.scene == "s2" else sample_corridor_episode
        rng = np.random.default_rng(args.seed)
        episode = None
        for _ in range(args.seed_index + 1):
            episode = sampler(rng, attempts=64)
        if episode is None:
            raise RuntimeError(f"the {args.scene} sampler refused every draw")
    walls = (
        room_obstacles(episode.spec)
        if args.scene == "s2"
        else corridor_all_walls(episode.spec, strict=False)
    )
    return episode, walls


def render_plan(
    path: Path,
    walls: list[ObstacleSpec],
    objects,
    waypoints: list[tuple[float, float]],
    trajectory: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    pixels_per_metre: int = 70,
    margin_m: float = 1.0,
) -> dict:
    xs_lo = min([w.position[0] - w.size[0] for w in walls] + [start_xy[0], goal_xy[0]]) - margin_m
    xs_hi = max([w.position[0] + w.size[0] for w in walls] + [start_xy[0], goal_xy[0]]) + margin_m
    ys_lo = min([w.position[1] - w.size[1] for w in walls] + [start_xy[1], goal_xy[1]]) - margin_m
    ys_hi = max([w.position[1] + w.size[1] for w in walls] + [start_xy[1], goal_xy[1]]) + margin_m
    width = int(math.ceil((xs_hi - xs_lo) * pixels_per_metre))
    height = int(math.ceil((ys_hi - ys_lo) * pixels_per_metre))

    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)

    def to_px(point) -> tuple[float, float]:
        return (
            (float(point[0]) - xs_lo) * pixels_per_metre,
            height - (float(point[1]) - ys_lo) * pixels_per_metre,
        )

    for obstacle in walls:
        x0, y0 = to_px((obstacle.position[0] - obstacle.size[0], obstacle.position[1] - obstacle.size[1]))
        x1, y1 = to_px((obstacle.position[0] + obstacle.size[0], obstacle.position[1] + obstacle.size[1]))
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=(70, 74, 80))

    if len(waypoints) >= 2:
        draw.line([to_px(p) for p in waypoints], fill=(230, 120, 30), width=3)
    for index, point in enumerate(waypoints):
        cx, cy = to_px(point)
        colour = (200, 90, 20) if index == len(waypoints) - 1 else (245, 170, 90)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=colour)

    if trajectory is not None and len(trajectory) >= 2:
        draw.line([to_px(p) for p in trajectory], fill=(30, 90, 200), width=2)

    for obj in objects:
        cx, cy = to_px(obj.position)
        colour = OBJECT_RGBA.get(obj.name, (120, 120, 120, 255))[:3]
        draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=colour, outline=(30, 30, 30))

    cx, cy = to_px(start_xy)
    draw.rectangle([cx - 6, cy - 6, cx + 6, cy + 6], fill=(20, 20, 20))
    cx, cy = to_px(goal_xy)
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(200, 30, 30), width=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"image": str(path), "width": width, "height": height}


def main() -> None:
    args = parse_args()
    episode, walls = _build_episode(args)
    target = next(obj for obj in episode.objects if obj.label == args.target_label)
    target_xy = np.asarray(target.position, dtype=np.float64)
    start_xy = np.asarray(episode.start_xy, dtype=np.float64)

    name = args.name or f"{args.scene}_{args.episode}_{args.target_label.replace(' ', '_')}"
    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / f"{name}.scene.xml"

    build_scene(scene_path, list(episode.objects), obstacles=walls)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model)
    controller.reset(data, float(episode.start_yaw), tuple(float(v) for v in start_xy))

    grid = GlobalPlanner(walls, GlobalPlannerConfig())
    initial_plan = grid.plan(start_xy, target_xy)
    planner = SearchMPCPlanner(
        model,
        target_xy,
        walls,
        SearchMPCConfig(),
        global_planner=grid,
    )

    renderer = None
    video = None
    if args.video:
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 6.0
        camera.azimuth = 132.0
        camera.elevation = -38.0
        camera.lookat[:] = np.array(
            (0.5 * (start_xy[0] + target_xy[0]), 0.5 * (start_xy[1] + target_xy[1]), 0.4)
        )
        video = open_video(out_dir / f"{name}.demo.mp4", args.video_width, args.video_height, fps=25)

    trajectory: list[tuple[float, float]] = []
    trace: list[dict] = []
    min_distance = float("inf")
    reached_step = -1
    stop_step = -1
    info: dict = {}
    try:
        for step in range(args.steps + args.warmup_steps):
            if step < args.warmup_steps:
                expert = np.zeros(4, dtype=np.float64)
            else:
                distance = float(np.linalg.norm(target_xy - np.asarray(data.qpos[:2], dtype=np.float64)))
                if distance <= args.success_radius:
                    if reached_step < 0:
                        reached_step = step - args.warmup_steps
                    expert = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                    if stop_step < 0:
                        stop_step = step - args.warmup_steps
                else:
                    expert, info = planner.recommend(data, controller)
            if step >= args.warmup_steps:
                current = np.asarray(data.qpos[:2], dtype=np.float64)
                trajectory.append((float(current[0]), float(current[1])))
                min_distance = min(min_distance, float(np.linalg.norm(target_xy - current)))
                if args.trace_every and (step - args.warmup_steps) % args.trace_every == 0:
                    yaw_now = yaw_from_quaternion_wxyz(np.asarray(data.qpos[3:7], dtype=np.float64))
                    trace.append(
                        {
                            "step": int(step - args.warmup_steps),
                            "xy": [round(float(current[0]), 3), round(float(current[1]), 3)],
                            "yaw_deg": round(math.degrees(float(yaw_now)), 1),
                            "bearing_deg": round(math.degrees(float(info.get("bearing", 0.0))), 1),
                            "route_goal": [round(float(v), 2) for v in info.get("planning_goal", [])],
                            "forward": round(float(info.get("best_forward", 0.0)), 3),
                            "yaw_cmd": round(float(info.get("best_yaw", 0.0)), 3),
                            "target_distance": round(float(info.get("current_distance", 0.0)), 3),
                            "replans": int(info.get("global_replans", 0)),
                        }
                    )
            controller.step(data, expert)
            if video is not None and renderer is not None:
                renderer.update_scene(data, camera=camera)
                video.stdin.write(renderer.render().tobytes())
            if reached_step >= 0 and (step - args.warmup_steps) >= reached_step + args.settle_steps:
                break
    finally:
        if video is not None:
            video.stdin.close()
            video.wait()
        if renderer is not None:
            renderer.close()

    trajectory_array = np.asarray(trajectory, dtype=np.float64) if trajectory else np.zeros((0, 2))
    plan_view = render_plan(
        out_dir / f"{name}_plan.png",
        walls,
        episode.objects,
        list(initial_plan.waypoints),
        trajectory_array,
        (float(start_xy[0]), float(start_xy[1])),
        (float(target_xy[0]), float(target_xy[1])),
    )

    report = {
        "schema": "m20pro_global_route_acceptance_v1",
        "scene": args.scene,
        "episode_kind": args.episode,
        "target_label": args.target_label,
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "target_xy": [float(target_xy[0]), float(target_xy[1])],
        "success_radius_m": float(args.success_radius),
        "initial_plan": initial_plan.as_dict(),
        "initial_waypoints": [[round(float(x), 3), round(float(y), 3)] for x, y in initial_plan.waypoints],
        "reached": bool(reached_step >= 0),
        "reached_step": int(reached_step),
        "stop_step": int(stop_step),
        "min_distance_m": round(float(min_distance), 4),
        "steps_executed": int(len(trajectory)),
        "planner_replans": int(info.get("global_replans", 0)) if info else 0,
        "trajectory_lost_plan": bool(not initial_plan.reachable),
        "final_xy": [round(float(data.qpos[0]), 3), round(float(data.qpos[1]), 3)],
        "final_base_height_m": round(float(data.qpos[2]), 4),
        "plan_view": plan_view,
        "trace": trace,
    }
    report_path = out_dir / f"{name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

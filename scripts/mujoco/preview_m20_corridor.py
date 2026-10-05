#!/usr/bin/env python3
"""Preview an S3 corridor scene: validate it, then render a plan and 3D views.

Review/inspection tool, not an experiment: one EGL context, no training, and
only images plus a JSON report are written.

    MUJOCO_GL=egl python scripts/mujoco/preview_m20_corridor.py

Outputs (default ``.runtime/scene_preview/s3_corridor/``):
    s3_corridor_plan.png        floor plan: rooms, doorways, occlusion rays, paths
    s3_corridor_overhead.png    MuJoCo 3D overhead view of the whole building
    s3_corridor_front.png       robot front camera at the start pose (outside)
    s3_corridor_pov.png         robot front camera just inside the corridor
    s3_corridor_policy_obs.png  the same view at the real contract resolution 80x48
    s3_corridor_report.json     validation + occlusion + reachability metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from m20pro_vla.low_level import build_low_level_controller
from m20pro_vla.sim.corridor import (
    CorridorEpisode,
    all_walls,
    default_episode,
    sample_episode,
    scene_report,
    touching_side,
)
from m20pro_vla.sim.mujoco import ObjectSpec, build_scene

INK = (38, 40, 46)
MUTED = (110, 114, 122)
GRID = (233, 233, 236)
WALL_FILL = (74, 78, 86)
WALL_EDGE = (44, 47, 53)
FLOOR_FILL = (247, 248, 250)
DOOR_FILL = (60, 165, 100)
RAY = (214, 68, 62)
PATH = (36, 96, 168)
START_RING = (36, 96, 168)
ROOM_TAG = (150, 154, 162)

FONT_CANDIDATES = (
    ("/mnt/c/Windows/Fonts/msyh.ttc", True),
    ("/mnt/c/Windows/Fonts/msyh.ttf", True),
    ("/mnt/c/Windows/Fonts/simhei.ttf", True),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", True),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", True),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", True),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", False),
)


def _resolve_font() -> tuple[str | None, bool]:
    for path, cjk in FONT_CANDIDATES:
        if os.path.isfile(path):
            return path, cjk
    return None, False


_FONT_PATH, _CJK = _resolve_font()


def _font(size: int) -> ImageFont.ImageFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _t(zh: str, en: str) -> str:
    return zh if _CJK else en


def _dashed_line(draw, p0, p1, fill, width=2, dash=9.0, gap=6.0) -> None:
    x0, y0 = p0
    x1, y1 = p1
    total = math.hypot(x1 - x0, y1 - y0)
    if total <= 1.0e-6:
        return
    dx, dy = (x1 - x0) / total, (y1 - y0) / total
    pos = 0.0
    while pos < total:
        end = min(pos + dash, total)
        draw.line(
            [(x0 + dx * pos, y0 + dy * pos), (x0 + dx * end, y0 + dy * end)],
            fill=fill,
            width=width,
        )
        pos = end + gap


def _dashed_circle(draw, centre, radius, fill, width=2, segments=64) -> None:
    cx, cy = centre
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    step = 360.0 / segments
    index = 0.0
    while index < 360.0:
        draw.arc(box, index, index + step * 0.55, fill=fill, width=width)
        index += step


def _object_shape(obj: ObjectSpec, scale: float, to_px) -> dict:
    cx, cy = to_px(float(obj.position[0]), float(obj.position[1]))
    if obj.kind == "cylinder":
        radius = float(obj.size[0]) * scale
        return {"kind": "ellipse", "box": [cx - radius, cy - radius, cx + radius, cy + radius], "centre": (cx, cy)}
    hx, hy = float(obj.size[0]) * scale, float(obj.size[1]) * scale
    return {"kind": "rect", "box": [cx - hx, cy - hy, cx + hx, cy + hy], "centre": (cx, cy)}


def render_plan(
    episode: CorridorEpisode, reach: dict, path: Path, scale: float = 96.0
) -> Path:
    spec = episode.spec
    walls = all_walls(spec)

    xs = [spec.corridor_x[0], spec.corridor_x[1], episode.start_xy[0]]
    ys = [spec.corridor_y[0], spec.corridor_y[1], episode.start_xy[1]]
    for wing in spec.rooms:
        xs += [wing.interior_x[0], wing.interior_x[1]]
        ys += [wing.interior_y[0], wing.interior_y[1]]
    for obj in episode.objects:
        xs.append(float(obj.position[0]))
        ys.append(float(obj.position[1]))

    pad = 0.70
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    margin_left, margin_top = 30, 78
    margin_right, margin_bottom = 30, 118
    width = int((xmax - xmin) * scale) + margin_left + margin_right
    height = int((ymax - ymin) * scale) + margin_top + margin_bottom

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (margin_left + (x - xmin) * scale, margin_top + (ymax - y) * scale)

    def rect(box_x, box_y, fill, outline=None, width_px=1):
        p0 = to_px(box_x[0], box_y[1])
        p1 = to_px(box_x[1], box_y[0])
        draw.rectangle([p0[0], p0[1], p1[0], p1[1]], fill=fill, outline=outline, width=width_px)

    # Grid.
    gx = math.floor(xmin)
    while gx <= math.ceil(xmax):
        draw.line([to_px(gx, ymin), to_px(gx, ymax)], fill=GRID, width=1)
        gx += 1
    gy = math.floor(ymin)
    while gy <= math.ceil(ymax):
        draw.line([to_px(xmin, gy), to_px(xmax, gy)], fill=GRID, width=1)
        gy += 1

    # Floor plates: corridor plus every room.
    rect(spec.corridor_x, spec.corridor_y, FLOOR_FILL)
    for wing in spec.rooms:
        rect(wing.interior_x, wing.interior_y, FLOOR_FILL)

    # Doorways (drawn before the walls so the stubs sit on top of the band).
    for wing in spec.rooms:
        side = touching_side(wing, spec)
        gap_lo, gap_hi = wing.door_gap()
        if side in ("north", "south"):
            line = wing.interior_y[0] - 0.5 * spec.wall_thickness if side == "north" else wing.interior_y[1] + 0.5 * spec.wall_thickness
            rect((gap_lo, gap_hi), (line - 0.09, line + 0.09), DOOR_FILL)
        else:
            line = wing.interior_x[0] - 0.5 * spec.wall_thickness if side == "east" else wing.interior_x[1] + 0.5 * spec.wall_thickness
            rect((line - 0.09, line + 0.09), (gap_lo, gap_hi), DOOR_FILL)
    e_lo, e_hi = spec.entrance_gap()
    ex = spec.corridor_x[0] - 0.5 * spec.wall_thickness
    rect((ex - 0.09, ex + 0.09), (e_lo, e_hi), DOOR_FILL)

    # Reachability paths: the proof that the scene is solvable.
    for name, entry in reach["open"]["rooms"].items():
        points = entry.get("path_xy") or []
        if len(points) < 2:
            continue
        draw.line([to_px(x, y) for x, y in points], fill=PATH, width=3)

    # Walls.
    for wall in walls:
        wx, wy = float(wall.position[0]), float(wall.position[1])
        hx, hy = float(wall.size[0]), float(wall.size[1])
        rect((wx - hx, wx + hx), (wy - hy, wy + hy), WALL_FILL, WALL_EDGE, 1)

    # Occlusion proof: every start-to-target ray crosses a wall.
    for obj in episode.objects:
        _dashed_line(
            draw,
            to_px(float(episode.start_xy[0]), float(episode.start_xy[1])),
            to_px(float(obj.position[0]), float(obj.position[1])),
            RAY,
            width=2,
            dash=9.0,
            gap=6.0,
        )

    # Room tags.
    for wing in spec.rooms:
        cx, cy = wing.interior_center
        px, py = to_px(cx, wing.interior_y[1] - 0.30)
        draw.text((px, py), wing.name, fill=ROOM_TAG, font=_font(14), anchor="mb")

    # Task objects, with the room they belong to.
    for obj in episode.objects:
        shape = _object_shape(obj, scale, to_px)
        colour = tuple(int(round(255 * c)) for c in obj.rgba[:3])
        if shape["kind"] == "ellipse":
            draw.ellipse(shape["box"], fill=colour, outline=INK, width=2)
        else:
            draw.rectangle(shape["box"], fill=colour, outline=INK, width=2)
        cx, cy = shape["centre"]
        draw.text((cx, cy - 26), obj.label, fill=INK, font=_font(15), anchor="mb")

    # Start pose.
    sx, sy = float(episode.start_xy[0]), float(episode.start_xy[1])
    radius = 0.375 * scale
    centre = to_px(sx, sy)
    _dashed_circle(draw, centre, radius, START_RING, width=2)
    head = (
        centre[0] + math.cos(episode.start_yaw) * radius * 2.6,
        centre[1] - math.sin(episode.start_yaw) * radius * 2.6,
    )
    draw.line([centre, head], fill=START_RING, width=3)
    draw.ellipse([head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5], fill=START_RING)
    draw.text((centre[0], centre[1] + radius + 8), _t("起点 朝 +X", "start facing +X"), fill=START_RING, font=_font(15), anchor="ma")

    # Titles.
    draw.text(
        (margin_left, 20),
        _t(
            f"S3 走廊多房间场景   走廊 {spec.corridor_length:.2f} × {spec.corridor_width:.2f} m   "
            f"{len(spec.rooms)} 个房间  每房一个物体",
            f"S3 corridor + rooms   corridor {spec.corridor_length:.2f} x {spec.corridor_width:.2f} m   "
            f"{len(spec.rooms)} rooms, one object each",
        ),
        fill=INK,
        font=_font(20),
    )
    draw.text(
        (margin_left, 46),
        _t(
            f"门洞净宽 {spec.rooms[0].door_width:.2f} m   机器人足迹 ⌀0.75 m   墙厚 {spec.wall_thickness:.2f} m 墙高 {spec.wall_height:.2f} m",
            f"door clear {spec.rooms[0].door_width:.2f} m   footprint 0.75 m   wall {spec.wall_thickness:.2f} m thick, {spec.wall_height:.2f} m high",
        ),
        fill=MUTED,
        font=_font(15),
    )

    legend = [
        (_t("墙（遮挡相机与激光）", "wall: occludes camera and LiDAR"), WALL_FILL),
        (_t("门洞（可通行）", "doorway (passable)"), DOOR_FILL),
        (_t("起点→每个物体 视线全部被墙挡住", "start->object rays: all blocked"), RAY),
        (_t("起点到各房间的可达路径（栅格 BFS）", "reachable path to each room (grid BFS)"), PATH),
    ]
    legend_y = height - margin_bottom + 16
    for index, (text, colour) in enumerate(legend):
        col = index % 2
        row = index // 2
        lx = margin_left + col * 330
        ly = legend_y + row * 24
        draw.rectangle([lx, ly + 3, lx + 14, ly + 15], fill=colour)
        draw.text((lx + 22, ly + 2), text, fill=MUTED, font=_font(14))

    draw.text(
        (margin_left, legend_y + 54),
        _t(
            "密封校验：把门洞全部堵上后，BFS 无法进入任何房间 -> 墙无缝隙，房间只能从门进",
            "seal test: with doorways plugged, BFS enters no room -> walls are airtight",
        ),
        fill=MUTED,
        font=_font(14),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _annotate_bottom(path: Path, title: str, subtitle: str) -> Path:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    bar = 56
    canvas = Image.new("RGB", (width, height + bar), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, height + 8), title, fill=INK, font=_font(15))
    draw.text((16, height + 31), subtitle, fill=MUTED, font=_font(13))
    canvas.save(path)
    return path


def render_3d(scene_path: Path, episode: CorridorEpisode, out_dir: Path) -> dict[str, Path]:
    spec = episode.spec
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model, backend="analytic")
    controller.reset(data, yaw=float(episode.start_yaw), base_xy=episode.start_xy)

    outputs: dict[str, Path] = {}
    renderer = mujoco.Renderer(model, height=420, width=640)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    building_x = 0.5 * (episode.start_xy[0] + max(w.interior_x[1] for w in spec.rooms))
    camera.lookat[:] = (building_x, 0.0, 0.30)
    camera.distance = 12.5
    camera.elevation = -74.0
    camera.azimuth = 90.0
    renderer.update_scene(data, camera=camera)
    overhead = out_dir / "s3_corridor_overhead.png"
    Image.fromarray(renderer.render()).save(overhead)
    _annotate_bottom(
        overhead,
        _t("S3 走廊 + 三个房间 · 俯视 3D", "S3 corridor + three rooms - overhead 3D"),
        _t(
            "机器人在走廊外（左）；北房、南房、东端房各有一个目标，全部被墙挡住",
            "robot outside the corridor (left); one object per room, all hidden by walls",
        ),
    )
    outputs["overhead"] = overhead

    renderer.update_scene(data, camera="front_rgb")
    front = out_dir / "s3_corridor_front.png"
    Image.fromarray(renderer.render()).save(front)
    _annotate_bottom(
        front,
        _t("策略相机 front_rgb @ 起点（走廊外）", "policy camera front_rgb at the start pose (outside)"),
        _t(
            "只能看到入口，看不到任何房间与目标 -> 遮挡成立，必须先选门",
            "only the entrance is visible; no room or target -> must pick a doorway",
        ),
    )
    outputs["front"] = front

    # A second pose: just inside the entrance, looking down the corridor.
    controller.reset(data, yaw=0.0, base_xy=(spec.corridor_x[0] + 0.45, 0.0))
    renderer.update_scene(data, camera="front_rgb")
    pov = out_dir / "s3_corridor_pov.png"
    Image.fromarray(renderer.render()).save(pov)
    _annotate_bottom(
        pov,
        _t("走廊内视角 front_rgb（站在入口内，朝 +X）", "in-corridor front_rgb (just inside, facing +X)"),
        _t(
            "看得见走廊与尽头的门，但两个侧房的门在画面外、房内目标依然不可见",
            "corridor and the far door are visible; side doors are off-frame, objects still hidden",
        ),
    )
    outputs["pov"] = pov
    renderer.close()

    # What the VLA actually consumes: the contract resolution.
    small_renderer = mujoco.Renderer(model, height=48, width=80)
    small_renderer.update_scene(data, camera="front_rgb")
    small = Image.fromarray(small_renderer.render())
    small_renderer.close()
    zoom = 8
    policy_obs = out_dir / "s3_corridor_policy_obs.png"
    small.resize((80 * zoom, 48 * zoom), Image.NEAREST).save(policy_obs)
    _annotate_bottom(
        policy_obs,
        _t("策略真实输入 front_rgb = 80x48（8x 最近邻放大，一方块 = 一像素）",
           "actual policy input front_rgb = 80x48 (8x nearest, one block = one pixel)"),
        _t(
            "走廊比单房间更深，这条走廊最远超 9 m，分辨率瓶颈比 S2 更紧",
            "the corridor is deeper than one room; the far end is >9 m away, so resolution matters more than in S2",
        ),
    )
    outputs["policy_obs"] = policy_obs
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview and validate an S3 corridor scene.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("M20PRO_VLA_DATA_ROOT", ".runtime")) / "scene_preview/s3_corridor",
    )
    parser.add_argument("--scene-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Sample a jittered episode instead.")
    parser.add_argument("--resolution", type=float, default=0.05, help="BFS grid resolution in metres.")
    parser.add_argument("--no-3d", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        episode = default_episode()
        origin = "default"
    else:
        rng = np.random.default_rng(args.seed)
        episode = sample_episode(rng)
        if episode is None:
            raise SystemExit("corridor sampler rejected every attempt; constraints are unsatisfiable")
        origin = f"sampled seed={args.seed}"

    report = scene_report(episode, resolution=args.resolution)
    validation = report["validation"]
    occlusion = report["occlusion"]
    reach = report["reachability"]

    scene_path = args.scene_path or (out_dir / "s3_corridor.scene.xml")
    build_scene(
        scene_path,
        objects=list(episode.objects),
        obstacles=all_walls(episode.spec),
    )

    plan = render_plan(episode, reach, out_dir / "s3_corridor_plan.png")
    renders = {} if args.no_3d else render_3d(scene_path, episode, out_dir)

    summary = {
        "origin": origin,
        "scene_path": str(scene_path),
        "ok": report["ok"],
        "validation": {
            key: value for key, value in validation.items() if key != "walls"
        },
        "occlusion": occlusion,
        "placement": report["placement"],
        "reachability": {
            "resolution_m": reach["resolution_m"],
            "ok": reach["ok"],
            "open": {
                "start_cell_free": reach["open"]["start_cell_free"],
                "all_rooms_reachable": reach["open"]["all_rooms_reachable"],
                "rooms": {
                    name: {
                        "reachable": entry["reachable"],
                        "path_length_cells": entry.get("path_length_cells"),
                    }
                    for name, entry in reach["open"]["rooms"].items()
                },
            },
            "sealed": {
                "no_room_reachable": reach["sealed"]["no_room_reachable"],
                "rooms": {
                    name: {"reachable": entry["reachable"]}
                    for name, entry in reach["sealed"]["rooms"].items()
                },
            },
        },
        "corridor": {
            "x": [float(v) for v in episode.spec.corridor_x],
            "y": [float(v) for v in episode.spec.corridor_y],
            "entrance_width_m": float(episode.spec.entrance_width),
        },
        "rooms": [
            {
                "name": wing.name,
                "interior_x": [float(v) for v in wing.interior_x],
                "interior_y": [float(v) for v in wing.interior_y],
                "door_center": float(wing.door_center),
                "door_width_m": float(wing.door_width),
                "attached_to_corridor": touching_side(wing, episode.spec),
                "object_name": wing.object_name,
            }
            for wing in episode.spec.rooms
        ],
        "walls": [
            {
                "name": wall.name,
                "position": [float(v) for v in wall.position],
                "size": [float(v) for v in wall.size],
            }
            for wall in validation["walls"]
        ],
        "start_xy": [float(v) for v in episode.start_xy],
        "start_yaw": float(episode.start_yaw),
        "objects": [
            {
                "name": obj.name,
                "label": obj.label,
                "position": [float(v) for v in obj.position],
                "room": report["placement"]["placement"].get(obj.name),
            }
            for obj in episode.objects
        ],
        "images": {"plan": str(plan), **{k: str(v) for k, v in renders.items()}},
    }
    report_path = out_dir / "s3_corridor_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = validation["metrics"]
    print("=" * 78)
    print(f"S3 corridor preview ({origin})")
    print("=" * 78)
    print(f"overall               : {'PASS' if report['ok'] else 'FAIL'}")
    print(f"geometry validation   : {'PASS' if validation['ok'] else 'FAIL'}")
    for issue in validation["issues"]:
        print(f"  ! {issue}")
    print(f"corridor              : {metrics['corridor_length_m']:.2f} m long x {metrics['corridor_width_m']:.2f} m wide")
    print(f"rooms                 : {metrics['room_count']}  min room gap {metrics['min_room_gap_m']:.2f} m")
    print(f"door clear width      : {metrics['min_door_width_m']:.2f} m (required {metrics['door_required_width_m']:.2f} m)")
    print(f"wall boxes            : {metrics['wall_count']}  (+{len(episode.spec.rooms) + 1} seal plugs)")
    for wing in episode.spec.rooms:
        side = touching_side(wing, episode.spec)
        print(
            f"  - {wing.name:12s} on corridor {side:5s}  interior {wing.width:.2f} x {wing.height:.2f} m  "
            f"door {wing.door_width:.2f} m  object={wing.object_name}"
        )
    print(f"objects occluded      : {'ALL' if occlusion['all_blocked'] else 'NO'}")
    for name, blocked in occlusion["per_object"].items():
        print(f"  - {name:16s} room={report['placement']['placement'].get(name):12s} blocked_from_start={blocked}")
    print(f"reachability open     : {'PASS' if reach['open']['all_rooms_reachable'] else 'FAIL'}")
    for name, entry in reach["open"]["rooms"].items():
        cells = entry.get("path_length_cells")
        print(f"  - {name:12s} reachable={entry['reachable']}  path={cells} cells ({cells * reach['resolution_m']:.2f} m)")
    print(f"reachability sealed   : {'PASS (no room reachable)' if reach['sealed']['no_room_reachable'] else 'FAIL (a leak!)'}")
    print(f"scene xml             : {scene_path}")
    print(f"atlas images          : {plan.name}" + "".join(f", {p.name}" for p in renders.values()))
    print(f"report json           : {report_path.name}")
    print("=" * 78)


if __name__ == "__main__":
    main()

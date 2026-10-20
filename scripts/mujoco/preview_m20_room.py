#!/usr/bin/env python3
"""Preview an S2 room scene: validate it, then render a plan and 3D views.

This is a review/inspection tool, not an experiment: it consumes no GPU beyond
a single EGL context, trains nothing, and writes only images plus a JSON report
under the run directory.

    MUJOCO_GL=egl python scripts/mujoco/preview_m20_room.py

Outputs (default ``.runtime/scene_preview/s2_room/``):
    s2_room_plan.png        top-down floor plan with the occlusion proof
    s2_room_overhead.png    MuJoCo 3D overhead view
    s2_room_front.png       robot front camera at the start pose
    s2_room_report.json     validation + occlusion metrics
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
from m20pro_vla.sim.mujoco import ObjectSpec, build_scene
from m20pro_vla.sim.rooms import (
    ROBOT_FOOTPRINT_DIAMETER_M,
    ROBOT_FOOTPRINT_RADIUS_M,
    RoomEpisode,
    default_episode,
    occlusion_report,
    room_obstacles,
    sample_episode,
    validate_room,
)

INK = (38, 40, 46)
MUTED = (110, 114, 122)
GRID = (233, 233, 236)
WALL_FILL = (74, 78, 86)
WALL_EDGE = (44, 47, 53)
DOOR_FILL = (60, 165, 100)
RAY = (214, 68, 62)
START_RING = (36, 96, 168)

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


def _dashed_line(draw, p0, p1, fill, width=2, dash=8.0, gap=6.0) -> None:
    x0, y0 = p0
    x1, y1 = p1
    total = math.hypot(x1 - x0, y1 - y0)
    if total <= 1.0e-6:
        return
    dx, dy = (x1 - x0) / total, (y1 - y0) / total
    pos = 0.0
    while pos < total:
        end = min(pos + dash, total)
        draw.line([(x0 + dx * pos, y0 + dy * pos), (x0 + dx * end, y0 + dy * end)], fill=fill, width=width)
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
    x, y = float(obj.position[0]), float(obj.position[1])
    cx, cy = to_px(x, y)
    if obj.kind == "cylinder":
        radius = float(obj.size[0]) * scale
        return {"kind": "ellipse", "box": [cx - radius, cy - radius, cx + radius, cy + radius], "centre": (cx, cy)}
    hx, hy = float(obj.size[0]) * scale, float(obj.size[1]) * scale
    return {"kind": "rect", "box": [cx - hx, cy - hy, cx + hx, cy + hy], "centre": (cx, cy)}


def render_plan(episode: RoomEpisode, path: Path, scale: float = 118.0) -> Path:
    spec = episode.spec
    walls = room_obstacles(spec)
    x0, x1 = spec.interior_x
    y0, y1 = spec.interior_y

    xs = [x0, x1, episode.start_xy[0]] + [float(o.position[0]) for o in episode.objects]
    ys = [y0, y1, episode.start_xy[1]] + [float(o.position[1]) for o in episode.objects]
    pad = 0.85
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    margin_left, margin_top = 30, 62
    margin_right, margin_bottom = 30, 96
    width = int((xmax - xmin) * scale) + margin_left + margin_right
    height = int((ymax - ymin) * scale) + margin_top + margin_bottom

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (
            margin_left + (x - xmin) * scale,
            margin_top + (ymax - y) * scale,
        )

    # One-metre grid.
    gx = math.floor(xmin)
    while gx <= math.ceil(xmax):
        p0 = to_px(gx, ymin)
        p1 = to_px(gx, ymax)
        draw.line([p0, p1], fill=GRID, width=1)
        gx += 1
    gy = math.floor(ymin)
    while gy <= math.ceil(ymax):
        p0 = to_px(xmin, gy)
        p1 = to_px(xmax, gy)
        draw.line([p0, p1], fill=GRID, width=1)
        gy += 1

    # Doorway band first, so the wall boxes are drawn on top of its flanks.
    (d0x, d0y), (d1x, d1y) = spec.door_world_segment()
    a = to_px(d0x, d0y)
    b = to_px(d1x, d1y)
    lo = (min(a[0], b[0]) - 7, min(a[1], b[1]) - 7)
    hi = (max(a[0], b[0]) + 7, max(a[1], b[1]) + 7)
    draw.rectangle([lo[0], lo[1], hi[0], hi[1]], fill=DOOR_FILL)

    # Walls.
    for wall in walls:
        wx, wy = float(wall.position[0]), float(wall.position[1])
        hx, hy = float(wall.size[0]), float(wall.size[1])
        p0 = to_px(wx - hx, wy + hy)
        p1 = to_px(wx + hx, wy - hy)
        draw.rectangle([p0[0], p0[1], p1[0], p1[1]], fill=WALL_FILL, outline=WALL_EDGE, width=1)

    # Occlusion proof: every start-to-target ray crosses a wall.
    for obj in episode.objects:
        p0 = to_px(float(episode.start_xy[0]), float(episode.start_xy[1]))
        p1 = to_px(float(obj.position[0]), float(obj.position[1]))
        _dashed_line(draw, p0, p1, RAY, width=2, dash=9.0, gap=6.0)

    # Task objects.
    for obj in episode.objects:
        shape = _object_shape(obj, scale, to_px)
        colour = tuple(int(round(255 * c)) for c in obj.rgba[:3])
        if shape["kind"] == "ellipse":
            draw.ellipse(shape["box"], fill=colour, outline=INK, width=2)
        else:
            draw.rectangle(shape["box"], fill=colour, outline=INK, width=2)
        cx, cy = shape["centre"]
        draw.text((cx, cy - 30), _t(obj.label.split()[0], obj.label), fill=INK, font=_font(15), anchor="mb")

    # Robot start pose: footprint ring plus heading arrow.
    sx, sy = float(episode.start_xy[0]), float(episode.start_xy[1])
    centre = to_px(sx, sy)
    radius = ROBOT_FOOTPRINT_RADIUS_M * scale
    _dashed_circle(draw, centre, radius, START_RING, width=2)
    head = (
        centre[0] + math.cos(episode.start_yaw) * radius * 2.6,
        centre[1] - math.sin(episode.start_yaw) * radius * 2.6,
    )
    draw.line([centre, head], fill=START_RING, width=3)
    draw.ellipse(
        [head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5], fill=START_RING
    )
    draw.text(
        (centre[0], centre[1] + radius + 8),
        _t("起点  朝向 +X", "start  facing +X"),
        fill=START_RING,
        font=_font(15),
        anchor="ma",
    )

    # Captions.
    title = _t(
        f"S2 单房间场景  内空 {x1 - x0:.2f} × {y1 - y0:.2f} m   门洞净宽 {spec.door_width:.2f} m",
        f"S2 single room  interior {x1 - x0:.2f} x {y1 - y0:.2f} m  door clear {spec.door_width:.2f} m",
    )
    draw.text((margin_left, 22), title, fill=INK, font=_font(20))

    side_clear = 0.5 * (spec.door_width - ROBOT_FOOTPRINT_DIAMETER_M)
    legend = [
        (_t("墙（遮挡视线与激光）", "wall: occludes camera and LiDAR"), WALL_FILL),
        (_t("门洞（可通行）", "doorway (passable)"), DOOR_FILL),
        (_t("起点到目标视线（全部被墙挡住）", "start-to-target rays: all blocked"), RAY),
        (
            _t(
                f"机器人足迹 ⌀{ROBOT_FOOTPRINT_DIAMETER_M:.2f} m，单侧余量 {side_clear:.2f} m",
                f"robot footprint {ROBOT_FOOTPRINT_DIAMETER_M:.2f} m, side clearance {side_clear:.2f} m",
            ),
            START_RING,
        ),
    ]
    legend_y = height - margin_bottom + 14
    for index, (text, colour) in enumerate(legend):
        col = index % 2
        row = index // 2
        lx = margin_left + col * 340
        ly = legend_y + row * 24
        draw.rectangle([lx, ly + 3, lx + 14, ly + 15], fill=colour)
        draw.text((lx + 22, ly + 2), text, fill=MUTED, font=_font(14))

    # Dimension annotations.
    dim = _t(
        f"内空宽 {x1 - x0:.2f} m  ×  内空高 {y1 - y0:.2f} m   |   墙高 {spec.wall_height:.2f} m  墙厚 {spec.wall_thickness:.2f} m",
        f"interior {x1 - x0:.2f} x {y1 - y0:.2f} m  |  wall height {spec.wall_height:.2f} m  thickness {spec.wall_thickness:.2f} m",
    )
    draw.text((margin_left, 44), dim, fill=MUTED, font=_font(15))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _annotate_bottom(path: Path, title: str, subtitle: str) -> Path:
    """Append a caption bar so a raw render is self-explanatory on review."""
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


def render_3d(scene_path: Path, episode: RoomEpisode, out_dir: Path) -> dict[str, Path]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = build_low_level_controller(model, backend="analytic")
    controller.reset(data, yaw=float(episode.start_yaw), base_xy=episode.start_xy)

    cx, cy = episode.spec.interior_center
    outputs: dict[str, Path] = {}
    renderer = mujoco.Renderer(model, height=420, width=640)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.5 * (episode.start_xy[0] + cx), cy, 0.30)
    camera.distance = 9.5
    camera.elevation = -62.0
    camera.azimuth = 88.0
    renderer.update_scene(data, camera=camera)
    overhead = out_dir / "s2_room_overhead.png"
    Image.fromarray(renderer.render()).save(overhead)
    _annotate_bottom(
        overhead,
        _t("S2 单房间 · 俯视 3D", "S2 single room - overhead 3D"),
        _t(
            "机器人在房外（左），门洞开在左墙下段，3 个目标都在房内 -> 必须先找到门才看得到目标",
            "robot outside (left); doorway low on the left wall; all 3 objects inside",
        ),
    )
    outputs["overhead"] = overhead

    renderer.update_scene(data, camera="front_rgb")
    front = out_dir / "s2_room_front.png"
    Image.fromarray(renderer.render()).save(front)
    _annotate_bottom(
        front,
        _t("策略相机 front_rgb @ 起点（评审分辨率 640x420）", "policy camera front_rgb at start (640x420)"),
        _t(
            "画面被墙面占满，看不到任何目标 -> 遮挡成立，这是搜索任务而非指向任务",
            "wall fills the frame, no target visible -> occlusion holds",
        ),
    )
    outputs["front"] = front
    renderer.close()

    # What the VLA actually consumes: the same camera at the contract resolution.
    observation_renderer = mujoco.Renderer(model, height=48, width=80)
    observation_renderer.update_scene(data, camera="front_rgb")
    small = Image.fromarray(observation_renderer.render())
    observation_renderer.close()
    zoom = 8
    policy_obs = out_dir / "s2_room_policy_obs.png"
    small.resize((80 * zoom, 48 * zoom), Image.NEAREST).save(policy_obs)
    _annotate_bottom(
        policy_obs,
        _t("策略真实输入 front_rgb = 80x48（8x 最近邻放大，方块 = 单个像素）",
           "actual policy input front_rgb = 80x48 (8x nearest, one block = one pixel)"),
        _t(
            "这是 VLA 真正看到的东西。隔 8 m 分辨绿圆柱/黄盒子是否够用，是需要独立验证的第二个瓶颈",
            "this is what the VLA really sees; long-range colour discrimination is a separate risk",
        ),
    )
    outputs["policy_obs"] = policy_obs
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview and validate an S2 room scene.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get("M20PRO_VLA_DATA_ROOT", ".runtime")
        )
        / "scene_preview/s2_room",
    )
    parser.add_argument("--scene-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Sample a jittered episode instead.")
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
            raise SystemExit("room sampler rejected every attempt; constraints are unsatisfiable")
        origin = f"sampled seed={args.seed}"

    report = validate_room(episode.spec)
    occlusion = occlusion_report(episode)

    scene_path = args.scene_path or (out_dir / "s2_room.scene.xml")
    build_scene(
        scene_path,
        objects=list(episode.objects),
        obstacles=room_obstacles(episode.spec),
    )

    plan = render_plan(episode, out_dir / "s2_room_plan.png")
    renders = {} if args.no_3d else render_3d(scene_path, episode, out_dir)

    summary = {
        "origin": origin,
        "scene_path": str(scene_path),
        "validation": {key: value for key, value in report.items() if key != "walls"},
        "walls": [
            {
                "name": wall.name,
                "position": [float(value) for value in wall.position],
                "size": [float(value) for value in wall.size],
            }
            for wall in report["walls"]
        ],
        "occlusion": occlusion,
        "start_xy": [float(value) for value in episode.start_xy],
        "start_yaw": float(episode.start_yaw),
        "objects": [
            {
                "name": obj.name,
                "label": obj.label,
                "position": [float(value) for value in obj.position],
            }
            for obj in episode.objects
        ],
        "images": {"plan": str(plan), **{k: str(v) for k, v in renders.items()}},
    }
    report_path = out_dir / "s2_room_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = report["metrics"]
    print("=" * 72)
    print(f"S2 room preview ({origin})")
    print("=" * 72)
    print(f"validation            : {'PASS' if report['ok'] else 'FAIL'}")
    for issue in report["issues"]:
        print(f"  ! {issue}")
    print(f"interior              : {metrics['interior_width_m']:.2f} x {metrics['interior_height_m']:.2f} m")
    print(f"door clear width      : {metrics['door_clear_width_m']:.2f} m (required {metrics['door_required_width_m']:.2f} m)")
    print(f"robot footprint       : {metrics['robot_footprint_diameter_m']:.2f} m diameter, {metrics['door_side_clearance_m']:.2f} m side clearance")
    print(f"wall boxes            : {metrics['wall_count']}")
    print(f"objects occluded      : {'ALL' if occlusion['all_blocked'] else 'NO'}")
    for name, blocked in occlusion["per_object"].items():
        print(f"  - {name:16s} blocked_from_start={blocked}")
    print(f"scene xml             : {scene_path}")
    print(f"atlas images          : {plan.name}" + "".join(f", {p.name}" for p in renders.values()))
    print(f"report json           : {report_path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()

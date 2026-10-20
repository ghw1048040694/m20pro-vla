"""MuJoCo-native M20 VLA scene, low-level bridge, and sensor contract."""

from __future__ import annotations

import math
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.low_level import M20LowLevelController

# Compatibility export for historical replayers. New modules import the
# canonical class from m20pro_vla.low_level.
M20NativeController = M20LowLevelController


WORKSPACE = Path(__file__).resolve().parents[3]
ASSET = WORKSPACE / ".runtime/mujoco_assets/M20_floating_actuated.xml"
MESH_ROOT = WORKSPACE / "src" / "m20pro_description" / "meshes"
CONTROL_DT = 0.02
PHYSICS_STEPS = 8
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 48
LIDAR_RAYS = 72
MAX_LIDAR_RANGE = 10.0
OBJECTS = (
    ("red_cube", "red cube", "box", (0.90, 0.08, 0.05, 1.0)),
    ("green_cylinder", "green cylinder", "cylinder", (0.05, 0.78, 0.12, 1.0)),
    ("yellow_box", "yellow box", "box", (0.95, 0.72, 0.05, 1.0)),
)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    label: str
    kind: str
    position: tuple[float, float]
    rgba: tuple[float, float, float, float]
    size: tuple[float, ...]


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    kind: str
    position: tuple[float, float, float]
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (0.30, 0.33, 0.36, 1.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contype: int = 1
    conaffinity: int = 1
    group: int = 0
    density: float | None = None


@dataclass(frozen=True)
class TerrainSpec:
    """Deterministic MuJoCo terrain profile used by low-level gates."""

    name: str
    version: str = "m20_rough_terrain_v1"


@dataclass(frozen=True)
class SceneLightSpec:
    pos: tuple[float, float, float] = (0.0, 0.0, 5.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    diffuse: tuple[float, float, float] = (0.85, 0.85, 0.85)
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    directional: bool = True


TERRAIN_PROFILES = ("flat", "slope", "bumps", "step")


def _format(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.4f}" for value in values)


def _append_terrain(worldbody: ET.Element, terrain: TerrainSpec | None) -> None:
    """Add modest physical obstacles while retaining the ground plane."""
    if terrain is None or terrain.name == "flat":
        return
    if terrain.name not in TERRAIN_PROFILES:
        raise ValueError(f"Unknown M20 terrain profile: {terrain.name}")
    common = {
        "friction": "1.2 0.01 0.0001",
        "contype": "1",
        "conaffinity": "1",
        "rgba": "0.30 0.33 0.36 1",
    }
    if terrain.name == "slope":
        # The upper surface rises continuously from the ground by about
        # 0.17 m over 4 m.  Most of the box stays below the plane so the
        # leading edge cannot become an accidental step.
        attrs = {
            **common,
            "name": "terrain_slope",
            "type": "box",
            "pos": "3.0 0 0.0073",
            "size": "2.0 2.0 0.08",
            "euler": "0 -0.043633 0",
        }
        ET.SubElement(worldbody, "geom", attrs)
    elif terrain.name == "bumps":
        for index, x in enumerate((1.0, 1.8, 2.6, 3.4)):
            ET.SubElement(worldbody, "geom", {
                **common,
                "name": f"terrain_bump_{index}",
                "type": "ellipsoid",
                "pos": f"{x:.3f} 0 0.02",
                "size": "0.16 1.5 0.02",
            })
    elif terrain.name == "step":
        # A short entry/exit ramp makes the platform a terrain step rather
        # than an accidental vertical wall against the wheel cylinder.
        ET.SubElement(worldbody, "geom", {
            **common,
            "name": "terrain_step_entry",
            "type": "box",
            "pos": "1.95 0 0.0",
            "size": "0.25 1.8 0.03",
            "euler": "0 -0.119429 0",
        })
        ET.SubElement(worldbody, "geom", {
            **common,
            "name": "terrain_step_platform",
            "type": "box",
            "pos": "2.60 0 0.03",
            "size": "0.40 1.8 0.03",
        })
        ET.SubElement(worldbody, "geom", {
            **common,
            "name": "terrain_step_exit",
            "type": "box",
            "pos": "3.25 0 0.0004",
            "size": "0.25 1.8 0.03",
            "euler": "0 0.119429 0",
        })


def _append_obstacles(worldbody: ET.Element, obstacles: list[ObstacleSpec] | None) -> None:
    if not obstacles:
        return
    for obstacle in obstacles:
        body_attrs = {
            "name": obstacle.name,
            "pos": _format(obstacle.position),
        }
        if any(abs(value) > 1.0e-9 for value in obstacle.euler):
            body_attrs["euler"] = _format(obstacle.euler)
        body = ET.SubElement(worldbody, "body", body_attrs)
        geom_attrs = {
            "name": f"{obstacle.name}_geom",
            "type": obstacle.kind,
            "rgba": _format(obstacle.rgba),
            "contype": str(int(obstacle.contype)),
            "conaffinity": str(int(obstacle.conaffinity)),
            "group": str(int(obstacle.group)),
            "size": _format(obstacle.size),
        }
        if obstacle.density is not None:
            geom_attrs["density"] = f"{float(obstacle.density):.4f}"
        ET.SubElement(body, "geom", geom_attrs)


def segment_intersects_aabb(
    start_xy: np.ndarray | tuple[float, float],
    end_xy: np.ndarray | tuple[float, float],
    minimum_xy: np.ndarray | tuple[float, float],
    maximum_xy: np.ndarray | tuple[float, float],
) -> bool:
    """Return whether a 2D segment intersects an axis-aligned box."""
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    minimum = np.asarray(minimum_xy, dtype=np.float64)
    maximum = np.asarray(maximum_xy, dtype=np.float64)
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(2):
        if abs(float(direction[axis])) < 1.0e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        inverse = 1.0 / float(direction[axis])
        t0 = (minimum[axis] - start[axis]) * inverse
        t1 = (maximum[axis] - start[axis]) * inverse
        lower = min(t0, t1)
        upper = max(t0, t1)
        t_min = max(t_min, lower)
        t_max = min(t_max, upper)
        if t_min > t_max:
            return False
    return True


def obstacle_blocks_segment(
    start_xy: np.ndarray | tuple[float, float],
    end_xy: np.ndarray | tuple[float, float],
    obstacle: ObstacleSpec,
    padding: float = 0.0,
) -> bool:
    """Return whether a 2D segment intersects an obstacle's footprint."""
    center = np.asarray(obstacle.position[:2], dtype=np.float64)
    half = np.asarray(obstacle.size[:2], dtype=np.float64) + float(padding)
    return segment_intersects_aabb(start_xy, end_xy, center - half, center + half)


def build_scene(
    path: Path,
    objects: list[ObjectSpec],
    obstacles: list[ObstacleSpec] | None = None,
    task_object_collisions: bool = True,
    terrain: TerrainSpec | None = None,
    light: SceneLightSpec | None = None,
) -> Path:
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the MuJoCo asset first: {ASSET}")
    root = ET.parse(ASSET).getroot()
    compiler = root.find("./compiler")
    if compiler is None:
        raise RuntimeError("M20 asset is missing a compiler element")
    # Generated MJCF files may contain an absolute path from the machine that
    # compiled them.  Rebind it at scene materialization time so a copied or
    # freshly cloned workspace remains runnable.
    compiler.set("meshdir", str(MESH_ROOT))
    worldbody = root.find("worldbody")
    base = worldbody.find("./body[@name='base_link']") if worldbody is not None else None
    if worldbody is None or base is None:
        raise RuntimeError("M20 asset is missing worldbody/base_link")
    # Keep robot collision geoms out of the policy LiDAR while preserving
    # their physical contacts. Scene objects remain in group 0.
    for geom in base.iter("geom"):
        geom.set("group", "1")
    # The camera frame is part of the observation contract, not a privileged sensor.
    base.append(ET.Element("camera", {
        "name": "front_rgb", "pos": "0.42 0 0.15",
        "xyaxes": "0 -1 0 0.174 0 0.985", "fovy": "65",
    }))
    base.append(ET.Element("camera", {
        "name": "rear_rgb", "pos": "-0.42 0 0.15",
        "xyaxes": "0 1 0 -0.174 0 0.985", "fovy": "65",
    }))
    light = light or SceneLightSpec()
    worldbody.insert(0, ET.Element("light", {
        "name": "overhead",
        "pos": _format(light.pos),
        "dir": _format(light.direction),
        "directional": "true" if light.directional else "false",
        "diffuse": _format(light.diffuse),
        "ambient": _format(light.ambient),
        "specular": _format(light.specular),
    }))
    _append_terrain(worldbody, terrain)
    _append_obstacles(worldbody, obstacles)
    for obj in objects:
        body = ET.SubElement(worldbody, "body", {
            "name": obj.name,
            "pos": f"{obj.position[0]:.6f} {obj.position[1]:.6f} 0.18",
        })
        attrs = {
            "name": f"{obj.name}_geom", "type": obj.kind,
            "rgba": " ".join(f"{value:.4f}" for value in obj.rgba),
            # Objects always participate in LiDAR ray casts. The language
            # counterfactual gate uses non-contact props so a nearer,
            # unselected object cannot mechanically block a farther target.
            "contype": "1" if task_object_collisions else "0",
            "conaffinity": "1" if task_object_collisions else "0",
        }
        if not task_object_collisions:
            # Counterfactual language swaps must change only the visible
            # scene, not the robot dynamics.  A non-contact prop with its
            # default density still contributes substantial welded-body mass
            # to MuJoCo and can make the wheel bridge stall.  Keep the geom
            # visible to RGB/LiDAR while making it physically massless.
            attrs["density"] = "0"
        if obj.kind == "box":
            attrs["size"] = " ".join(f"{value:.4f}" for value in obj.size)
        else:
            attrs["size"] = " ".join(f"{value:.4f}" for value in obj.size)
        ET.SubElement(body, "geom", attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def planar_lidar(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # Mount the planar scan near wheel height so the low target objects enter
    # the same geometric sensor stream as obstacles.
    origin = data.xpos[base_id].copy() + np.array((0.0, 0.0, -0.20))
    rotation = data.xmat[base_id].reshape(3, 3)
    geomid = np.empty(1, dtype=np.int32)
    geom_groups = np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8)
    values = np.empty(LIDAR_RAYS, dtype=np.float32)
    for index, heading in enumerate(np.linspace(-math.pi, math.pi, LIDAR_RAYS, endpoint=False)):
        direction = rotation @ np.array((math.cos(heading), math.sin(heading), 0.0))
        distance = mujoco.mj_ray(model, data, origin, direction, geom_groups, 1, base_id, geomid)
        values[index] = MAX_LIDAR_RANGE if distance < 0.0 else min(float(distance), MAX_LIDAR_RANGE)
    return values


def proprioception(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    value = np.concatenate((data.qpos.copy(), data.qvel.copy())).astype(np.float32)
    # Absolute world position is not useful to the policy and would leak the
    # simulator layout. Orientation, velocity and joint state remain visible.
    value[:3] = 0.0
    return value


def open_video(path: Path, width: int, height: int, fps: int = 50):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install imageio-ffmpeg for H.264 episode videos") from exc
    return subprocess.Popen(
        [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )

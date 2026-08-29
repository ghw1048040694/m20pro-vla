#!/usr/bin/env python3
"""Collect a compact randomized MuJoCo M20 high-level VLA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.sim.mujoco import (
    ASSET,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECTS,
    TERRAIN_PROFILES,
    ObstacleSpec,
    ObjectSpec,
    SceneLightSpec,
    TerrainSpec,
    build_scene,
    obstacle_blocks_segment,
    planar_lidar,
    proprioception,
    open_video,
)
from m20pro_vla.low_level import M20LowLevelController
from m20pro_vla.planning import SearchMPCConfig, SearchMPCPlanner, obstacle_clearance_xy


SEARCH_LAYOUT_TEMPLATES = (
    np.array(((3.50, 0.24), (3.40, 0.34), (3.90, -0.36)), dtype=np.float64),
    np.array(((3.20, 0.32), (3.50, 0.28), (3.80, -0.40)), dtype=np.float64),
    np.array(((3.30, 0.34), (3.60, 0.24), (3.95, -0.38)), dtype=np.float64),
    np.array(((3.50, 0.46), (3.70, -0.38), (3.90, -0.42)), dtype=np.float64),
)
SEARCH_SLOT_JITTER_XY = np.array((0.02, 0.015), dtype=np.float64)
SEARCH_TERRAIN_LAYOUT_SCALE = {
    "flat": 1.00,
    "slope": 0.74,
    "bumps": 0.70,
    "step": 0.64,
}
SEARCH_TERRAIN_LAYOUT_TEMPLATE_MAX_ID = {
    "flat": len(SEARCH_LAYOUT_TEMPLATES) - 1,
    "slope": 2,
    "bumps": 2,
    "step": 1,
}
SEARCH_OBSTACLE_TEMPLATES = (
    ObstacleSpec(
        name="search_wall",
        kind="box",
        position=(2.022822837243513, 0.0, 0.5205213822947318),
        size=(0.07892867288141192, 0.16899908178407988, 0.5205213822947318),
        rgba=(0.28, 0.30, 0.34, 1.0),
    ),
    ObstacleSpec(
        name="search_wall",
        kind="box",
        position=(2.022822837243513, 0.0, 0.5205213822947318),
        size=(0.07892867288141192, 0.16899908178407988, 0.5205213822947318),
        rgba=(0.30, 0.32, 0.36, 1.0),
    ),
    ObstacleSpec(
        name="search_wall",
        kind="box",
        position=(2.022822837243513, 0.0, 0.5205213822947318),
        size=(0.07892867288141192, 0.16899908178407988, 0.5205213822947318),
        rgba=(0.25, 0.27, 0.31, 1.0),
    ),
    ObstacleSpec(
        name="search_wall_wide",
        kind="box",
        position=(2.022822837243513, 0.0, 0.5205213822947318),
        size=(0.07892867288141192, 0.24000000000000002, 0.5205213822947318),
        rgba=(0.27, 0.29, 0.33, 1.0),
    ),
)
SEARCH_CLUTTER_TEMPLATES = (
    ObstacleSpec(
        name="search_clutter",
        kind="box",
        position=(2.66, 0.0, 0.30),
        size=(0.07, 0.18, 0.30),
        rgba=(0.38, 0.29, 0.22, 1.0),
    ),
    ObstacleSpec(
        name="search_clutter",
        kind="box",
        position=(2.88, 0.0, 0.34),
        size=(0.08, 0.20, 0.34),
        rgba=(0.30, 0.34, 0.30, 1.0),
    ),
    ObstacleSpec(
        name="search_clutter",
        kind="box",
        position=(3.08, 0.0, 0.28),
        size=(0.06, 0.16, 0.28),
        rgba=(0.26, 0.28, 0.33, 1.0),
    ),
)
SEARCH_OUTER_CLUTTER_TEMPLATES = (
    ObstacleSpec(
        name="search_outer_clutter",
        kind="box",
        position=(2.48, 1.32, 0.26),
        size=(0.08, 0.13, 0.26),
        rgba=(0.42, 0.36, 0.28, 1.0),
    ),
    ObstacleSpec(
        name="search_outer_clutter",
        kind="box",
        position=(2.82, -1.32, 0.31),
        size=(0.09, 0.14, 0.31),
        rgba=(0.24, 0.33, 0.40, 1.0),
    ),
    ObstacleSpec(
        name="search_outer_clutter",
        kind="box",
        position=(3.18, 1.42, 0.29),
        size=(0.07, 0.15, 0.29),
        rgba=(0.36, 0.30, 0.38, 1.0),
    ),
    ObstacleSpec(
        name="search_outer_clutter",
        kind="box",
        position=(3.36, -1.42, 0.27),
        size=(0.08, 0.12, 0.27),
        rgba=(0.31, 0.38, 0.30, 1.0),
    ),
)

START_POSITION_TEMPLATES = (
    np.array((0.00, 0.00), dtype=np.float64),
    np.array((0.04, 0.00), dtype=np.float64),
    np.array((0.06, 0.00), dtype=np.float64),
    np.array((-0.04, 0.00), dtype=np.float64),
)
SEARCH_START_POSITION_TEMPLATES = (
    np.array((-0.04, 0.00), dtype=np.float64),
    np.array((-0.02, 0.00), dtype=np.float64),
    np.array((0.00, 0.00), dtype=np.float64),
    np.array((0.02, 0.00), dtype=np.float64),
    np.array((0.04, 0.00), dtype=np.float64),
    np.array((0.06, 0.00), dtype=np.float64),
    np.array((0.08, 0.00), dtype=np.float64),
    np.array((0.10, 0.00), dtype=np.float64),
    np.array((0.12, 0.00), dtype=np.float64),
    np.array((-0.02, 0.02), dtype=np.float64),
    np.array((0.00, 0.02), dtype=np.float64),
    np.array((0.02, 0.02), dtype=np.float64),
)
SEARCH_START_YAW_OFFSETS = (
    -0.72,
    -0.45,
    -0.18,
    0.18,
    0.45,
    0.72,
)
OBJECT_COLOR_MULTIPLIER_RANGE = (0.92, 1.08)
OBJECT_COLOR_OFFSET_RANGE = (-0.025, 0.025)
OBSTACLE_COLOR_MULTIPLIER_RANGE = (0.88, 1.12)
OBSTACLE_COLOR_OFFSET_RANGE = (-0.035, 0.035)
LIGHT_POS_XY_RANGE = (-0.45, 0.45)
LIGHT_POS_Z_RANGE = (4.5, 6.0)
LIGHT_DIFFUSE_RANGE = (0.72, 0.98)
LIGHT_AMBIENT_RANGE = (0.02, 0.08)
LIGHT_SPECULAR_RANGE = (0.0, 0.04)

OBJECT_TEXT = {
    "red cube": {"en": "red cube", "zh": "红色方块"},
    "green cylinder": {"en": "green cylinder", "zh": "绿色圆柱"},
    "yellow box": {"en": "yellow box", "zh": "黄色盒子"},
}

TASK_TEMPLATES = (
    ("en_00", "en", "go to the {object_en}"),
    ("en_01", "en", "walk to the {object_en}"),
    ("en_02", "en", "find the {object_en}"),
    ("en_03", "en", "navigate to the {object_en}"),
    ("en_04", "en", "move near the {object_en}"),
    ("en_05", "en", "approach the {object_en}"),
    ("en_06", "en", "head toward the {object_en}"),
    ("en_07", "en", "locate the {object_en} and stop"),
    ("en_08", "en", "drive over to the {object_en}"),
    ("en_09", "en", "go where the {object_en} is"),
    ("en_10", "en", "stop beside the {object_en}"),
    ("en_11", "en", "search for the {object_en}"),
    ("zh_00", "zh", "去{object_zh}那里"),
    ("zh_01", "zh", "走到{object_zh}旁边"),
    ("zh_02", "zh", "找到{object_zh}"),
    ("zh_03", "zh", "导航到{object_zh}"),
    ("zh_04", "zh", "靠近{object_zh}"),
    ("zh_05", "zh", "朝{object_zh}前进"),
    ("zh_06", "zh", "去有{object_zh}的位置"),
    ("zh_07", "zh", "在{object_zh}旁边停下"),
    ("zh_08", "zh", "搜索{object_zh}"),
    ("zh_09", "zh", "请前往{object_zh}"),
    ("zh_10", "zh", "找一下{object_zh}在哪里"),
    ("zh_11", "zh", "移动到{object_zh}附近"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("randomized", "counterfactual", "search"), default="randomized")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--layouts", type=int, default=12, help="Counterfactual layouts; each yields one episode per instruction.")
    parser.add_argument("--layout-offset", type=int, default=0, help="First counterfactual layout ID; used only for disjoint collection shards.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--search-policy",
        choices=("waypoint", "active_mpc", "auto"),
        default="auto",
        help="Search-only control policy. waypoint preserves the stable route planner; active_mpc uses MPC scoring when obstacles are present; auto uses waypoint on easy flat layouts and MPC on harder ones.",
    )
    parser.add_argument(
        "--search-start-mode",
        choices=("balanced", "anchor", "varied"),
        default="balanced",
        help=(
            "Search-only start-pose curriculum. anchor preserves the stable canonical "
            "hidden-search route; varied uses randomized start xy/yaw; balanced emits both."
        ),
    )
    parser.add_argument(
        "--search-stop-after-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a search episode early after it has held curriculum success for a short tail.",
    )
    parser.add_argument(
        "--search-success-hold-steps",
        type=int,
        default=20,
        help="Extra steps to keep after a search episode first reaches curriculum success.",
    )
    parser.add_argument(
        "--search-curriculum-success-radius",
        type=float,
        default=0.45,
        help=(
            "Search-only radius that triggers expert stop labels and early-stop hold. "
            "Use a safe standoff radius so the expert stops before the robot collides with the target."
        ),
    )
    parser.add_argument(
        "--search-layout-scale",
        type=float,
        default=1.0,
        help=(
            "Extra multiplicative scale applied to search target slot positions. "
            "Values below 1.0 make the hidden-search curriculum closer while preserving occlusion."
        ),
    )
    parser.add_argument(
        "--search-terrain-profile",
        choices=("cycle", *TERRAIN_PROFILES),
        default="cycle",
        help="Search-only terrain curriculum. cycle preserves the layout-id mapping; a terrain name forces all collected search layouts to that terrain.",
    )
    parser.add_argument(
        "--search-layout-template-id",
        type=int,
        default=None,
        help="Force one search layout template for targeted data collection; default samples templates from the layout RNG.",
    )
    parser.add_argument(
        "--counterfactual-distance-scale",
        type=float,
        default=1.0,
        help=(
            "Extra multiplicative scale applied to counterfactual target distances. "
            "Use values below 1.0 to create an easier strict approach curriculum."
        ),
    )
    parser.add_argument(
        "--language-mode",
        choices=("varied", "fixed"),
        default="varied",
        help="Task-language distribution for new episodes. fixed preserves the historical 'go to the X' prompt.",
    )
    parser.add_argument("--policy-width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--policy-height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("M20PRO_VLA_DATA_ROOT", ".runtime")) / "datasets/m20_mujoco_vla_v1")
    parser.add_argument("--video-dir", type=Path, default=Path(os.environ.get("M20PRO_VLA_DATA_ROOT", ".runtime")) / "videos/m20_mujoco_vla_v1")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an H.264 episode video. Disable for faster distribution smoke tests.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip RGB/LiDAR/proprio/action arrays and write metadata only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: int
    layout_id: int | None
    terrain_profile: str
    search_template_id: int | None
    search_obstacle_template_id: int | None
    search_secondary_obstacle_template_id: int | None
    search_outer_obstacle_template_id: int | None
    yaw: float
    start_xy: np.ndarray
    target_xy: np.ndarray
    target_label: str
    objects: list[ObjectSpec]
    obstacles: list[ObstacleSpec]
    scene_light: SceneLightSpec | None = None
    search_start_mode: str = ""
    search_start_variant: str = ""
    search_start_xy_template_id: int = -1
    search_start_yaw_offset: float = 0.0
    search_required: bool = False
    initial_target_visible: bool = True


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _target_visible_from(
    start_xy: np.ndarray | tuple[float, float],
    target_xy: np.ndarray | tuple[float, float],
    obstacles: list[ObstacleSpec],
    padding: float = 0.04,
) -> bool:
    return not any(
        obstacle_blocks_segment(start_xy, target_xy, obstacle, padding=padding)
        for obstacle in obstacles
    )


def _minimum_obstacle_clearance(
    point_xy: np.ndarray | tuple[float, float],
    obstacles: list[ObstacleSpec],
) -> float | None:
    if not obstacles:
        return None
    return min(obstacle_clearance_xy(point_xy, obstacle) for obstacle in obstacles)


def _appearance_rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _jitter_rgba(
    rng: np.random.Generator,
    rgba: tuple[float, float, float, float],
    *,
    multiplier_range: tuple[float, float],
    offset_range: tuple[float, float],
) -> tuple[float, float, float, float]:
    rgb = np.asarray(rgba[:3], dtype=np.float64)
    multiplier = rng.uniform(multiplier_range[0], multiplier_range[1], size=3)
    offset = rng.uniform(offset_range[0], offset_range[1], size=3)
    jittered = np.clip(rgb * multiplier + offset, 0.0, 1.0)
    return (
        float(jittered[0]),
        float(jittered[1]),
        float(jittered[2]),
        float(rgba[3]),
    )


def _sample_scene_light(rng: np.random.Generator) -> SceneLightSpec:
    return SceneLightSpec(
        pos=(
            float(rng.uniform(LIGHT_POS_XY_RANGE[0], LIGHT_POS_XY_RANGE[1])),
            float(rng.uniform(LIGHT_POS_XY_RANGE[0], LIGHT_POS_XY_RANGE[1])),
            float(rng.uniform(LIGHT_POS_Z_RANGE[0], LIGHT_POS_Z_RANGE[1])),
        ),
        direction=(0.0, 0.0, -1.0),
        diffuse=tuple(float(value) for value in rng.uniform(LIGHT_DIFFUSE_RANGE[0], LIGHT_DIFFUSE_RANGE[1], size=3)),
        ambient=tuple(float(value) for value in rng.uniform(LIGHT_AMBIENT_RANGE[0], LIGHT_AMBIENT_RANGE[1], size=3)),
        specular=tuple(float(value) for value in rng.uniform(LIGHT_SPECULAR_RANGE[0], LIGHT_SPECULAR_RANGE[1], size=3)),
        directional=True,
    )


def _initial_xy_for(*, seed: int, scope: str, scope_id: int, target_label: str | None = None) -> np.ndarray:
    del seed, scope, target_label
    template_id = int(scope_id) % len(START_POSITION_TEMPLATES)
    return START_POSITION_TEMPLATES[template_id].copy()


def _search_terrain_profile_for_layout(layout_id: int) -> str:
    return TERRAIN_PROFILES[int(layout_id) % len(TERRAIN_PROFILES)]


def _search_initial_xy_for(
    *,
    seed: int,
    layout_id: int,
    target_label: str,
) -> tuple[np.ndarray, int]:
    digest = hashlib.sha256(f"{seed}:{layout_id}:{target_label}:search_xy".encode("utf-8")).digest()
    template_id = int.from_bytes(digest[:4], "little") % len(SEARCH_START_POSITION_TEMPLATES)
    return SEARCH_START_POSITION_TEMPLATES[template_id].copy(), template_id


def _search_initial_yaw_for(
    *,
    seed: int,
    layout_id: int,
    target_label: str,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[float, float]:
    target_heading = math.atan2(float(target_xy[1] - start_xy[1]), float(target_xy[0] - start_xy[0]))
    digest = hashlib.sha256(f"{seed}:{layout_id}:{target_label}:search_yaw".encode("utf-8")).digest()
    offset = SEARCH_START_YAW_OFFSETS[int.from_bytes(digest[:4], "little") % len(SEARCH_START_YAW_OFFSETS)]
    return _wrap_angle(target_heading + offset), float(offset)


def _search_initial_pose_variants_for(
    *,
    mode: str,
    seed: int,
    layout_id: int,
    target_label: str,
    target_xy: np.ndarray,
) -> list[tuple[str, np.ndarray, int, float, float]]:
    if mode not in {"balanced", "anchor", "varied"}:
        raise ValueError(f"unsupported search start mode: {mode}")
    variants: list[tuple[str, np.ndarray, int, float, float]] = []
    if mode in {"balanced", "anchor"}:
        variants.append((
            "anchor",
            np.array((0.0, 0.0), dtype=np.float64),
            -1,
            0.0,
            0.0,
        ))
    if mode in {"balanced", "varied"}:
        start_xy, start_xy_template_id = _search_initial_xy_for(
            seed=seed,
            layout_id=layout_id,
            target_label=target_label,
        )
        start_yaw, start_yaw_offset = _search_initial_yaw_for(
            seed=seed,
            layout_id=layout_id,
            target_label=target_label,
            start_xy=start_xy,
            target_xy=target_xy,
        )
        variants.append((
            "varied",
            start_xy,
            start_xy_template_id,
            start_yaw,
            start_yaw_offset,
        ))
    return variants


def _task_language_for(
    *,
    target_label: str,
    episode_id: int,
    seed: int,
    language_mode: str,
) -> dict[str, str]:
    if target_label not in OBJECT_TEXT:
        raise KeyError(f"Missing object text aliases for target label: {target_label}")
    if language_mode == "fixed":
        template_id, language, template = TASK_TEMPLATES[0]
    else:
        digest = hashlib.sha256(f"{seed}:{episode_id}:{target_label}".encode("utf-8")).digest()
        template_id, language, template = TASK_TEMPLATES[int.from_bytes(digest[:4], "little") % len(TASK_TEMPLATES)]
    object_text = OBJECT_TEXT[target_label]
    return {
        "task_text": template.format(object_en=object_text["en"], object_zh=object_text["zh"]),
        "task_template_id": template_id,
        "task_language": language,
        "task_object_text_en": object_text["en"],
        "task_object_text_zh": object_text["zh"],
    }


def make_episode(
    rng: np.random.Generator,
    episode_id: int,
    *,
    seed: int = 0,
    appearance_rng: np.random.Generator | None = None,
):
    target_index = episode_id % len(OBJECTS)
    _, target_label, _, _ = OBJECTS[target_index]
    yaw = float(rng.uniform(-0.45, 0.45))
    # Keep the first dataset within the verified forward-only bridge range;
    # later datasets will add explicit turn episodes after yaw calibration.
    distance = float(rng.uniform(1.25, 1.55))
    target_xy = np.array((distance * math.cos(yaw), distance * math.sin(yaw)), dtype=np.float64)
    appearance_rng = appearance_rng or _appearance_rng(seed, "episode", episode_id, target_label)
    objects: list[ObjectSpec] = []
    for index, (name, label, kind, rgba) in enumerate(OBJECTS):
        if index == target_index:
            position = target_xy
        else:
            angle = yaw + (0.72 if index < target_index else -0.72)
            radius = float(rng.uniform(1.35, 2.35))
            position = np.array((radius * math.cos(angle), radius * math.sin(angle)), dtype=np.float64)
        size = (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)
        objects.append(
            ObjectSpec(
                name,
                label,
                kind,
                (float(position[0]), float(position[1])),
                _jitter_rgba(
                    appearance_rng,
                    rgba,
                    multiplier_range=OBJECT_COLOR_MULTIPLIER_RANGE,
                    offset_range=OBJECT_COLOR_OFFSET_RANGE,
                ),
                size,
            )
        )
    return yaw, target_xy, target_label, objects


def make_counterfactual_layout(
    rng: np.random.Generator,
    *,
    seed: int = 0,
    distance_scale: float = 1.0,
    appearance_rng: np.random.Generator | None = None,
) -> tuple[list[ObjectSpec], dict[str, np.ndarray]]:
    """Make one visible scene shared unchanged by all language instructions.

    Each object is randomly assigned to a near/middle/far slot. This prevents
    a policy from passing by treating a colour word as a fixed drive duration.
    """
    # Keep target regions disjoint under the counterfactual gate's 0.22 m
    # success radius. Object identities are randomly assigned to slots.
    distances = (np.array((0.95, 1.40, 1.85), dtype=np.float64) + rng.uniform(-0.04, 0.04, size=3)) * float(distance_scale)
    # The current fixed-wheel bridge is nonholonomic. Keep counterfactual
    # targets in forward slots so the expert labels are physically executable;
    # broad lateral displacement belongs to a separate turning dataset.
    laterals = rng.uniform(-0.015, 0.015, size=3)
    slot_for_object = rng.permutation(len(OBJECTS))
    appearance_rng = appearance_rng or _appearance_rng(
        seed,
        "counterfactual",
        tuple(np.round(distances, 4)),
        tuple(int(slot) for slot in slot_for_object.tolist()),
    )
    objects: list[ObjectSpec] = []
    targets: dict[str, np.ndarray] = {}
    for object_index, (name, label, kind, rgba) in enumerate(OBJECTS):
        slot = int(slot_for_object[object_index])
        position = np.array((distances[slot], laterals[slot]), dtype=np.float64)
        size = (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)
        objects.append(
            ObjectSpec(
                name,
                label,
                kind,
                (float(position[0]), float(position[1])),
                _jitter_rgba(
                    appearance_rng,
                    rgba,
                    multiplier_range=OBJECT_COLOR_MULTIPLIER_RANGE,
                    offset_range=OBJECT_COLOR_OFFSET_RANGE,
                ),
                size,
            )
        )
        targets[label] = position
    return objects, targets


def make_search_layout(
    rng: np.random.Generator,
    *,
    terrain_profile: str = "flat",
    seed: int = 0,
    layout_scale: float = 1.0,
    appearance_rng: np.random.Generator | None = None,
    forced_template_id: int | None = None,
) -> tuple[list[ObjectSpec], dict[str, np.ndarray], list[ObstacleSpec], int, int, int, int]:
    # Search/obstacle curricula need enough run-up for the current wheel-leg
    # bridge to make a stable arc.  Use a small set of verified target slots
    # that are all initially occluded and all reachable with the current
    # stateful route planner.  The label-to-slot assignment is still permuted
    # so the dataset does not collapse to a single semantic mapping.
    max_template_id = int(SEARCH_TERRAIN_LAYOUT_TEMPLATE_MAX_ID.get(terrain_profile, len(SEARCH_LAYOUT_TEMPLATES) - 1))
    if forced_template_id is None:
        template_id = int(rng.integers(max_template_id + 1))
    else:
        template_id = int(forced_template_id)
        if template_id < 0 or template_id > max_template_id:
            raise ValueError(
                f"forced search layout template id {template_id} is invalid for terrain {terrain_profile!r}; "
                f"valid range is [0, {max_template_id}]"
            )
    slot_positions = SEARCH_LAYOUT_TEMPLATES[template_id] + rng.uniform(
        -SEARCH_SLOT_JITTER_XY,
        SEARCH_SLOT_JITTER_XY,
        size=SEARCH_LAYOUT_TEMPLATES[template_id].shape,
    )
    slot_positions = slot_positions.copy()
    terrain_scale = float(SEARCH_TERRAIN_LAYOUT_SCALE.get(terrain_profile, 1.0))
    slot_positions *= terrain_scale * float(layout_scale)
    slot_for_object = rng.permutation(len(OBJECTS))
    appearance_rng = appearance_rng or _appearance_rng(
        seed,
        "search",
        terrain_profile,
        template_id,
        tuple(int(slot) for slot in slot_for_object.tolist()),
    )
    objects: list[ObjectSpec] = []
    targets: dict[str, np.ndarray] = {}
    for object_index, (name, label, kind, rgba) in enumerate(OBJECTS):
        slot = int(slot_for_object[object_index])
        position = slot_positions[slot]
        size = (0.16, 0.16, 0.16) if kind == "box" else (0.14, 0.20)
        objects.append(
            ObjectSpec(
                name,
                label,
                kind,
                (float(position[0]), float(position[1])),
                _jitter_rgba(
                    appearance_rng,
                    rgba,
                    multiplier_range=OBJECT_COLOR_MULTIPLIER_RANGE,
                    offset_range=OBJECT_COLOR_OFFSET_RANGE,
                ),
                size,
            )
        )
        targets[label] = position
    if template_id == len(SEARCH_LAYOUT_TEMPLATES) - 1:
        obstacle_template_id = len(SEARCH_OBSTACLE_TEMPLATES) - 1
    else:
        obstacle_template_id = int(rng.integers(len(SEARCH_OBSTACLE_TEMPLATES) - 1))
    clutter_template_id = int(rng.integers(len(SEARCH_CLUTTER_TEMPLATES)))
    outer_clutter_template_id = int(rng.integers(len(SEARCH_OUTER_CLUTTER_TEMPLATES)))
    obstacle = SEARCH_OBSTACLE_TEMPLATES[obstacle_template_id]
    clutter = SEARCH_CLUTTER_TEMPLATES[clutter_template_id]
    outer_clutter = SEARCH_OUTER_CLUTTER_TEMPLATES[outer_clutter_template_id]
    scale_xy = float(layout_scale)
    obstacle = ObstacleSpec(
        name=obstacle.name,
        kind=obstacle.kind,
        position=(
            float(obstacle.position[0] * scale_xy),
            float(obstacle.position[1] * scale_xy),
            float(obstacle.position[2]),
        ),
        size=obstacle.size,
        rgba=_jitter_rgba(
            appearance_rng,
            obstacle.rgba,
            multiplier_range=OBSTACLE_COLOR_MULTIPLIER_RANGE,
            offset_range=OBSTACLE_COLOR_OFFSET_RANGE,
        ),
        euler=obstacle.euler,
        contype=obstacle.contype,
        conaffinity=obstacle.conaffinity,
        group=obstacle.group,
        density=obstacle.density,
    )
    clutter = ObstacleSpec(
        name=clutter.name,
        kind=clutter.kind,
        position=(
            float(clutter.position[0] * scale_xy),
            float(clutter.position[1] * scale_xy),
            float(clutter.position[2]),
        ),
        size=clutter.size,
        rgba=_jitter_rgba(
            appearance_rng,
            clutter.rgba,
            multiplier_range=OBSTACLE_COLOR_MULTIPLIER_RANGE,
            offset_range=OBSTACLE_COLOR_OFFSET_RANGE,
        ),
        euler=clutter.euler,
        contype=clutter.contype,
        conaffinity=clutter.conaffinity,
        group=clutter.group,
        density=clutter.density,
    )
    outer_clutter = ObstacleSpec(
        name=outer_clutter.name,
        kind=outer_clutter.kind,
        position=(
            float(outer_clutter.position[0] * scale_xy),
            float(outer_clutter.position[1] * scale_xy),
            float(outer_clutter.position[2]),
        ),
        size=outer_clutter.size,
        rgba=_jitter_rgba(
            appearance_rng,
            outer_clutter.rgba,
            multiplier_range=OBSTACLE_COLOR_MULTIPLIER_RANGE,
            offset_range=OBSTACLE_COLOR_OFFSET_RANGE,
        ),
        euler=outer_clutter.euler,
        contype=outer_clutter.contype,
        conaffinity=outer_clutter.conaffinity,
        group=outer_clutter.group,
        density=outer_clutter.density,
    )
    return (
        objects,
        targets,
        [obstacle, clutter, outer_clutter],
        template_id,
        obstacle_template_id,
        clutter_template_id,
        outer_clutter_template_id,
    )


def _expert_command(
    *,
    mode: str,
    base_xy: np.ndarray,
    base_yaw: float,
    goal_xy: np.ndarray,
    detour_goal: bool,
    distance: float,
    bearing: float,
) -> np.ndarray:
    del base_xy, base_yaw, goal_xy
    if mode == "search":
        if detour_goal and abs(bearing) > 0.35:
            speed = 0.0
            yaw_action = float(np.clip(0.55 * bearing, -0.15, 0.15))
        else:
            speed = 0.30 if detour_goal else 0.35
            if distance < 0.35 and not detour_goal:
                speed = 0.10
            yaw_gain = 0.45 if detour_goal else 0.35
            yaw_action = float(np.clip(yaw_gain * bearing, -0.15, 0.15))
    else:
        speed = 0.35 if mode == "counterfactual" else (0.35 if distance > 1.0 else 0.16)
        yaw_action = 0.0 if mode == "counterfactual" else float(np.clip(0.35 * bearing, -0.12, 0.12))
    return np.array((speed, 0.0, yaw_action, 0.0), dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.layouts <= 0 or args.layout_offset < 0 or args.steps <= args.warmup_steps or args.warmup_steps < 1:
        raise ValueError("invalid episode length")
    if args.policy_width <= 0 or args.policy_height <= 0:
        raise ValueError("policy RGB resolution must be positive")
    if args.search_terrain_profile != "cycle" and args.mode != "search":
        raise ValueError("--search-terrain-profile is only valid with --mode search")
    if args.search_success_hold_steps < 0:
        raise ValueError("search-success-hold-steps must be non-negative")
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the asset first: {ASSET}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    # Sharded collection must advance the random stream; otherwise each
    # worker silently regenerates the same layouts from the same seed.
    rng = np.random.default_rng(args.seed + 1009 * args.layout_offset)
    summaries = []
    if args.mode == "counterfactual":
        plans = []
        for layout_id in range(args.layout_offset, args.layout_offset + args.layouts):
            layout_seed = args.seed + 1009 * layout_id
            layout_appearance_rng = _appearance_rng(layout_seed, "counterfactual", layout_id)
            objects, targets = make_counterfactual_layout(
                rng,
                seed=layout_seed,
                distance_scale=float(args.counterfactual_distance_scale),
                appearance_rng=layout_appearance_rng,
            )
            start_xy = _initial_xy_for(seed=args.seed, scope="counterfactual", scope_id=layout_id)
            scene_light = _sample_scene_light(layout_appearance_rng)
            for target_index, (_, target_label, _, _) in enumerate(OBJECTS):
                plans.append(
                    EpisodePlan(
                        episode_id=layout_id * len(OBJECTS) + target_index,
                        layout_id=layout_id,
                        terrain_profile="flat",
                        search_template_id=None,
                        search_obstacle_template_id=None,
                        search_secondary_obstacle_template_id=None,
                        search_outer_obstacle_template_id=None,
                        yaw=0.0,
                        start_xy=start_xy,
                        target_xy=targets[target_label],
                        target_label=target_label,
                        objects=objects,
                        obstacles=[],
                        scene_light=scene_light,
                    )
                )
    elif args.mode == "search":
        plans = []
        for layout_id in range(args.layout_offset, args.layout_offset + args.layouts):
            layout_seed = args.seed + 1009 * layout_id
            terrain_profile = (
                _search_terrain_profile_for_layout(layout_id)
                if args.search_terrain_profile == "cycle"
                else str(args.search_terrain_profile)
            )
            layout_appearance_rng = _appearance_rng(layout_seed, "search", layout_id, terrain_profile)
            (
                objects,
                targets,
                obstacles,
                template_id,
                obstacle_template_id,
                secondary_obstacle_template_id,
                outer_obstacle_template_id,
            ) = make_search_layout(
                rng,
                terrain_profile=terrain_profile,
                seed=layout_seed,
                layout_scale=float(args.search_layout_scale),
                appearance_rng=layout_appearance_rng,
                forced_template_id=args.search_layout_template_id,
            )
            scene_light = _sample_scene_light(layout_appearance_rng)
            for target_index, (_, target_label, _, _) in enumerate(OBJECTS):
                target_xy = targets[target_label]
                variants = _search_initial_pose_variants_for(
                    mode=args.search_start_mode,
                    seed=args.seed,
                    layout_id=layout_id,
                    target_label=target_label,
                    target_xy=target_xy,
                )
                for variant_index, (variant_name, start_xy, start_xy_template_id, start_yaw, start_yaw_offset) in enumerate(variants):
                    episode_base_id = layout_id * len(OBJECTS) + target_index
                    episode_id = (
                        episode_base_id * len(variants) + variant_index
                        if args.search_start_mode == "balanced"
                        else episode_base_id
                    )
                    plans.append(
                        EpisodePlan(
                            episode_id=episode_id,
                            layout_id=layout_id,
                            terrain_profile=terrain_profile,
                            search_template_id=template_id,
                            search_obstacle_template_id=obstacle_template_id,
                            search_secondary_obstacle_template_id=secondary_obstacle_template_id,
                            search_outer_obstacle_template_id=outer_obstacle_template_id,
                            yaw=start_yaw,
                            start_xy=start_xy,
                            target_xy=target_xy,
                            target_label=target_label,
                            objects=objects,
                            obstacles=obstacles,
                            scene_light=scene_light,
                            search_start_mode=args.search_start_mode,
                            search_start_variant=variant_name,
                            search_start_xy_template_id=start_xy_template_id,
                            search_start_yaw_offset=start_yaw_offset,
                            search_required=True,
                            initial_target_visible=_target_visible_from(start_xy, target_xy, obstacles),
                        )
                    )
    else:
        plans = []
        for episode_id in range(args.episodes):
            episode_seed = args.seed + 1009 * episode_id
            episode_appearance_rng = _appearance_rng(episode_seed, "randomized", episode_id)
            yaw, target_xy, target_label, objects = make_episode(
                rng,
                episode_id,
                seed=episode_seed,
                appearance_rng=episode_appearance_rng,
            )
            start_xy = _initial_xy_for(seed=args.seed, scope="randomized", scope_id=episode_id, target_label=target_label)
            plans.append(
                    EpisodePlan(
                        episode_id=episode_id,
                        layout_id=None,
                        terrain_profile="flat",
                        search_template_id=None,
                        search_obstacle_template_id=None,
                        search_secondary_obstacle_template_id=None,
                        search_outer_obstacle_template_id=None,
                    yaw=yaw,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    target_label=target_label,
                    objects=objects,
                    obstacles=[],
                    scene_light=_sample_scene_light(episode_appearance_rng),
                )
            )
    for plan in plans:
        out = args.output_dir / f"episode_{plan.episode_id:04d}.npz"
        video_path = args.video_dir / f"episode_{plan.episode_id:04d}.mp4"
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"Episode exists; pass --overwrite: {out}")
        scene_path = args.output_dir / f"scene_{plan.episode_id:04d}.xml"
        task_object_collisions = args.mode == "randomized"
        canonical_success_radius = 0.22 if args.mode in {"counterfactual", "search"} else 0.70
        curriculum_success_radius = (
            float(args.search_curriculum_success_radius) if args.mode == "search" else canonical_success_radius
        )
        collect_arrays = not args.metadata_only
        terrain = None if plan.terrain_profile == "flat" else TerrainSpec(plan.terrain_profile)
        build_scene(
            scene_path,
            plan.objects,
            obstacles=plan.obstacles,
            task_object_collisions=task_object_collisions,
            terrain=terrain,
            light=plan.scene_light,
        )
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        controller = M20LowLevelController(model)
        controller.reset(data, plan.yaw, plan.start_xy)
        effective_search_policy = ""
        use_route_planner = True
        if args.mode == "search":
            if args.search_policy == "waypoint":
                use_route_planner = True
            elif args.search_policy == "active_mpc":
                use_route_planner = False
            else:
                use_route_planner = (
                    plan.search_start_variant == "anchor"
                    or (
                        plan.terrain_profile == "flat"
                        and abs(float(plan.search_start_yaw_offset)) <= 0.22
                    )
                )
            effective_search_policy = "waypoint" if use_route_planner else "active_mpc"
        search_planner = (
            SearchMPCPlanner(
                model,
                plan.target_xy,
                plan.obstacles,
                SearchMPCConfig(use_route_planner_when_obstacles=use_route_planner),
            )
            if args.mode == "search"
            else None
        )
        renderer = mujoco.Renderer(model, height=args.policy_height, width=args.policy_width) if collect_arrays else None
        front = np.zeros((args.steps, args.policy_height, args.policy_width, 3), dtype=np.uint8) if collect_arrays else None
        rear = np.zeros_like(front) if collect_arrays else None
        lidar = np.zeros((args.steps, 72), dtype=np.float32) if collect_arrays else None
        proprio = np.zeros((args.steps, 45), dtype=np.float32) if collect_arrays else None
        action = np.zeros((args.steps, 4), dtype=np.float32) if collect_arrays else None
        video = open_video(video_path, args.policy_width, args.policy_height) if (args.video and collect_arrays) else None
        reached_step = -1
        canonical_reached_step = -1
        min_distance = float("inf")
        min_height = float(data.qpos[2])
        success = False
        planner_info: dict[str, float | list[float]] = {}
        target_first_visible_step = -1
        target_visible_steps = 0
        min_obstacle_clearance = None
        collected_steps = 0
        terminated_early = False
        try:
            for step in range(args.steps + args.warmup_steps):
                target_visible_now = False
                clearance_now = None
                if step < args.warmup_steps:
                    # Warmup must match the replay contract: zero body
                    # command, standing controller only.
                    expert = np.zeros(4, dtype=np.float64)
                    distance = float("inf")
                else:
                    base_xy = data.qpos[:2].copy()
                    target_visible_now = _target_visible_from(base_xy, plan.target_xy, plan.obstacles)
                    clearance_now = _minimum_obstacle_clearance(base_xy, plan.obstacles)
                    base_yaw = math.atan2(
                        2.0 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]),
                        1.0 - 2.0 * (data.qpos[5] ** 2 + data.qpos[6] ** 2),
                    )
                    delta = plan.target_xy - base_xy
                    distance = float(np.linalg.norm(delta))
                    bearing = _wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - base_yaw)
                    if distance <= curriculum_success_radius and reached_step < 0:
                        reached_step = max(0, step - args.warmup_steps)
                    if distance <= canonical_success_radius and canonical_reached_step < 0:
                        canonical_reached_step = max(0, step - args.warmup_steps)
                    if distance <= curriculum_success_radius:
                        expert = np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                    elif args.mode == "search":
                        expert, planner_info = search_planner.recommend(data, controller)
                    else:
                        # The first stable bridge keeps the target initially in
                        # front. A small heading label is retained for the VLA
                        # contract but is clamped at execution for stability.
                        expert = _expert_command(
                            mode=args.mode,
                            base_xy=base_xy,
                            base_yaw=base_yaw,
                            goal_xy=plan.target_xy,
                            detour_goal=False,
                            distance=distance,
                            bearing=bearing,
                        )
                if step >= args.warmup_steps:
                    min_distance = min(min_distance, distance)
                    min_height = min(min_height, float(data.qpos[2]))
                    if target_visible_now:
                        target_visible_steps += 1
                        if target_first_visible_step < 0:
                            target_first_visible_step = step - args.warmup_steps
                    if clearance_now is not None:
                        min_obstacle_clearance = (
                            clearance_now
                            if min_obstacle_clearance is None
                            else min(min_obstacle_clearance, clearance_now)
                        )
                    if collect_arrays:
                        # Policy observation and action label describe the same
                        # physical state. This is essential for closed-loop BC.
                        index = collected_steps
                        assert renderer is not None and front is not None and rear is not None and lidar is not None and proprio is not None and action is not None
                        renderer.update_scene(data, camera="front_rgb")
                        front[index] = renderer.render().copy()
                        renderer.update_scene(data, camera="rear_rgb")
                        rear[index] = renderer.render().copy()
                        lidar[index] = planar_lidar(model, data)
                        proprio[index] = proprioception(model, data)
                        action[index] = expert.astype(np.float32)
                        if video is not None:
                            video.stdin.write(front[index].tobytes())
                    collected_steps += 1
                controller.step(data, expert)
                if (
                    args.mode == "search"
                    and args.search_stop_after_success
                    and reached_step >= 0
                    and (step - args.warmup_steps) >= reached_step + args.search_success_hold_steps
                ):
                    terminated_early = True
                    break
            proprio_finite = True if proprio is None else bool(np.isfinite(proprio).all())
            success = bool(reached_step >= 0 and min_height >= 0.45 and proprio_finite)
        finally:
            if video is not None:
                video.stdin.close()
                return_code = video.wait()
                if return_code != 0:
                    raise RuntimeError(f"ffmpeg failed for {video_path}: {return_code}")
        task_language = _task_language_for(
            target_label=plan.target_label,
            episode_id=plan.episode_id,
            seed=args.seed,
            language_mode=args.language_mode,
        )
        metadata = {
            "schema": "m20pro_mujoco_vla_episode_v1",
            "episode_id": plan.episode_id,
            "collection_mode": args.mode,
            "search_policy": effective_search_policy if args.mode == "search" else "",
            "search_policy_requested": args.search_policy if args.mode == "search" else "",
            "search_policy_effective": effective_search_policy if args.mode == "search" else "",
            "layout_id": plan.layout_id,
            "terrain_profile": plan.terrain_profile,
            "terrain_version": terrain.version if terrain is not None else "",
            "initial_yaw": plan.yaw,
            "initial_xy": [float(plan.start_xy[0]), float(plan.start_xy[1])],
            "scene_light": (
                {
                    "pos": list(plan.scene_light.pos),
                    "direction": list(plan.scene_light.direction),
                    "diffuse": list(plan.scene_light.diffuse),
                    "ambient": list(plan.scene_light.ambient),
                    "specular": list(plan.scene_light.specular),
                    "directional": bool(plan.scene_light.directional),
                }
                if plan.scene_light is not None
                else {}
            ),
            "search_start_xy_template_id": plan.search_start_xy_template_id,
            "search_start_mode": plan.search_start_mode if args.mode == "search" else "",
            "search_start_variant": plan.search_start_variant if args.mode == "search" else "",
            "search_start_yaw_offset_rad": float(plan.search_start_yaw_offset),
            "search_start_yaw_offset_deg": float(math.degrees(plan.search_start_yaw_offset)),
            "metadata_only": args.metadata_only,
            **task_language,
            "target_label": plan.target_label,
            "target_xy_privileged_label_only": plan.target_xy.tolist(),
            "objects": [
                {
                    "name": obj.name,
                    "label": obj.label,
                    "position": list(obj.position),
                    "rgba": list(obj.rgba),
                    "size": list(obj.size),
                }
                for obj in plan.objects
            ],
            "obstacles": [
                {
                    "name": obstacle.name,
                    "kind": obstacle.kind,
                    "position": list(obstacle.position),
                    "size": list(obstacle.size),
                    "rgba": list(obstacle.rgba),
                }
                for obstacle in plan.obstacles
            ],
            "steps": collected_steps if collected_steps > 0 else args.steps,
            "requested_steps": args.steps,
            "terminated_early": terminated_early,
            "search_stop_after_success": args.search_stop_after_success,
            "search_success_hold_steps": args.search_success_hold_steps,
            "warmup_steps": args.warmup_steps,
            "success": success,
            "target_reached_step": reached_step,
            "canonical_target_reached_step": canonical_reached_step,
            "min_target_distance": min_distance,
            "min_base_height": min_height,
            "task_object_collisions": task_object_collisions,
            "success_radius": curriculum_success_radius if args.mode == "search" else canonical_success_radius,
            "canonical_success_radius": canonical_success_radius,
            "curriculum_success_radius": curriculum_success_radius,
            "search_required": plan.search_required,
            "initial_target_visible": plan.initial_target_visible,
            "search_layout_template_id": plan.search_template_id,
            "search_obstacle_template_id": plan.search_obstacle_template_id,
            "search_secondary_obstacle_template_id": plan.search_secondary_obstacle_template_id,
            "search_outer_obstacle_template_id": plan.search_outer_obstacle_template_id,
            "search_layout_jitter_xy_max": SEARCH_SLOT_JITTER_XY.tolist() if plan.search_template_id is not None else [],
            "search_layout_template_positions": (
                [[float(value) for value in pair] for pair in SEARCH_LAYOUT_TEMPLATES[plan.search_template_id]]
                if plan.search_template_id is not None
                else []
            ),
            "search_obstacle_template": (
                {
                    "position": list(SEARCH_OBSTACLE_TEMPLATES[plan.search_obstacle_template_id].position),
                    "size": list(SEARCH_OBSTACLE_TEMPLATES[plan.search_obstacle_template_id].size),
                    "rgba": list(SEARCH_OBSTACLE_TEMPLATES[plan.search_obstacle_template_id].rgba),
                }
                if plan.search_obstacle_template_id is not None
                else {}
            ),
            "search_secondary_obstacle_template": (
                {
                    "position": list(SEARCH_CLUTTER_TEMPLATES[plan.search_secondary_obstacle_template_id].position),
                    "size": list(SEARCH_CLUTTER_TEMPLATES[plan.search_secondary_obstacle_template_id].size),
                    "rgba": list(SEARCH_CLUTTER_TEMPLATES[plan.search_secondary_obstacle_template_id].rgba),
                }
                if plan.search_secondary_obstacle_template_id is not None
                else {}
            ),
            "search_outer_obstacle_template": (
                {
                    "position": list(SEARCH_OUTER_CLUTTER_TEMPLATES[plan.search_outer_obstacle_template_id].position),
                    "size": list(SEARCH_OUTER_CLUTTER_TEMPLATES[plan.search_outer_obstacle_template_id].size),
                    "rgba": list(SEARCH_OUTER_CLUTTER_TEMPLATES[plan.search_outer_obstacle_template_id].rgba),
                }
                if plan.search_outer_obstacle_template_id is not None
                else {}
            ),
            "search_obstacle_count": len(plan.obstacles),
            "target_first_visible_step": target_first_visible_step,
            "target_visible_steps": target_visible_steps,
            "target_visibility_fraction": float(target_visible_steps / collected_steps) if collected_steps > 0 else 0.0,
            "target_discovered": target_first_visible_step >= 0,
            "target_discovered_before_reach": (
                target_first_visible_step >= 0
                and (reached_step < 0 or target_first_visible_step <= reached_step)
            ),
            "min_obstacle_clearance": min_obstacle_clearance,
            "planner": planner_info if args.mode == "search" else {},
            "video": str(video_path) if args.video else "",
            "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "language"],
            "policy_input_resolution": {"width": int(args.policy_width), "height": int(args.policy_height)},
            "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
            "canonical_success": bool(
                canonical_reached_step >= 0 and min_height >= 0.45
            ),
        }
        (args.output_dir / f"episode_{plan.episode_id:04d}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if collect_arrays:
            assert front is not None and rear is not None and lidar is not None and proprio is not None and action is not None
            np.savez_compressed(
                out,
                front_rgb=front[:collected_steps],
                rear_rgb=rear[:collected_steps],
                lidar=lidar[:collected_steps],
                proprio=proprio[:collected_steps],
                action=action[:collected_steps],
            )
        summaries.append(metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        scene_path.unlink(missing_ok=True)
    terrain_profiles = sorted({str(item.get("terrain_profile", "")) for item in summaries if item.get("terrain_profile")})
    summary = {
        "schema": "m20pro_mujoco_vla_dataset_v1",
        "episodes": summaries,
        "success_count": sum(item["success"] for item in summaries),
        "canonical_success_count": sum(bool(item.get("canonical_success", False)) for item in summaries),
        "target_discovered_count": sum(bool(item.get("target_discovered", False)) for item in summaries),
        "search_start_variants": sorted({
            str(item.get("search_start_variant", ""))
            for item in summaries
            if item.get("search_start_variant")
        }),
        "terrain_profiles": terrain_profiles,
        "policy_input_resolution": {"width": int(args.policy_width), "height": int(args.policy_height)},
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.output_dir), "episodes": len(summaries), "success_count": summary["success_count"]}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

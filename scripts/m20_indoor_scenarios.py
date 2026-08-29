#!/usr/bin/env python3

"""Build and validate deterministic indoor ObjectNav scenario manifests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


VISIBLE_SCHEMA = "m20pro_visible_objectnav_scenarios_v1"
HIDDEN_SCHEMA = "m20pro_hidden_objectnav_scenarios_v1"
HIDDEN_SEARCH_FAMILY_COUNT = 4
HIDDEN_SEARCH_FAMILY_NAMES = (
    "screen_only",
    "screen_with_upper_panel",
    "screen_with_lower_panel",
    "screen_combo",
)

OBJECT_CATALOG = (
    ("master_chef_can", "主厨罐头", "master chef can", "Props/YCB/Axis_Aligned/002_master_chef_can.usd", 4.0),
    ("cracker_box", "苏打饼干盒", "cracker box", "Props/YCB/Axis_Aligned/003_cracker_box.usd", 3.0),
    ("sugar_box", "糖盒", "sugar box", "Props/YCB/Axis_Aligned/004_sugar_box.usd", 3.2),
    ("tomato_soup_can", "番茄汤罐头", "tomato soup can", "Props/YCB/Axis_Aligned/005_tomato_soup_can.usd", 5.0),
    ("mustard_bottle", "芥末瓶", "mustard bottle", "Props/YCB/Axis_Aligned/006_mustard_bottle.usd", 3.0),
    ("tuna_fish_can", "金枪鱼罐头", "tuna fish can", "Props/YCB/Axis_Aligned/007_tuna_fish_can.usd", 7.0),
    ("pudding_box", "布丁盒", "pudding box", "Props/YCB/Axis_Aligned/008_pudding_box.usd", 4.5),
    ("gelatin_box", "果冻盒", "gelatin box", "Props/YCB/Axis_Aligned/009_gelatin_box.usd", 5.0),
    ("potted_meat_can", "午餐肉罐头", "potted meat can", "Props/YCB/Axis_Aligned/010_potted_meat_can.usd", 5.5),
    ("banana", "香蕉", "banana", "Props/YCB/Axis_Aligned/011_banana.usd", 3.0),
    ("pitcher", "水壶", "pitcher", "Props/YCB/Axis_Aligned/019_pitcher_base.usd", 2.5),
    ("bleach_cleanser", "清洁剂瓶", "bleach cleanser bottle", "Props/YCB/Axis_Aligned/021_bleach_cleanser.usd", 2.4),
)

INSTRUCTION_TEMPLATES = (
    ("zh_00", "zh", "到{object_zh}那里去"),
    ("zh_01", "zh", "走到{object_zh}旁边"),
    ("zh_02", "zh", "去找{object_zh}"),
    ("zh_03", "zh", "导航到{object_zh}"),
    ("zh_04", "zh", "前往放着{object_zh}的地方"),
    ("zh_05", "zh", "靠近{object_zh}"),
    ("zh_06", "zh", "请带我到{object_zh}附近"),
    ("zh_07", "zh", "找到{object_zh}并在它旁边停下"),
    ("zh_08", "zh", "向{object_zh}的位置前进"),
    ("zh_09", "zh", "去有{object_zh}的位置"),
    ("zh_10", "zh", "请在{object_zh}前面停下"),
    ("zh_11", "zh", "找一下{object_zh}在哪里并走过去"),
    ("en_00", "en", "go to the {object_en}"),
    ("en_01", "en", "walk over to the {object_en}"),
    ("en_02", "en", "find the {object_en}"),
    ("en_03", "en", "navigate to the {object_en}"),
    ("en_04", "en", "move to the place with the {object_en}"),
    ("en_05", "en", "approach the {object_en}"),
    ("en_06", "en", "take me near the {object_en}"),
    ("en_07", "en", "find the {object_en} and stop beside it"),
    ("en_08", "en", "head toward the {object_en}"),
    ("en_09", "en", "go where the {object_en} is"),
    ("en_10", "en", "stop in front of the {object_en}"),
    ("en_11", "en", "locate the {object_en} and move to it"),
    ("zh_val_00", "zh", "朝{object_zh}所在的地方走"),
    ("zh_val_01", "zh", "去{object_zh}跟前"),
    ("zh_val_02", "zh", "在房间里找到{object_zh}"),
    ("zh_val_03", "zh", "请前往{object_zh}的位置"),
    ("zh_val_04", "zh", "移动到{object_zh}旁"),
    ("zh_val_05", "zh", "向{object_zh}靠拢"),
    ("en_val_00", "en", "walk toward the location of the {object_en}"),
    ("en_val_01", "en", "go over beside the {object_en}"),
    ("en_val_02", "en", "find the {object_en} in the room"),
    ("en_val_03", "en", "please travel to the {object_en}"),
    ("en_val_04", "en", "move next to the {object_en}"),
    ("en_val_05", "en", "make your way toward the {object_en}"),
    ("zh_test_00", "zh", "带我去看{object_zh}"),
    ("zh_test_01", "zh", "开到{object_zh}那边"),
    ("zh_test_02", "zh", "去找到房间中的{object_zh}"),
    ("zh_test_03", "zh", "请去{object_zh}所在处"),
    ("zh_test_04", "zh", "到{object_zh}附近待命"),
    ("zh_test_05", "zh", "把{object_zh}作为目的地前进"),
    ("en_test_00", "en", "take me to see the {object_en}"),
    ("en_test_01", "en", "drive over to the {object_en}"),
    ("en_test_02", "en", "seek out the {object_en} in this room"),
    ("en_test_03", "en", "please go to where the {object_en} is located"),
    ("en_test_04", "en", "wait near the {object_en}"),
    ("en_test_05", "en", "use the {object_en} as your destination"),
)


def _box(name: str, position: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float]) -> dict:
    return {
        "id": name,
        "position": list(position),
        "size": list(size),
        "color": list(color),
        "collision": True,
        "lidar_visible": True,
    }


def _room_geometry(layout_index: int) -> list[dict]:
    wall = (0.70, 0.72, 0.74)
    wood = (0.32, 0.18, 0.08)
    fabric = (0.12, 0.34, 0.48)
    room_x = 9.0 + 0.3 * (layout_index % 3)
    room_y = 6.5 + 0.35 * ((layout_index + 1) % 3)
    geometry = [
        _box("wall_north", (0.0, room_y / 2.0, 1.25), (room_x, 0.16, 2.5), wall),
        _box("wall_south", (0.0, -room_y / 2.0, 1.25), (room_x, 0.16, 2.5), wall),
        _box("wall_east", (room_x / 2.0, 0.0, 1.25), (0.16, room_y, 2.5), wall),
        _box("wall_west", (-room_x / 2.0, 0.0, 1.25), (0.16, room_y, 2.5), wall),
    ]
    side = -1.0 if layout_index % 2 else 1.0
    geometry.extend(
        [
            _box("table_top", (1.0, side * 2.25, 0.66), (1.25, 0.72, 0.12), wood),
            _box("cabinet", (-2.75, -side * 2.35, 0.65), (1.15, 0.50, 1.30), (0.22, 0.25, 0.28)),
            _box("sofa", (-1.35, side * 2.35, 0.42), (1.70, 0.70, 0.84), fabric),
        ]
    )
    if layout_index in {1, 3, 5, 7, 9, 11}:
        geometry.append(_box("low_shelf", (2.8, -side * 2.25, 0.48), (1.15, 0.48, 0.96), (0.36, 0.29, 0.18)))
    if layout_index in {2, 6, 10}:
        geometry.append(_box("short_partition", (-0.65, -side * 2.55, 0.90), (2.1, 0.14, 1.8), wall))
    return geometry


def _hidden_room_geometry(layout_index: int) -> list[dict]:
    geometry = _room_geometry(layout_index)
    family_id = layout_index % HIDDEN_SEARCH_FAMILY_COUNT
    side = -1.0 if layout_index % 2 else 1.0
    screen_color = (0.35, 0.29, 0.27) if side > 0.0 else (0.28, 0.31, 0.36)
    screen_half_y = (1.05, 0.95, 1.15, 1.20)[family_id]
    screen_x = (1.48, 1.42, 1.56, 1.50)[family_id]
    geometry.append(
        _box(
            "search_screen",
            (screen_x, 0.0, 1.10),
            (0.10, screen_half_y, 1.10),
            screen_color,
        )
    )
    if family_id in {1, 3}:
        geometry.append(
            _box(
                "search_upper_panel",
                (2.08, 0.92 * side, 0.78),
                (0.08, 0.38, 0.78),
                (0.26, 0.24, 0.20),
            )
        )
    if family_id in {2, 3}:
        geometry.append(
            _box(
                "search_lower_panel",
                (2.28, -0.84 * side, 0.58),
                (0.08, 0.32, 0.58),
                (0.24, 0.27, 0.32),
            )
        )
    if family_id == 3:
        geometry.append(
            _box(
                "search_combo_bar",
                (1.92, 0.34 * side, 0.68),
                (0.10, 0.26, 0.68),
                (0.30, 0.26, 0.22),
            )
        )
    return geometry


def _hidden_target_slots(layout_index: int) -> list[dict]:
    side = -1.0 if layout_index % 2 else 1.0
    family_id = layout_index % HIDDEN_SEARCH_FAMILY_COUNT
    target_x = (3.12, 3.28, 3.44, 3.58)[family_id] + 0.04 * (layout_index % 2)
    target_y = side * (0.52 + 0.10 * (layout_index % 3))
    return [
        {"id": "hidden_a", "position": [target_x, target_y, 0.0], "visible_at_start": False},
        {"id": "hidden_b", "position": [target_x - 0.28, -target_y, 0.0], "visible_at_start": False},
    ]


def build_manifest(seed: int = 20260723, *, include_hidden_search: bool = False) -> dict:
    objects = [
        {
            "id": object_id,
            "label_zh": label_zh,
            "label_en": label_en,
            "usd_path": usd_path,
            "uniform_scale": scale,
            "source": "NVIDIA Isaac Sim 5.1 YCB",
            "license": "YCB dataset terms",
        }
        for object_id, label_zh, label_en, usd_path, scale in OBJECT_CATALOG
    ]
    templates = [
        {"id": template_id, "language": language, "text": value}
        for template_id, language, value in INSTRUCTION_TEMPLATES
    ]
    if include_hidden_search:
        scenes = []
        for index in range(12):
            if index < 8:
                split = "hidden_train"
                split_index = index
            elif index < 10:
                split = "hidden_validation"
                split_index = index - 8
            else:
                split = "hidden_test"
                split_index = index - 10
            side = -1.0 if index % 2 else 1.0
            family_id = index % HIDDEN_SEARCH_FAMILY_COUNT
            target_y = side * (0.52 + 0.14 * (index % 3))
            target_x = (3.12, 3.28, 3.44, 3.58)[family_id] + 0.04 * (index % 2)
            target_heading = math.degrees(math.atan2(target_y, target_x))
            scenes.append(
                {
                    "id": f"indoor_{split}_{split_index:02d}",
                    "split": split,
                    "layout_index": index,
                    "search_family_id": family_id,
                    "search_family_name": HIDDEN_SEARCH_FAMILY_NAMES[family_id],
                    "geometry": _hidden_room_geometry(index),
                    "target_slots": _hidden_target_slots(index),
                    "start_slots": [
                        {"id": "start_a", "position": [0.0, 0.0, 0.54], "yaw_deg": target_heading},
                        {"id": "start_b", "position": [-0.35, -side * 0.35, 0.54], "yaw_deg": target_heading + side * 12.0},
                    ],
                }
            )
        episodes = []
        split_scene_counts = {"hidden_train": 8, "hidden_validation": 2, "hidden_test": 2}
        split_offsets = {"hidden_train": 0, "hidden_validation": 8, "hidden_test": 10}
        split_episode_counts = {"hidden_train": 96, "hidden_validation": 24, "hidden_test": 36}
        template_ranges = {
            "hidden_train": (0, 24),
            "hidden_validation": (24, 36),
            "hidden_test": (36, 48),
        }
        for split, count in split_episode_counts.items():
            template_start, template_end = template_ranges[split]
            for index in range(count):
                scene_index = split_offsets[split] + index % split_scene_counts[split]
                object_index = (index // split_scene_counts[split]) % len(objects)
                template_index = template_start + index % (template_end - template_start)
                episodes.append(
                    {
                        "id": f"{split}_{index:04d}",
                        "split": split,
                        "scene_id": scenes[scene_index]["id"],
                        "object_category": objects[object_index]["id"],
                        "instruction_template_id": templates[template_index]["id"],
                        "target_slot_id": "hidden_a" if index % 2 == 0 else "hidden_b",
                        "start_slot_id": "start_a" if index % 3 else "start_b",
                        "search_required": True,
                        "target_visible_at_start": False,
                        "search_family_id": scenes[scene_index]["search_family_id"],
                        "search_family_name": scenes[scene_index]["search_family_name"],
                        "seed": seed + index + 10000 * split_offsets[split],
                    }
                )
    else:
        scenes = []
        for index in range(12):
            if index < 8:
                split = "train"
                split_index = index
            elif index < 10:
                split = "validation"
                split_index = index - 8
            else:
                split = "test_visible"
                split_index = index - 10
            side = -1.0 if index % 2 else 1.0
            target_y = side * (0.45 + 0.18 * (index % 3))
            target_x = 3.0 + 0.18 * (index % 4)
            target_heading = math.degrees(math.atan2(target_y, target_x))
            scenes.append(
                {
                    "id": f"indoor_{split}_{split_index:02d}",
                    "split": split,
                    "layout_index": index,
                    "geometry": _room_geometry(index),
                    "target_slots": [
                        {"id": "visible_a", "position": [target_x, target_y, 0.0], "visible_at_start": True},
                        {"id": "visible_b", "position": [target_x - 0.25, -target_y, 0.0], "visible_at_start": True},
                    ],
                    "start_slots": [
                        {"id": "start_a", "position": [0.0, 0.0, 0.54], "yaw_deg": target_heading},
                        {"id": "start_b", "position": [-0.35, -side * 0.35, 0.54], "yaw_deg": target_heading + side * 12.0},
                    ],
                }
            )

        episodes = []
        split_scene_counts = {"train": 8, "validation": 2, "test_visible": 2}
        split_offsets = {"train": 0, "validation": 8, "test_visible": 10}
        split_episode_counts = {"train": 96, "validation": 24, "test_visible": 36}
        template_ranges = {
            "train": (0, 24),
            "validation": (24, 36),
            "test_visible": (36, 48),
        }
        for split, count in split_episode_counts.items():
            template_start, template_end = template_ranges[split]
            for index in range(count):
                scene_index = split_offsets[split] + index % split_scene_counts[split]
                object_index = (index // split_scene_counts[split]) % len(objects)
                template_index = template_start + index % (template_end - template_start)
                episodes.append(
                    {
                        "id": f"{split}_{index:04d}",
                        "split": split,
                        "scene_id": scenes[scene_index]["id"],
                        "object_category": objects[object_index]["id"],
                        "instruction_template_id": templates[template_index]["id"],
                        # Training starts face visible_a. Keeping the target slot
                        # aligned with that heading makes the camera visibility
                        # claim physically testable instead of manifest-only.
                        "target_slot_id": (
                            "visible_a"
                            if split == "train"
                            else ("visible_a" if index % 2 == 0 else "visible_b")
                        ),
                        "start_slot_id": (
                            "start_a"
                            if split == "train"
                            else ("start_a" if index % 3 else "start_b")
                        ),
                        "seed": seed + index + 10000 * split_offsets[split],
                    }
                )

    return {
        "schema": HIDDEN_SCHEMA if include_hidden_search else VISIBLE_SCHEMA,
        "seed": seed,
        "simulation_only": True,
        "asset_root": "{ISAAC_NUCLEUS_DIR}",
        "object_catalog": objects,
        "instruction_templates": templates,
        "scenes": scenes,
        "episodes": episodes,
    }


def manifest_sha256(manifest: dict) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> dict:
    schema = manifest.get("schema")
    if schema not in {VISIBLE_SCHEMA, HIDDEN_SCHEMA}:
        raise ValueError(f"Unsupported scenario schema: {schema!r}")
    objects = manifest.get("object_catalog", [])
    templates = manifest.get("instruction_templates", [])
    scenes = manifest.get("scenes", [])
    episodes = manifest.get("episodes", [])
    object_ids = {item["id"] for item in objects}
    template_ids = {item["id"] for item in templates}
    scene_ids = {item["id"] for item in scenes}
    if len(object_ids) != len(objects) or len(object_ids) < 12:
        raise ValueError("Manifest must contain at least 12 unique object categories")
    if len(template_ids) != len(templates) or len(template_ids) < 24:
        raise ValueError("Manifest must contain at least 24 unique instruction templates")
    if len(scene_ids) != len(scenes):
        raise ValueError("Scene ids must be unique")
    if schema == VISIBLE_SCHEMA:
        train_scenes = {item["id"] for item in scenes if item.get("split") == "train"}
        if len(train_scenes) < 8:
            raise ValueError("Manifest must contain at least 8 unique training scenes")
    else:
        train_scenes = {item["id"] for item in scenes if item.get("split") == "hidden_train"}
        if len(train_scenes) < 8:
            raise ValueError("Hidden manifest must contain at least 8 unique hidden training scenes")
        hidden_families = {
            int(item.get("search_family_id"))
            for item in scenes
            if item.get("split", "").startswith("hidden_") and item.get("search_family_id") is not None
        }
        if len(hidden_families) < HIDDEN_SEARCH_FAMILY_COUNT:
            raise ValueError(
                f"Hidden manifest must cover at least {HIDDEN_SEARCH_FAMILY_COUNT} search families"
            )
    for scene in scenes:
        if not scene.get("geometry") or not scene.get("target_slots") or not scene.get("start_slots"):
            raise ValueError(f"Scene is incomplete: {scene['id']}")
        if not all(item.get("collision") and item.get("lidar_visible") for item in scene["geometry"]):
            raise ValueError(f"Every geometry item must collide and enter LiDAR: {scene['id']}")
        if schema == HIDDEN_SCHEMA:
            if not any(item["id"] == "search_screen" for item in scene["geometry"]):
                raise ValueError(f"Hidden scenes must include an occluding screen: {scene['id']}")
            if not all(not item.get("visible_at_start", True) for item in scene["target_slots"]):
                raise ValueError(f"Hidden scene targets must start occluded: {scene['id']}")
            if scene.get("search_family_id") not in range(HIDDEN_SEARCH_FAMILY_COUNT):
                raise ValueError(f"Hidden scene has invalid search family: {scene['id']}")
    episode_ids = set()
    for episode in episodes:
        if episode["id"] in episode_ids:
            raise ValueError(f"Duplicate episode id: {episode['id']}")
        episode_ids.add(episode["id"])
        if episode["scene_id"] not in scene_ids:
            raise ValueError(f"Unknown scene in {episode['id']}")
        if episode["object_category"] not in object_ids:
            raise ValueError(f"Unknown object in {episode['id']}")
        if episode["instruction_template_id"] not in template_ids:
            raise ValueError(f"Unknown instruction template in {episode['id']}")
    return {
        "schema": manifest["schema"],
        "sha256": manifest_sha256(manifest),
        "training_scenes": len(train_scenes),
        "all_scenes": len(scene_ids),
        "object_categories": len(object_ids),
        "instruction_templates": len(template_ids),
        "hidden_search_families": len(
            {
                int(item.get("search_family_id"))
                for item in scenes
                if item.get("split", "").startswith("hidden_") and item.get("search_family_id") is not None
            }
        )
        if schema == HIDDEN_SCHEMA
        else 0,
        "episodes": len(episodes),
    }


def resolve_episode(manifest: dict, episode_id: str) -> dict:
    episode = next((item for item in manifest["episodes"] if item["id"] == episode_id), None)
    if episode is None:
        raise KeyError(f"Unknown scenario episode: {episode_id}")
    scene = next(item for item in manifest["scenes"] if item["id"] == episode["scene_id"])
    object_cfg = next(item for item in manifest["object_catalog"] if item["id"] == episode["object_category"])
    template = next(
        item for item in manifest["instruction_templates"] if item["id"] == episode["instruction_template_id"]
    )
    target_slot = next(item for item in scene["target_slots"] if item["id"] == episode["target_slot_id"])
    start_slot = next(item for item in scene["start_slots"] if item["id"] == episode["start_slot_id"])
    task_text = template["text"].format(object_zh=object_cfg["label_zh"], object_en=object_cfg["label_en"])
    return {
        "episode": episode,
        "scene": scene,
        "object": object_cfg,
        "instruction_template": template,
        "target_slot": target_slot,
        "start_slot": start_slot,
        "task_text": task_text,
        "search_required": bool(episode.get("search_required", False)),
        "target_visible_at_start": bool(target_slot.get("visible_at_start", True)),
        "search_family_id": scene.get("search_family_id"),
        "search_family_name": scene.get("search_family_name"),
        "manifest_sha256": manifest_sha256(manifest),
    }

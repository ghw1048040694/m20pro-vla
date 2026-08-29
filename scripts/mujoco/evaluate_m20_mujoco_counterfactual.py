#!/usr/bin/env python3
"""Evaluate same-scene language swaps without exposing target poses to VLA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from play_m20_mujoco_vla import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--stop-threshold", type=float, default=0.50)
    parser.add_argument("--stop-confirm", type=int, default=2)
    parser.add_argument("--motion-smoothing", type=float, default=0.35)
    parser.add_argument("--min-stop-separation", type=float, default=0.25)
    return parser.parse_args()


def canonical_scene(metadata: dict) -> str:
    objects = sorted(metadata["objects"], key=lambda item: item["name"])
    return json.dumps(objects, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if config.get("split_unit") != "layout_id":
        raise RuntimeError("Counterfactual evaluation requires a layout-disjoint training split")
    episode_paths = [Path(value) for value in config.get("val_episodes", [])]
    episode_json_paths = [path if path.suffix == ".json" else path.with_suffix(".json") for path in episode_paths]
    if not episode_json_paths:
        raise RuntimeError("Checkpoint contains no held-out episodes")
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in episode_json_paths]
    if any(item.get("collection_mode") != "counterfactual" for item in metadata):
        raise RuntimeError("Held-out episodes are not counterfactual data")
    groups: dict[int, list[tuple[Path, dict]]] = {}
    for path, item in zip(episode_json_paths, metadata):
        groups.setdefault(int(item["layout_id"]), []).append((path, item))
    if any(len(group) != 3 for group in groups.values()):
        raise RuntimeError("Each held-out layout must contain exactly three language swaps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layouts = []
    for layout_id, group in sorted(groups.items()):
        scene_ids = {canonical_scene(item) for _, item in group}
        initial_yaws = {float(item.get("initial_yaw", float("nan"))) for _, item in group}
        initial_xys = {
            tuple(float(value) for value in item.get("initial_xy", (0.0, 0.0)))
            for _, item in group
        }
        same_scene = len(scene_ids) == 1 and initial_yaws == {0.0} and len(initial_xys) == 1
        runs = []
        for episode_path, item in sorted(group, key=lambda value: value[1]["target_label"]):
            stem = f"layout_{layout_id:03d}_{item['target_label'].replace(' ', '_')}"
            report = run_episode(argparse.Namespace(
                checkpoint=args.checkpoint,
                episode_json=episode_path,
                output=args.output_dir / f"{stem}.mp4",
                metrics=args.output_dir / f"{stem}.json",
                device=args.device,
                decision_mode="direct",
                trajectory_scorer=None,
                stop_threshold=args.stop_threshold,
                stop_confirm=args.stop_confirm,
                motion_smoothing=args.motion_smoothing,
                steps=args.steps,
                width=640,
                height=360,
            ))
            final_xy = np.asarray(report["final_base_xy"], dtype=np.float64)
            distances = {
                obj["label"]: float(np.linalg.norm(final_xy - np.asarray(obj["position"], dtype=np.float64)))
                for obj in item["objects"]
            }
            selected_label = min(distances, key=distances.get)
            runs.append({
                "target_label": item["target_label"],
                "task_text": item["task_text"],
                "success": report["success"],
                "final_base_xy": report["final_base_xy"],
                "nearest_object_label_metrics_only": selected_label,
                "target_selection_correct": selected_label == item["target_label"],
                "report": str(args.output_dir / f"{stem}.json"),
                "video": str(args.output_dir / f"{stem}.mp4"),
            })
        final_points = [np.asarray(run["final_base_xy"], dtype=np.float64) for run in runs]
        pairwise = [float(np.linalg.norm(left - right)) for index, left in enumerate(final_points) for right in final_points[index + 1:]]
        distinct_stops = min(pairwise) >= args.min_stop_separation
        layouts.append({
            "layout_id": layout_id,
            "same_scene_and_initial_pose": same_scene,
            "runs": runs,
            "pairwise_final_stop_separation_m": pairwise,
            "distinct_stops": distinct_stops,
            "passed": same_scene and distinct_stops and all(run["success"] and run["target_selection_correct"] for run in runs),
        })
    summary = {
        "schema": "m20pro_mujoco_counterfactual_language_gate_v1",
        "checkpoint": str(args.checkpoint),
        "split_unit": config["split_unit"],
        "validation_layout_ids": config.get("validation_layout_ids", []),
        "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "language"],
        "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
        "acceptance": "For every held-out identical layout, all three instructions must select their own target and finish at distinct stops.",
        "layouts": layouts,
        "passed": bool(layouts) and all(layout["passed"] for layout in layouts),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

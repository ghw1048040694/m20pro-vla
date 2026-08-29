#!/usr/bin/env python3
"""Collect MuJoCo M20 DAgger correction data for closed-loop VLA failures.

The saved training label is always the privileged expert command.  The command
executed in simulation is a blend of the current VLA command and the expert,
with temporary full-expert interventions when the learner stalls, tries to
stop far from the target, or gets too close to obstacles.  This puts the
dataset on the learner's state distribution without throwing away recovery
labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_m20_mujoco_vla import (  # noqa: E402
    EpisodePlan,
    OBJECTS,
    SEARCH_CLUTTER_TEMPLATES,
    SEARCH_LAYOUT_TEMPLATES,
    SEARCH_OBSTACLE_TEMPLATES,
    SEARCH_OUTER_CLUTTER_TEMPLATES,
    _appearance_rng,
    _minimum_obstacle_clearance,
    _sample_scene_light,
    _search_initial_pose_variants_for,
    _search_terrain_profile_for_layout,
    _target_visible_from,
    _task_language_for,
    make_search_layout,
)
from m20pro_vla.low_level import M20LowLevelController  # noqa: E402
from m20pro_vla.data import HISTORY_FEATURE_LABELS, VisualHistoryTracker  # noqa: E402
from m20pro_vla.planning import SearchMPCConfig, SearchMPCPlanner  # noqa: E402
from m20pro_vla.policies import M20MuJoCoVLA, encode_text  # noqa: E402
from m20pro_vla.sim.mujoco import (  # noqa: E402
    ASSET,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ObjectSpec,
    ObstacleSpec,
    SceneLightSpec,
    TerrainSpec,
    build_scene,
    open_video,
    planar_lidar,
    proprioception,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/datasets/m20_search_strict_flat_highres_dagger_v1"))
    parser.add_argument("--video-dir", type=Path, default=Path(".runtime/videos/m20_search_strict_flat_highres_dagger_v1"))
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--layout-offset", type=int, default=500)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--source-episode-json",
        type=Path,
        action="append",
        default=[],
        help="Existing episode metadata JSON to replay as a DAgger correction scene. Can be passed multiple times.",
    )
    parser.add_argument(
        "--source-episode-glob",
        default="",
        help="Glob for existing episode metadata JSON files to replay as DAgger correction scenes.",
    )
    parser.add_argument("--steps", type=int, default=1300)
    parser.add_argument("--warmup-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--search-start-mode", choices=("balanced", "anchor", "varied"), default="balanced")
    parser.add_argument("--search-terrain-profile", choices=("cycle", "flat", "slope", "bumps", "step"), default="flat")
    parser.add_argument("--search-layout-template-id", type=int, default=None)
    parser.add_argument("--search-layout-scale", type=float, default=1.0)
    parser.add_argument("--language-mode", choices=("varied", "fixed"), default="varied")
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--success-hold-steps", type=int, default=35)
    parser.add_argument(
        "--expert-execution-alpha",
        type=float,
        default=0.35,
        help="Blend factor for executed motion: alpha*expert + (1-alpha)*learner outside intervention bursts.",
    )
    parser.add_argument("--motion-smoothing", type=float, default=0.35)
    parser.add_argument("--intervention-stall-steps", type=int, default=140)
    parser.add_argument("--intervention-burst-steps", type=int, default=70)
    parser.add_argument("--intervention-clearance-m", type=float, default=0.20)
    parser.add_argument("--stop-threshold", type=float, default=0.50)
    parser.add_argument("--policy-query-interval", type=int, default=4)
    parser.add_argument("--policy-width", type=int, default=160)
    parser.add_argument("--policy-height", type=int, default=96)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def yaw_from_qpos(qpos: np.ndarray) -> float:
    return float(
        math.atan2(
            2.0 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
            1.0 - 2.0 * (qpos[5] ** 2 + qpos[6] ** 2),
        )
    )


def clip_body_motion(command: np.ndarray) -> np.ndarray:
    values = np.asarray(command, dtype=np.float64).copy()
    return np.asarray(
        (
            float(np.clip(values[0], 0.0, 0.35)),
            float(np.clip(values[1], -0.10, 0.10)),
            float(np.clip(values[2], -0.15, 0.15)),
        ),
        dtype=np.float64,
    )


@torch.no_grad()
def learner_action(
    policy: M20MuJoCoVLA,
    *,
    device: torch.device,
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    lidar: np.ndarray,
    proprio: np.ndarray,
    history: np.ndarray,
    task_text: str,
) -> tuple[np.ndarray, float, int]:
    rgb = np.concatenate((front_rgb, rear_rgb), axis=-1).transpose(2, 0, 1)
    rgb_tensor = torch.from_numpy(rgb.copy()).unsqueeze(0).to(device).float().div_(255.0)
    lidar_tensor = torch.from_numpy(lidar.copy()).unsqueeze(0).to(device).float().div_(10.0)
    proprio_tensor = torch.from_numpy(proprio.copy()).unsqueeze(0).to(device).float()
    history_tensor = torch.from_numpy(history.copy()).unsqueeze(0).to(device).float()
    language_tensor = torch.from_numpy(encode_text(task_text)).unsqueeze(0).to(device)
    latent = policy.encode(rgb_tensor, lidar_tensor, proprio_tensor, language_tensor, history=history_tensor)
    target_logits = policy.target_head(latent)
    phase_logits = policy.phase_head(latent)
    action = policy.action_from_latent(
        latent,
        policy._target_context_from_logits(target_logits),
        phase_logits=phase_logits,
        history=history_tensor,
    )[0].detach().cpu().numpy()
    stop_probability = float(torch.sigmoid(torch.tensor(float(action[3]))).item())
    phase_id = int(torch.softmax(phase_logits, dim=-1).argmax(dim=-1).item())
    return action.astype(np.float64, copy=False), stop_probability, phase_id


def make_search_plans(args: argparse.Namespace) -> list[EpisodePlan]:
    source_paths = [Path(path) for path in args.source_episode_json]
    if str(args.source_episode_glob or ""):
        source_paths.extend(Path(path) for path in sorted(glob.glob(str(args.source_episode_glob))))
    if source_paths:
        plans = []
        seen: set[Path] = set()
        for source_index, path in enumerate(source_paths):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            plans.append(
                episode_plan_from_metadata(
                    metadata,
                    episode_id=int(args.episode_offset) + len(plans),
                    fallback_layout_id=int(args.layout_offset) + source_index,
                )
            )
        return plans

    plans: list[EpisodePlan] = []
    rng = np.random.default_rng(args.seed)
    episode_id = int(args.episode_offset)
    for layout_index in range(int(args.layouts)):
        layout_id = int(args.layout_offset) + layout_index
        terrain_profile = (
            _search_terrain_profile_for_layout(layout_id)
            if args.search_terrain_profile == "cycle"
            else str(args.search_terrain_profile)
        )
        appearance_rng = _appearance_rng(args.seed, "dagger", layout_id)
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
            seed=args.seed,
            layout_scale=float(args.search_layout_scale),
            appearance_rng=appearance_rng,
            forced_template_id=args.search_layout_template_id,
        )
        for _, target_label, _, _ in OBJECTS:
            target_xy = targets[target_label]
            variants = _search_initial_pose_variants_for(
                mode=args.search_start_mode,
                seed=args.seed,
                layout_id=layout_id,
                target_label=target_label,
                target_xy=target_xy,
            )
            for variant_name, start_xy, start_xy_template_id, start_yaw, start_yaw_offset in variants:
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
                        scene_light=_sample_scene_light(_appearance_rng(args.seed, "dagger_light", episode_id, target_label)),
                        search_start_mode=args.search_start_mode,
                        search_start_variant=variant_name,
                        search_start_xy_template_id=start_xy_template_id,
                        search_start_yaw_offset=start_yaw_offset,
                        search_required=True,
                        initial_target_visible=_target_visible_from(start_xy, target_xy, obstacles),
                    )
                )
                episode_id += 1
    return plans


def _metadata_object_specs(metadata: dict) -> list[ObjectSpec]:
    objects = []
    for item in metadata["objects"]:
        name = str(item["name"])
        kind = str(item.get("kind") or ("cylinder" if "cylinder" in name else "box"))
        default_size = (0.14, 0.20) if kind == "cylinder" else (0.16, 0.16, 0.16)
        objects.append(
            ObjectSpec(
                name=name,
                label=str(item.get("label", name)),
                kind=kind,
                position=tuple(float(value) for value in item["position"][:2]),
                rgba=tuple(float(value) for value in item.get("rgba", (0.8, 0.8, 0.8, 1.0))),
                size=tuple(float(value) for value in item.get("size", default_size)),
            )
        )
    return objects


def _metadata_obstacle_specs(metadata: dict) -> list[ObstacleSpec]:
    obstacles = []
    for item in metadata.get("obstacles", []):
        obstacles.append(
            ObstacleSpec(
                name=str(item["name"]),
                kind=str(item.get("kind", "box")),
                position=tuple(float(value) for value in item["position"]),
                size=tuple(float(value) for value in item["size"]),
                rgba=tuple(float(value) for value in item.get("rgba", (0.30, 0.33, 0.36, 1.0))),
                euler=tuple(float(value) for value in item.get("euler", (0.0, 0.0, 0.0))),
                contype=int(item.get("contype", 1)),
                conaffinity=int(item.get("conaffinity", 1)),
                group=int(item.get("group", 0)),
                density=(None if item.get("density") is None else float(item["density"])),
            )
        )
    return obstacles


def _metadata_scene_light(metadata: dict) -> SceneLightSpec | None:
    item = metadata.get("scene_light") or {}
    if not item:
        return None
    return SceneLightSpec(
        pos=tuple(float(value) for value in item.get("pos", (0.0, 0.0, 5.0))),
        direction=tuple(float(value) for value in item.get("direction", (0.0, 0.0, -1.0))),
        diffuse=tuple(float(value) for value in item.get("diffuse", (0.85, 0.85, 0.85))),
        ambient=tuple(float(value) for value in item.get("ambient", (0.0, 0.0, 0.0))),
        specular=tuple(float(value) for value in item.get("specular", (0.0, 0.0, 0.0))),
        directional=bool(item.get("directional", True)),
    )


def episode_plan_from_metadata(
    metadata: dict,
    *,
    episode_id: int,
    fallback_layout_id: int,
) -> EpisodePlan:
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    start_xy = np.asarray(metadata.get("initial_xy", (0.0, 0.0)), dtype=np.float64)
    obstacles = _metadata_obstacle_specs(metadata)
    return EpisodePlan(
        episode_id=int(episode_id),
        layout_id=int(metadata.get("layout_id", fallback_layout_id)),
        terrain_profile=str(metadata.get("terrain_profile", "flat")),
        search_template_id=(
            None
            if metadata.get("search_layout_template_id") is None
            else int(metadata.get("search_layout_template_id"))
        ),
        search_obstacle_template_id=(
            None
            if metadata.get("search_obstacle_template_id") is None
            else int(metadata.get("search_obstacle_template_id"))
        ),
        search_secondary_obstacle_template_id=(
            None
            if metadata.get("search_secondary_obstacle_template_id") is None
            else int(metadata.get("search_secondary_obstacle_template_id"))
        ),
        search_outer_obstacle_template_id=(
            None
            if metadata.get("search_outer_obstacle_template_id") is None
            else int(metadata.get("search_outer_obstacle_template_id"))
        ),
        yaw=float(metadata.get("initial_yaw", 0.0)),
        start_xy=start_xy,
        target_xy=target_xy,
        target_label=str(metadata["target_label"]),
        objects=_metadata_object_specs(metadata),
        obstacles=obstacles,
        scene_light=_metadata_scene_light(metadata),
        search_start_mode=str(metadata.get("search_start_mode", "source_json")),
        search_start_variant=str(metadata.get("search_start_variant", "source_json")),
        search_start_xy_template_id=int(metadata.get("search_start_xy_template_id", -1)),
        search_start_yaw_offset=float(metadata.get("search_start_yaw_offset_rad", 0.0)),
        search_required=bool(metadata.get("search_required", True)),
        initial_target_visible=bool(
            metadata.get(
                "initial_target_visible",
                _target_visible_from(start_xy, target_xy, obstacles),
            )
        ),
    )


def run_episode(
    args: argparse.Namespace,
    *,
    policy: M20MuJoCoVLA,
    device: torch.device,
    plan: EpisodePlan,
) -> dict:
    out = args.output_dir / f"episode_{plan.episode_id:04d}.npz"
    meta_out = args.output_dir / f"episode_{plan.episode_id:04d}.json"
    video_path = args.video_dir / f"episode_{plan.episode_id:04d}.mp4"
    if (out.exists() or meta_out.exists()) and not args.overwrite:
        raise FileExistsError(f"Episode exists; pass --overwrite: {out}")
    scene_path = args.output_dir / f"scene_{plan.episode_id:04d}.xml"
    terrain = None if plan.terrain_profile == "flat" else TerrainSpec(plan.terrain_profile)
    build_scene(
        scene_path,
        plan.objects,
        obstacles=plan.obstacles,
        task_object_collisions=False,
        terrain=terrain,
        light=plan.scene_light,
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controller = M20LowLevelController(model)
    controller.reset(data, plan.yaw, plan.start_xy)
    planner = SearchMPCPlanner(
        model,
        plan.target_xy,
        plan.obstacles,
        SearchMPCConfig(
            target_radius=float(args.success_radius),
            stop_distance=max(0.24, float(args.success_radius) + 0.02),
            use_route_planner_when_obstacles=True,
        ),
    )
    task_language = _task_language_for(
        target_label=plan.target_label,
        episode_id=plan.episode_id,
        seed=args.seed,
        language_mode=args.language_mode,
    )
    task_text = task_language["task_text"]
    renderer = mujoco.Renderer(model, height=args.policy_height, width=args.policy_width)
    video = open_video(video_path, args.policy_width, args.policy_height) if args.video else None

    front = np.zeros((args.steps, args.policy_height, args.policy_width, 3), dtype=np.uint8)
    rear = np.zeros_like(front)
    lidar = np.zeros((args.steps, 72), dtype=np.float32)
    proprio = np.zeros((args.steps, 45), dtype=np.float32)
    action = np.zeros((args.steps, 4), dtype=np.float32)
    learner_action_array = np.zeros_like(action)
    executed_action = np.zeros_like(action)
    expert_intervention = np.zeros((args.steps,), dtype=np.float32)
    learner_stop_probability = np.zeros((args.steps,), dtype=np.float32)
    learner_phase_id = np.zeros((args.steps,), dtype=np.int64)

    reached_step = -1
    min_distance = float("inf")
    min_height = float(data.qpos[2])
    min_obstacle_clearance = _minimum_obstacle_clearance(data.qpos[:2], plan.obstacles)
    target_first_visible_step = -1
    target_visible_steps = 0
    collected_steps = 0
    planner_info: dict[str, float | list[float]] = {}
    previous_motion = np.zeros(3, dtype=np.float64)
    previous_executed_action = np.zeros(4, dtype=np.float32)
    history_tracker = VisualHistoryTracker()
    best_distance = float("inf")
    stall_steps = 0
    intervention_remaining = 0
    intervention_count = 0
    learner_far_stop_count = 0
    terminated_early = False
    cached_learner = np.zeros(4, dtype=np.float64)
    cached_stop_prob = 0.0
    cached_phase_id = 0
    policy_query_count = 0
    policy_cached_count = 0

    try:
        for step in range(int(args.steps) + int(args.warmup_steps)):
            if step < int(args.warmup_steps):
                controller.step(data, np.zeros(4, dtype=np.float64))
                continue

            index = collected_steps
            base_xy = data.qpos[:2].copy()
            distance = float(np.linalg.norm(plan.target_xy - base_xy))
            min_distance = min(min_distance, distance)
            min_height = min(min_height, float(data.qpos[2]))
            target_visible_now = _target_visible_from(base_xy, plan.target_xy, plan.obstacles)
            if target_visible_now:
                target_visible_steps += 1
                if target_first_visible_step < 0:
                    target_first_visible_step = index
            clearance_now = _minimum_obstacle_clearance(base_xy, plan.obstacles)
            if clearance_now is not None:
                min_obstacle_clearance = (
                    clearance_now
                    if min_obstacle_clearance is None
                    else min(float(min_obstacle_clearance), clearance_now)
                )

            if distance <= float(args.success_radius):
                expert = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                if reached_step < 0:
                    reached_step = index
            else:
                expert, planner_info = planner.recommend(data, controller)
                expert = np.asarray(expert, dtype=np.float64)
                expert[3] = 0.0

            renderer.update_scene(data, camera="front_rgb")
            front[index] = renderer.render().copy()
            renderer.update_scene(data, camera="rear_rgb")
            rear[index] = renderer.render().copy()
            lidar[index] = planar_lidar(model, data)
            proprio[index] = proprioception(model, data)
            action[index] = expert.astype(np.float32)
            history_feature = history_tracker.observe(
                front[index],
                rear[index],
                plan.target_label,
                previous_action=previous_executed_action,
            )

            if index == 0 or index % int(args.policy_query_interval) == 0:
                learner, stop_prob, phase_id = learner_action(
                    policy,
                    device=device,
                    front_rgb=front[index],
                    rear_rgb=rear[index],
                    lidar=lidar[index],
                    proprio=proprio[index],
                    history=history_feature,
                    task_text=task_text,
                )
                cached_learner = learner.copy()
                cached_stop_prob = float(stop_prob)
                cached_phase_id = int(phase_id)
                policy_query_count += 1
            else:
                learner = cached_learner.copy()
                stop_prob = float(cached_stop_prob)
                phase_id = int(cached_phase_id)
                policy_cached_count += 1
            learner_action_array[index] = learner.astype(np.float32)
            learner_stop_probability[index] = stop_prob
            learner_phase_id[index] = phase_id

            if distance < best_distance - 0.03:
                best_distance = distance
                stall_steps = 0
            else:
                stall_steps += 1
            if stop_prob >= float(args.stop_threshold) and distance > float(args.success_radius) + 0.20:
                learner_far_stop_count += 1
                intervention_remaining = max(intervention_remaining, int(args.intervention_burst_steps))
            if clearance_now is not None and clearance_now < float(args.intervention_clearance_m):
                intervention_remaining = max(intervention_remaining, int(args.intervention_burst_steps))
            if stall_steps >= int(args.intervention_stall_steps) and (target_visible_now or index > 280):
                intervention_remaining = max(intervention_remaining, int(args.intervention_burst_steps))
                stall_steps = 0

            if expert[3] >= 0.5:
                executed = expert.copy()
            else:
                learner_motion = clip_body_motion(learner)
                expert_motion = clip_body_motion(expert)
                if intervention_remaining > 0:
                    mixed_motion = expert_motion
                    expert_intervention[index] = 1.0
                    intervention_remaining -= 1
                    intervention_count += 1
                else:
                    alpha = float(args.expert_execution_alpha)
                    mixed_motion = alpha * expert_motion + (1.0 - alpha) * learner_motion
                motion = float(args.motion_smoothing) * mixed_motion + (1.0 - float(args.motion_smoothing)) * previous_motion
                previous_motion = motion
                executed = np.asarray((motion[0], motion[1], motion[2], 0.0), dtype=np.float64)

            executed_action[index] = executed.astype(np.float32)
            previous_executed_action = executed_action[index].copy()
            if video is not None:
                video.stdin.write(front[index].tobytes())
            collected_steps += 1
            controller.step(data, executed)

            if reached_step >= 0 and index >= reached_step + int(args.success_hold_steps):
                terminated_early = True
                break
    finally:
        if video is not None:
            video.stdin.close()
            return_code = video.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed for {video_path}: {return_code}")
        renderer.close()
        scene_path.unlink(missing_ok=True)

    proprio_finite = bool(np.isfinite(proprio[:collected_steps]).all())
    success = bool(reached_step >= 0 and min_height >= 0.45 and proprio_finite)
    metadata = {
        "schema": "m20pro_mujoco_vla_episode_v1",
        "episode_id": plan.episode_id,
        "collection_mode": "search",
        "dagger": True,
        "dagger_contract": "observations are from mixed learner/expert execution; action labels are privileged SearchMPC expert commands",
        "checkpoint": str(args.checkpoint),
        "execution_policy": "mixed_vla_expert_dagger",
        "expert_execution_alpha": float(args.expert_execution_alpha),
        "intervention_count": int(intervention_count),
        "intervention_fraction": float(intervention_count / max(1, collected_steps)),
        "learner_far_stop_count": int(learner_far_stop_count),
        "policy_query_interval": int(args.policy_query_interval),
        "policy_query_count": int(policy_query_count),
        "policy_cached_count": int(policy_cached_count),
        "layout_id": plan.layout_id,
        "terrain_profile": plan.terrain_profile,
        "terrain_version": terrain.version if terrain is not None else "",
        "initial_yaw": plan.yaw,
        "initial_xy": plan.start_xy.tolist(),
        "warmup_steps": int(args.warmup_steps),
        "steps": int(collected_steps),
        "terminated_early": bool(terminated_early),
        "success_radius": float(args.success_radius),
        "target_reached_step": int(reached_step),
        "min_target_distance": float(min_distance),
        "final_target_distance": float(np.linalg.norm(plan.target_xy - data.qpos[:2])),
        "min_base_height": float(min_height),
        "success": success,
        "canonical_success": success,
        "search_required": True,
        "initial_target_visible": bool(plan.initial_target_visible),
        "search_start_mode": plan.search_start_mode,
        "search_start_variant": plan.search_start_variant,
        "search_start_xy_template_id": int(plan.search_start_xy_template_id),
        "search_start_yaw_offset_rad": float(plan.search_start_yaw_offset),
        "search_policy": "dagger_mixed_execution_search_mpc_labels",
        "search_policy_requested": "dagger",
        "search_policy_effective": "search_mpc_labels",
        "target_label": plan.target_label,
        "target_xy_privileged_label_only": plan.target_xy.tolist(),
        "objects": [
            {
                "name": obj.name,
                "label": obj.label,
                "kind": obj.kind,
                "position": list(obj.position),
                "rgba": list(obj.rgba),
                "size": list(obj.size),
            }
            for obj in plan.objects
        ],
        "obstacles": [
            {
                "name": obs.name,
                "kind": obs.kind,
                "position": list(obs.position),
                "size": list(obs.size),
                "rgba": list(obs.rgba),
            }
            for obs in plan.obstacles
        ],
        "task_object_collisions": False,
        **task_language,
        "search_layout_template_id": plan.search_template_id,
        "search_obstacle_template_id": plan.search_obstacle_template_id,
        "search_secondary_obstacle_template_id": plan.search_secondary_obstacle_template_id,
        "search_outer_obstacle_template_id": plan.search_outer_obstacle_template_id,
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
        "target_first_visible_step": int(target_first_visible_step),
        "target_visible_steps": int(target_visible_steps),
        "target_visibility_fraction": float(target_visible_steps / max(1, collected_steps)),
        "target_discovered": target_first_visible_step >= 0,
        "target_discovered_before_reach": (
            target_first_visible_step >= 0 and (reached_step < 0 or target_first_visible_step <= reached_step)
        ),
        "min_obstacle_clearance": min_obstacle_clearance,
        "planner": planner_info,
        "video": str(video_path) if args.video else "",
        "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "visual_history_16", "language"],
        "policy_input_resolution": {"width": int(args.policy_width), "height": int(args.policy_height)},
        "history_contract": "short memory from policy RGB target pixels + previous executed body command; no target_xy/object_id/semantic mask",
        "history_feature_labels": list(HISTORY_FEATURE_LABELS),
        "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
        "extra_arrays": ["learner_action", "executed_action", "expert_intervention", "learner_stop_probability", "learner_phase_id"],
    }
    meta_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out,
        front_rgb=front[:collected_steps],
        rear_rgb=rear[:collected_steps],
        lidar=lidar[:collected_steps],
        proprio=proprio[:collected_steps],
        action=action[:collected_steps],
        learner_action=learner_action_array[:collected_steps],
        executed_action=executed_action[:collected_steps],
        expert_intervention=expert_intervention[:collected_steps],
        learner_stop_probability=learner_stop_probability[:collected_steps],
        learner_phase_id=learner_phase_id[:collected_steps],
    )
    return metadata


def main() -> None:
    args = parse_args()
    if args.layouts <= 0 or args.steps <= args.warmup_steps or args.warmup_steps < 1:
        raise ValueError("invalid DAgger episode configuration")
    if not 0.0 <= args.expert_execution_alpha <= 1.0:
        raise ValueError("--expert-execution-alpha must be in [0, 1]")
    if not 0.0 < args.motion_smoothing <= 1.0:
        raise ValueError("--motion-smoothing must be in (0, 1]")
    if args.success_radius <= 0.0 or args.success_hold_steps < 0:
        raise ValueError("success settings are invalid")
    if args.policy_width <= 0 or args.policy_height <= 0:
        raise ValueError("policy RGB resolution must be positive")
    if args.policy_query_interval <= 0:
        raise ValueError("--policy-query-interval must be positive")
    if not ASSET.is_file():
        raise FileNotFoundError(f"Build the asset first: {ASSET}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    policy = M20MuJoCoVLA.for_state_dict(state_dict).to(device)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    plans = make_search_plans(args)
    summaries = []
    for plan in plans:
        metadata = run_episode(args, policy=policy, device=device, plan=plan)
        summaries.append(metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)

    summary = {
        "schema": "m20pro_mujoco_dagger_dataset_v1",
        "checkpoint": str(args.checkpoint),
        "episodes": summaries,
        "episode_count": len(summaries),
        "success_count": sum(bool(item.get("success", False)) for item in summaries),
        "canonical_success_count": sum(bool(item.get("canonical_success", False)) for item in summaries),
        "target_discovered_count": sum(bool(item.get("target_discovered", False)) for item in summaries),
        "mean_intervention_fraction": float(np.mean([float(item.get("intervention_fraction", 0.0)) for item in summaries])) if summaries else 0.0,
        "policy_input_resolution": {"width": int(args.policy_width), "height": int(args.policy_height)},
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": str(args.output_dir),
                "episodes": len(summaries),
                "success_count": summary["success_count"],
                "target_discovered_count": summary["target_discovered_count"],
                "mean_intervention_fraction": summary["mean_intervention_fraction"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

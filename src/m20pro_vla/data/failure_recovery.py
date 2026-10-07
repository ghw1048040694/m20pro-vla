"""Expert continuations from safe states reached by a failed closed-loop policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from m20pro_vla.low_level import M20LowLevelController, backend_of, build_low_level_controller
from m20pro_vla.planning import (
    GlobalPlanner, GlobalPlannerConfig, SearchMPCConfig, SearchMPCPlanner, obstacle_clearance_xy,
)
from m20pro_vla.sim.mujoco import planar_lidar, proprioception


def capture_state(data: mujoco.MjData, controller: M20LowLevelController, step: int, target_xy: np.ndarray,
                  obstacle_clearance: float | None) -> dict:
    return {
        "step": int(step),
        "target_distance": float(np.linalg.norm(data.qpos[:2] - target_xy)),
        "obstacle_clearance": float(obstacle_clearance) if obstacle_clearance is not None else float("inf"),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy(),
        "ctrl": data.ctrl.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
        "time": float(data.time),
        "controller": controller.snapshot(),
        # Which execution layer produced this state. Without it a recovery
        # episode built later could silently replay an analytic state into the
        # v5 expert and inherit dynamics the source episode never had.
        "low_level_backend": backend_of(controller),
    }


def select_states(states: list[dict], max_episodes: int) -> list[dict]:
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if not states:
        return []
    closest = min(states, key=lambda state: state["target_distance"])
    earlier = [state for state in states if state["step"] <= closest["step"] - 50]
    ranked = [
        closest,
        max(earlier, key=lambda state: state["step"]) if earlier else states[0],
        min(states, key=lambda state: state["obstacle_clearance"]),
        states[-1],
        states[0],
    ]
    selected: list[dict] = []
    for state in ranked:
        if all(abs(state["step"] - other["step"]) >= 25 for other in selected):
            selected.append(state)
        if len(selected) >= max_episodes:
            break
    return selected


def restore_controller_state(controller, captured, *, backend, step):
    """Restore a typed executor snapshot only into its matching execution layer."""
    expected = controller.snapshot()
    if type(captured) is not type(expected):
        raise ValueError(
            f"recovery state at step {step} was captured by backend {backend!r} "
            f"as {type(captured).__name__}, but {type(controller).__name__} "
            f"requires {type(expected).__name__}; re-collect with the matching backend"
        )
    controller.restore(captured)


def build_recovery_planner(model, target_xy, obstacles, metadata, success_radius):
    """Use the same doorway-aware teacher as structured demonstration collection."""
    structured = metadata.get("scene_kind") in {"s2", "s3"}
    return SearchMPCPlanner(
        model,
        target_xy,
        obstacles,
        SearchMPCConfig(
            target_radius=success_radius,
            stop_distance=success_radius,
            use_route_planner_when_obstacles=True,
        ),
        global_planner=GlobalPlanner(obstacles, GlobalPlannerConfig()) if structured else None,
    )


def collect_continuations(
    *,
    model: mujoco.MjModel,
    states: list[dict],
    metadata: dict,
    obstacles: list,
    source_episode_json: Path,
    checkpoint: Path,
    output_dir: Path,
    max_episodes: int,
    max_steps: int,
    success_radius: float,
) -> list[dict]:
    if max_steps <= 20:
        raise ValueError("max_steps must exceed the stop hold")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_xy = np.asarray(metadata["target_xy_privileged_label_only"], dtype=np.float64)
    obstacle_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obstacle.name}_geom")
        for obstacle in obstacles
    }
    obstacle_ids.discard(-1)
    reports = []
    renderer = mujoco.Renderer(model, height=96, width=160)
    try:
        for state in select_states(states, max_episodes):
            episode_id = int(metadata["episode_id"]) * 1000 + int(state["step"])
            stem = f"episode_{episode_id:07d}"
            path = output_dir / f"{stem}.npz"
            info_path = output_dir / f"{stem}.json"
            if path.exists() and info_path.exists():
                prior = json.loads(info_path.read_text(encoding="utf-8"))
                if prior.get("source_episode_json") != str(source_episode_json):
                    raise FileExistsError(f"Recovery episode ID collision: {stem}")
                reports.append({
                    "episode_id": episode_id,
                    "source_policy_step": int(state["step"]),
                    "accepted": True,
                    "steps": int(prior["steps"]),
                    "target_reached_step": int(prior["target_reached_step"]),
                    "obstacle_contact_steps": int(prior["obstacle_contact_step_count"]),
                    "reused": True,
                })
                continue
            if path.exists() or info_path.exists():
                raise FileExistsError(f"Incomplete recovery episode already exists: {stem}")

            data = mujoco.MjData(model)
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            data.act[:] = state["act"]
            data.ctrl[:] = state["ctrl"]
            data.qacc_warmstart[:] = state["qacc_warmstart"]
            data.time = state["time"]
            mujoco.mj_forward(model, data)
            # Replay the exact execution layer that captured this state. States
            # written before the field existed fall back to the factory default.
            state_backend = state.get("low_level_backend")
            controller = build_low_level_controller(model, state_backend)
            restore_controller_state(
                controller, state["controller"], backend=state_backend, step=state.get("step"),
            )
            planner = build_recovery_planner(model, target_xy, obstacles, metadata, success_radius)

            frames: dict[str, list[np.ndarray]] = {
                "front_rgb": [], "rear_rgb": [], "lidar": [], "proprio": [], "action": [],
            }
            min_height = float(data.qpos[2])
            max_roll = 0.0
            max_pitch = 0.0
            min_clearance = float("inf")
            contact_steps = 0
            reached_step = -1
            stop_frames = 0
            for step in range(max_steps):
                distance = float(np.linalg.norm(data.qpos[:2] - target_xy))
                if distance <= success_radius:
                    if reached_step < 0:
                        reached_step = step
                    command = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                    stop_frames += 1
                else:
                    command, _ = planner.recommend(data, controller)
                    stop_frames = 0

                renderer.update_scene(data, camera="front_rgb")
                frames["front_rgb"].append(renderer.render().copy())
                renderer.update_scene(data, camera="rear_rgb")
                frames["rear_rgb"].append(renderer.render().copy())
                frames["lidar"].append(planar_lidar(model, data).astype(np.float32))
                frames["proprio"].append(proprioception(model, data).astype(np.float32))
                frames["action"].append(np.asarray(command, dtype=np.float32))
                controller.step(data, command)
                min_height = min(min_height, float(data.qpos[2]))
                w, x, y, z = (float(value) for value in data.qpos[3:7])
                roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
                max_roll = max(max_roll, abs(math.degrees(roll)))
                max_pitch = max(max_pitch, abs(math.degrees(pitch)))
                if obstacles:
                    min_clearance = min(
                        min_clearance,
                        *(obstacle_clearance_xy(data.qpos[:2], obstacle) for obstacle in obstacles),
                    )
                if any(
                    int(data.contact[index].geom1) in obstacle_ids
                    or int(data.contact[index].geom2) in obstacle_ids
                    for index in range(data.ncon)
                ):
                    contact_steps += 1
                if stop_frames >= 20:
                    break

            accepted = bool(
                reached_step >= 0 and stop_frames >= 20 and min_height >= 0.45
                and max(max_roll, max_pitch) <= 8.0 and contact_steps == 0
                and all(np.isfinite(np.stack(values)).all() for values in frames.values())
            )
            w, x, y, z = (float(value) for value in state["qpos"][3:7])
            initial_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            quality_gates = {
                "reached_and_stopped": reached_step >= 0 and stop_frames >= 20,
                "min_base_height": min_height >= 0.45,
                "max_abs_roll_or_pitch": max(max_roll, max_pitch) <= 8.0,
                "zero_obstacle_contact": contact_steps == 0,
            }
            report = {
                **{key: metadata[key] for key in (
                    "schema", "layout_id", "terrain_profile", "task_text", "target_label",
                    "target_xy_privileged_label_only", "objects", "obstacles", "scene_light",
                    "search_start_variant", "search_start_mode", "scene_kind", "scene_name", "scene_episode",
                    "task_language", "task_template_id",
                ) if key in metadata},
                "episode_id": episode_id,
                "collection_mode": "failure_recovery",
                "source_search_policy_effective": metadata.get("search_policy_effective"),
                "search_policy_effective": "waypoint" if planner.global_planner is not None or obstacles else "active_mpc",
                "recovery_route_planner": "global_astar" if planner.global_planner is not None else "legacy",
                "source_episode_json": str(source_episode_json),
                "source_checkpoint": str(checkpoint),
                "source_policy_step": int(state["step"]),
                "source_target_distance_m": float(state["target_distance"]),
                "source_obstacle_clearance_m": (
                    float(state["obstacle_clearance"])
                    if math.isfinite(state["obstacle_clearance"]) else None
                ),
                "initial_xy": state["qpos"][:2].astype(float).tolist(),
                "initial_yaw": initial_yaw,
                "final_xy": data.qpos[:2].astype(float).tolist(),
                "episode_displacement_m": float(np.linalg.norm(data.qpos[:2] - state["qpos"][:2])),
                "steps": len(frames["action"]),
                "success": accepted,
                "target_reached_step": reached_step,
                "min_base_height": min_height,
                "max_abs_roll_deg": max_roll,
                "max_abs_pitch_deg": max_pitch,
                "max_abs_roll_or_pitch_deg": max(max_roll, max_pitch),
                "min_obstacle_clearance": min_clearance if math.isfinite(min_clearance) else None,
                "obstacle_contact_step_count": contact_steps,
                "quality_passed": accepted,
                "quality_gates": quality_gates,
                "success_radius": success_radius,
                "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "language"],
                "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
                "video": "",
            }
            if accepted:
                np.savez_compressed(path, **{key: np.stack(value) for key, value in frames.items()})
                info_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                rejected_path = output_dir / f"rejected_{stem}.json"
                rejected_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            reports.append({
                "episode_id": episode_id,
                "source_policy_step": int(state["step"]),
                "accepted": accepted,
                "steps": len(frames["action"]),
                "target_reached_step": reached_step,
                "obstacle_contact_steps": contact_steps,
            })
    finally:
        renderer.close()
    return reports

#!/usr/bin/env python3
"""Evaluate learner-only hidden target search and obstacle avoidance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m20pro_vla.data import TARGET_PIXEL_THRESHOLD
from play_m20_mujoco_vla import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decision-mode", choices=("direct", "trajectory_scorer", "belief_guided", "search_mpc", "phase_routed"), default="direct")
    parser.add_argument("--trajectory-scorer", type=Path, default=None)
    parser.add_argument(
        "--trajectory-scorer-mode",
        choices=("selector", "guarded_selector", "gate"),
        default="selector",
        help="How the learned scorer is used when provided.",
    )
    parser.add_argument("--trajectory-scorer-interval", type=int, default=1)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--candidate-expert-success",
        choices=("required", "any", "failure"),
        default="required",
        help="Filter held-out hidden-search candidates by whether the source expert episode succeeded.",
    )
    parser.add_argument("--clearance-threshold", type=float, default=0.10)
    parser.add_argument("--stop-threshold", type=float, default=0.50)
    parser.add_argument("--stop-visible-threshold", type=float, default=0.50)
    parser.add_argument("--stop-reach-threshold", type=float, default=0.50)
    parser.add_argument("--stop-confirm", type=int, default=2)
    parser.add_argument("--phase-stop-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase-stop-threshold", type=float, default=0.55)
    parser.add_argument("--visual-target-pixel-threshold", type=int, default=TARGET_PIXEL_THRESHOLD)
    parser.add_argument(
        "--visual-goal-hint-pixel-threshold",
        type=int,
        default=5,
        help="Low policy-RGB target-pixel threshold used only for visual-goal world-model proposals/ranking.",
    )
    parser.add_argument("--visual-goal-hint-memory-steps", type=int, default=160)
    parser.add_argument("--visual-goal-hint-memory-min-pixels", type=int, default=20)
    parser.add_argument("--visual-stop-pixel-threshold", type=int, default=TARGET_PIXEL_THRESHOLD)
    parser.add_argument(
        "--visual-close-pixel-threshold",
        type=int,
        default=None,
        help="Policy-RGB close evidence threshold; default is resolution-scaled in play_m20_mujoco_vla.py.",
    )
    parser.add_argument("--selector-direct-margin", type=float, default=0.02)
    parser.add_argument("--visual-discovery-stop-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-search-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-overshoot-stop-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-smoothing", type=float, default=0.35)
    parser.add_argument(
        "--success-radius-override",
        type=float,
        default=None,
        help="Metrics-only success radius override passed through to play_m20_mujoco_vla.py.",
    )
    parser.add_argument(
        "--search-mpc-route-planner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the route-planner branch inside search MPC when obstacles exist.",
    )
    parser.add_argument("--policy-width", type=int, default=None)
    parser.add_argument("--policy-height", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--demo-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-camera-smoothing", type=float, default=0.08)
    parser.add_argument(
        "--write-policy-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write the exact front/rear RGB streams fed to the policy for visual debugging.",
    )
    parser.add_argument("--progress-interval", type=int, default=0)
    return parser.parse_args()


def hidden_search_passed(report: dict, clearance_threshold: float) -> bool:
    """Strict task gate for "hidden target -> search -> avoid obstacle -> stop"."""
    clearance = report.get("min_obstacle_clearance")
    clearance_ok = clearance is not None and float(clearance) >= float(clearance_threshold)
    return bool(
        report.get("search_required", False)
        and not bool(report.get("initial_target_visible", True))
        and bool(report.get("target_discovered", False))
        and bool(report.get("target_discovered_before_reach", False))
        and not bool(report.get("false_stop_before_discovery", False))
        and clearance_ok
        and bool(report.get("success", False))
    )


def summarize_reports(reports: list[dict], clearance_threshold: float) -> dict:
    judged = []
    for report in reports:
        passed = hidden_search_passed(report, clearance_threshold)
        judged.append({**report, "hidden_search_obstacle_avoidance_passed": passed})
    total = len(judged)
    successes = [item for item in judged if item.get("success")]
    discovered = [item for item in judged if item.get("target_discovered")]
    strict_passes = [item for item in judged if item.get("hidden_search_obstacle_avoidance_passed")]
    clearances = [
        float(item["min_obstacle_clearance"])
        for item in judged
        if item.get("min_obstacle_clearance") is not None
    ]
    return {
        "schema": "m20pro_mujoco_hidden_search_obstacle_gate_v1",
        "clearance_threshold_m": float(clearance_threshold),
        "policy_input": ["front_rgb", "rear_rgb", "planar_lidar_72", "proprioception_45", "visual_history_16", "language"],
        "prohibited_policy_input": ["target_xy", "object_id", "semantic_mask", "privileged_bearing"],
        "acceptance": (
            "Target must be hidden at start, later discovered from learner-only rollout, "
            "reach/stop near the instructed target, and keep obstacle clearance above threshold."
        ),
        "episodes": judged,
        "episode_count": total,
        "success_count": len(successes),
        "discovery_count": len(discovered),
        "strict_pass_count": len(strict_passes),
        "success_rate": float(len(successes) / total) if total else 0.0,
        "discovery_rate": float(len(discovered) / total) if total else 0.0,
        "strict_pass_rate": float(len(strict_passes) / total) if total else 0.0,
        "min_obstacle_clearance_mean_m": mean(clearances) if clearances else None,
        "min_obstacle_clearance_min_m": min(clearances) if clearances else None,
        "passed": bool(total) and len(strict_passes) == total,
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    episode_paths = [Path(value) for value in config.get("val_episodes", [])]
    episode_json_paths = [
        path if path.suffix == ".json" else path.with_suffix(".json")
        for path in episode_paths
    ]
    all_candidates = []
    for path in episode_json_paths:
        if not path.is_file():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("collection_mode") == "search"
            and bool(item.get("search_required", False))
            and not bool(item.get("initial_target_visible", True))
            and int(item.get("search_obstacle_count", len(item.get("obstacles", [])))) > 0
        ):
            all_candidates.append((path, item))
    if args.candidate_expert_success == "required":
        candidates = [(path, item) for path, item in all_candidates if bool(item.get("success", False))]
    elif args.candidate_expert_success == "failure":
        candidates = [(path, item) for path, item in all_candidates if not bool(item.get("success", False))]
    else:
        candidates = list(all_candidates)
    if args.max_episodes is not None:
        candidates = candidates[: int(args.max_episodes)]
    if not candidates:
        success_count = sum(1 for _, item in all_candidates if bool(item.get("success", False)))
        failure_count = len(all_candidates) - success_count
        raise RuntimeError(
            "Checkpoint contains no held-out hidden-search episodes after candidate filter "
            f"{args.candidate_expert_success!r}; all={len(all_candidates)} success={success_count} failure={failure_count}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (episode_path, item) in enumerate(candidates):
        stem = f"hidden_search_{index:03d}_layout_{int(item.get('layout_id', -1)):03d}_{item['target_label'].replace(' ', '_')}"
        policy_front_output = args.output_dir / f"{stem}.policy_front_rgb.mp4" if args.write_policy_videos else None
        policy_rear_output = args.output_dir / f"{stem}.policy_rear_rgb.mp4" if args.write_policy_videos else None
        report = run_episode(
            argparse.Namespace(
                checkpoint=args.checkpoint,
                episode_json=episode_path,
                output=args.output_dir / f"{stem}.mp4",
                metrics=args.output_dir / f"{stem}.json",
                device=args.device,
                decision_mode=args.decision_mode,
                trajectory_scorer=args.trajectory_scorer,
                trajectory_scorer_mode=args.trajectory_scorer_mode,
                trajectory_scorer_interval=args.trajectory_scorer_interval,
                stop_threshold=args.stop_threshold,
                stop_visible_threshold=args.stop_visible_threshold,
                stop_reach_threshold=args.stop_reach_threshold,
                stop_confirm=args.stop_confirm,
                phase_stop_gate=args.phase_stop_gate,
                phase_stop_threshold=args.phase_stop_threshold,
                visual_target_pixel_threshold=args.visual_target_pixel_threshold,
                visual_goal_hint_pixel_threshold=args.visual_goal_hint_pixel_threshold,
                visual_goal_hint_memory_steps=args.visual_goal_hint_memory_steps,
                visual_goal_hint_memory_min_pixels=args.visual_goal_hint_memory_min_pixels,
                visual_stop_pixel_threshold=args.visual_stop_pixel_threshold,
                visual_close_pixel_threshold=args.visual_close_pixel_threshold,
                selector_direct_margin=args.selector_direct_margin,
                visual_discovery_stop_gate=args.visual_discovery_stop_gate,
                visual_search_guard=args.visual_search_guard,
                visual_overshoot_stop_guard=args.visual_overshoot_stop_guard,
                motion_smoothing=args.motion_smoothing,
                success_radius_override=args.success_radius_override,
                search_mpc_route_planner=args.search_mpc_route_planner,
                steps=args.steps,
                policy_width=args.policy_width,
                policy_height=args.policy_height,
                width=args.width,
                height=args.height,
                demo_video=args.demo_video,
                demo_camera_smoothing=args.demo_camera_smoothing,
                policy_front_output=policy_front_output,
                policy_rear_output=policy_rear_output,
                progress_interval=args.progress_interval,
            )
        )
        reports.append(report)
    summary = summarize_reports(reports, args.clearance_threshold)
    summary["candidate_expert_success_filter"] = args.candidate_expert_success
    summary["heldout_hidden_candidate_count_before_filter"] = len(all_candidates)
    summary["heldout_hidden_expert_success_count"] = sum(1 for _, item in all_candidates if bool(item.get("success", False)))
    summary["heldout_hidden_expert_failure_count"] = sum(1 for _, item in all_candidates if not bool(item.get("success", False)))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

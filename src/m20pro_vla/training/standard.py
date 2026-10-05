"""Build reproducible WSL2 training plans and evaluate promotion artifacts."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from m20pro_vla.data import M20MuJoCoDistributionThresholds, audit_m20_mujoco_dataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_STANDARD = WORKSPACE_ROOT / "configs/experiment.json"


def load_training_standard(path: Path = DEFAULT_TRAINING_STANDARD) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "m20pro_vla_experiment_v1":
        raise ValueError(f"Unsupported training standard: {payload.get('schema')!r}")
    return payload


def _environment_report() -> dict[str, Any]:
    release = platform.release()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
    except ImportError:
        cuda_available = False
        cuda_device = None
        torch_version = None
    return {
        "platform_release": release,
        "wsl2": "microsoft-standard-WSL2" in release,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }


def _distribution_thresholds(values: dict[str, Any]) -> M20MuJoCoDistributionThresholds:
    return M20MuJoCoDistributionThresholds(
        min_episodes=int(values["min_episodes"]),
        min_layouts=int(values["min_layouts"]),
        min_terrain_profiles=int(values["min_terrain_profiles"]),
        min_languages=int(values["min_languages"]),
        min_search_anchor_episodes=int(values["min_search_anchor_episodes"]),
        min_search_varied_episodes=int(values["min_search_varied_episodes"]),
        # Optional: absent means 0, which keeps the legacy anchor/varied gate and
        # leaves the structured-scene clause disabled.
        min_search_structured_episodes=int(values.get("min_search_structured_episodes", 0)),
        min_search_successes=int(values["min_search_successes"]),
        min_target_discovered=int(values["min_target_discovered"]),
        min_search_clearance_m=float(values["min_search_clearance_m"]),
        min_search_displacement_m=float(values["min_search_displacement_m"]),
        min_base_height_m=float(values["min_base_height_m"]),
        max_abs_roll_deg=float(values["max_abs_roll_deg"]),
        max_abs_pitch_deg=float(values["max_abs_pitch_deg"]),
    )


def build_standard_training_plan(
    *,
    dataset: Path,
    output_dir: Path,
    standard_path: Path = DEFAULT_TRAINING_STANDARD,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    standard = load_training_standard(standard_path)
    environment = _environment_report()
    audit = audit_m20_mujoco_dataset(
        Path(dataset),
        thresholds=_distribution_thresholds(standard["dataset"]),
    )
    environment_gates = {
        "running_in_wsl2": not standard["environment"]["require_wsl2"] or environment["wsl2"],
        "cuda_available": not standard["environment"]["require_cuda"] or environment["cuda_available"],
    }
    init_ok = init_checkpoint is None or Path(init_checkpoint).is_file()
    training = standard["training"]
    forwarded_args = [
        "--dataset", str(dataset),
        "--output-dir", str(output_dir),
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--learning-rate", str(training["learning_rate"]),
        "--val-fraction", str(training["val_fraction"]),
        "--stride", str(training["stride"]),
        "--target-loss-weight", str(training["target_loss_weight"]),
        "--phase-loss-weight", str(training["phase_loss_weight"]),
        "--visual-geometry-loss-weight", str(training["visual_geometry_loss_weight"]),
        "--close-approach-loss-weight", str(training["close_approach_loss_weight"]),
        "--seed", str(training["seed"]),
        "--device", "cuda",
        "--require-distribution-gate",
    ]
    dataset_values = standard["dataset"]
    threshold_flags = {
        "min_episodes": "--min-distribution-episodes",
        "min_layouts": "--min-distribution-layouts",
        "min_terrain_profiles": "--min-distribution-terrain-profiles",
        "min_languages": "--min-distribution-languages",
        "min_search_anchor_episodes": "--min-distribution-search-anchor-episodes",
        "min_search_varied_episodes": "--min-distribution-search-varied-episodes",
        "min_search_successes": "--min-distribution-search-successes",
        "min_target_discovered": "--min-distribution-target-discovered",
        "min_search_clearance_m": "--min-distribution-search-clearance-m",
        "min_search_displacement_m": "--min-distribution-search-displacement-m",
    }
    for key, flag in threshold_flags.items():
        forwarded_args.extend((flag, str(dataset_values[key])))
    forwarded_args.extend(
        (
            "--min-distribution-search-structured-episodes",
            str(dataset_values.get("min_search_structured_episodes", 0)),
        )
    )
    if init_checkpoint is not None:
        forwarded_args.extend(("--init-checkpoint", str(init_checkpoint)))
    gates = {**environment_gates, "dataset_distribution": bool(audit["passed"]), "init_checkpoint": init_ok}
    return {
        "schema": "m20pro_vla_standard_training_plan_v1",
        "standard": str(Path(standard_path)),
        "stage": standard["stage"],
        "dataset": str(Path(dataset)),
        "output_dir": str(Path(output_dir)),
        "environment": environment,
        "distribution_audit": audit,
        "gates": gates,
        "eligible_to_train": all(gates.values()),
        "forwarded_args": forwarded_args,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_training_candidate(
    *,
    checkpoint: Path,
    eval_summary: Path,
    standard_path: Path = DEFAULT_TRAINING_STANDARD,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    artifact_dir = checkpoint.parent
    standard = load_training_standard(standard_path)
    history = _read_json(artifact_dir / "history.json")
    config = _read_json(artifact_dir / "config.json")
    distribution = _read_json(artifact_dir / "distribution_audit.json")
    closed_loop = _read_json(eval_summary)
    if not history:
        raise ValueError("Training history is empty")
    best = min(history, key=lambda row: float(row["val_loss"]))
    offline = standard["offline_acceptance"]
    val_episodes = config.get("val_episodes", [])
    offline_gates = {
        "layout_disjoint_split": config.get("split_unit") == "layout_id",
        "min_validation_episodes": len(val_episodes) >= int(offline["min_validation_episodes"]),
        "max_val_loss": float(best["val_loss"]) <= float(offline["max_val_loss"]),
        "min_stop_accuracy": float(best["val_stop_accuracy"]) >= float(offline["min_stop_accuracy"]),
        "min_phase_accuracy": float(best["val_phase_accuracy"]) >= float(offline["min_phase_accuracy"]),
        "min_visual_visible_accuracy": float(best["val_visual_visible_accuracy"]) >= float(offline["min_visual_visible_accuracy"]),
        "max_visual_offset_mae_visible": float(best["val_visual_offset_mae_visible"]) <= float(offline["max_visual_offset_mae_visible"]),
        "max_visual_area_mae_visible": float(best["val_visual_area_mae_visible"]) <= float(offline["max_visual_area_mae_visible"]),
        "max_phase_generalization_gap": float(best["train_phase_accuracy"] - best["val_phase_accuracy"]) <= float(offline["max_phase_generalization_gap"]),
        "max_visual_generalization_gap": float(best["train_visual_visible_accuracy"] - best["val_visual_visible_accuracy"]) <= float(offline["max_visual_generalization_gap"]),
    }
    loop = standard["closed_loop_acceptance"]
    episodes = closed_loop.get("episodes", [])
    false_stops = sum(bool(item.get("false_stop_before_discovery", False)) for item in episodes)
    episode_count = int(closed_loop.get("episode_count", len(episodes)))
    false_stop_rate = float(false_stops / episode_count) if episode_count else 1.0
    clearance = closed_loop.get("min_obstacle_clearance_min_m")
    closed_loop_gates = {
        "min_episodes": episode_count >= int(loop["min_episodes"]),
        "min_success_rate": float(closed_loop.get("success_rate", 0.0)) >= float(loop["min_success_rate"]),
        "min_discovery_rate": float(closed_loop.get("discovery_rate", 0.0)) >= float(loop["min_discovery_rate"]),
        "min_strict_pass_rate": float(closed_loop.get("strict_pass_rate", 0.0)) >= float(loop["min_strict_pass_rate"]),
        "min_obstacle_clearance_m": clearance is not None and float(clearance) >= float(loop["min_obstacle_clearance_m"]),
        "max_false_stop_rate": false_stop_rate <= float(loop["max_false_stop_rate"]),
    }
    gates = {
        "checkpoint_exists": checkpoint.is_file(),
        "dataset_distribution": bool(distribution.get("passed", False)),
        "offline": all(offline_gates.values()),
        "closed_loop": all(closed_loop_gates.values()),
    }
    return {
        "schema": "m20pro_vla_training_candidate_gate_v1",
        "checkpoint": str(checkpoint),
        "eval_summary": str(Path(eval_summary)),
        "best_epoch_metrics": best,
        "false_stop_rate": false_stop_rate,
        "offline_gates": offline_gates,
        "closed_loop_gates": closed_loop_gates,
        "gates": gates,
        "eligible_for_shadow": all(gates.values()),
        "promotion": standard["promotion"],
    }

"""Dataset distribution audit helpers for MuJoCo VLA training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections import Counter
from pathlib import Path


@dataclass(frozen=True)
class M20MuJoCoDistributionThresholds:
    min_episodes: int = 4
    min_layouts: int = 2
    min_terrain_profiles: int = 2
    min_languages: int = 2
    min_search_anchor_episodes: int = 1
    min_search_varied_episodes: int = 1
    # Structured scenes (S2/S3) have no anchor/varied start-pose curriculum: the
    # building itself is resampled, so start pose, doorway and target all move
    # together. They declare ``search_start_variant == "structured"`` and are
    # held to this count instead. The default of 0 keeps the legacy gate exactly
    # as it was, because the structured clause only fires when it is non-zero.
    min_search_structured_episodes: int = 0
    min_search_successes: int = 1
    min_target_discovered: int = 1
    min_search_clearance_m: float = 0.10
    min_search_displacement_m: float = 0.50
    min_base_height_m: float = 0.45
    max_abs_roll_deg: float = 8.0
    max_abs_pitch_deg: float = 8.0


def _load_episode_metadata(dataset_root: Path) -> list[dict]:
    # Training consumes only JSON/NPZ pairs. Dataset summaries can become stale
    # after composing symlinked views, so they must not inflate audit coverage.
    episodes: list[dict] = []
    for path in sorted(dataset_root.glob("episode_*.json")):
        if path.is_file() and path.with_suffix(".npz").is_file():
            episodes.append(json.loads(path.read_text(encoding="utf-8")))
    return episodes


def _is_search_episode(item: dict) -> bool:
    return str(item.get("collection_mode", "")) in {"search", "failure_recovery"}


def _attitude_limit(item: dict, axis: str) -> float:
    return float(item.get(f"max_abs_{axis}_deg", item.get("max_abs_roll_or_pitch_deg", float("inf"))))


def audit_m20_mujoco_dataset(
    dataset_root: Path,
    *,
    thresholds: M20MuJoCoDistributionThresholds | None = None,
) -> dict:
    thresholds = thresholds or M20MuJoCoDistributionThresholds()
    dataset_root = Path(dataset_root)
    episodes = _load_episode_metadata(dataset_root)
    npz_count = sum(1 for path in dataset_root.glob("episode_*.npz") if path.is_file())

    collection_modes = Counter(str(item.get("collection_mode", "legacy")) for item in episodes)
    terrain_profiles = Counter(str(item.get("terrain_profile", "flat")) for item in episodes)
    task_languages = Counter(str(item.get("task_language", "")) for item in episodes if item.get("task_language"))
    search_start_modes = Counter(
        str(item.get("search_start_mode", ""))
        for item in episodes
        if _is_search_episode(item) and item.get("search_start_mode")
    )
    search_start_variants = Counter(
        str(item.get("search_start_variant", ""))
        for item in episodes
        if _is_search_episode(item) and item.get("search_start_variant")
    )
    search_policies = Counter(
        str(item.get("search_policy_effective", item.get("search_policy", "")))
        for item in episodes
        if _is_search_episode(item)
    )
    target_labels = Counter(str(item.get("target_label", "")) for item in episodes if item.get("target_label"))
    layout_ids = sorted(
        {
            int(item["layout_id"])
            for item in episodes
            if item.get("layout_id") is not None
        }
    )

    search_episodes = [item for item in episodes if _is_search_episode(item)]
    hidden_search_episodes = [
        item
        for item in search_episodes
        if bool(item.get("search_required", False))
        and not bool(item.get("initial_target_visible", True))
    ]
    search_successes = [item for item in search_episodes if bool(item.get("success", False))]
    search_discoveries = [item for item in search_episodes if bool(item.get("target_discovered", False))]
    search_pre_reach_discoveries = [
        item for item in search_episodes if bool(item.get("target_discovered_before_reach", False))
    ]
    hidden_search_successes = [
        item
        for item in hidden_search_episodes
        if bool(item.get("success", False))
    ]
    clearances = [
        float(item["min_obstacle_clearance"])
        for item in hidden_search_episodes
        if item.get("min_obstacle_clearance") is not None
    ]
    anchor_search_count = int(search_start_variants.get("anchor", 0))
    varied_search_count = int(search_start_variants.get("varied", 0))
    structured_search_count = int(search_start_variants.get("structured", 0))
    stability_records_present = all(
        "min_base_height" in item
        and (
            all(key in item for key in ("max_abs_roll_deg", "max_abs_pitch_deg"))
            or "max_abs_roll_or_pitch_deg" in item
        )
        for item in episodes
    )
    unstable_episodes = [
        int(item.get("episode_id", -1))
        for item in episodes
        if (
            float(item.get("min_base_height", float("-inf"))) < float(thresholds.min_base_height_m)
            or _attitude_limit(item, "roll") > float(thresholds.max_abs_roll_deg)
            or _attitude_limit(item, "pitch") > float(thresholds.max_abs_pitch_deg)
        )
    ]
    low_motion_episode_ids = []
    for item in search_episodes:
        if item.get("collection_mode") == "failure_recovery":
            continue
        start, end = item.get("initial_xy"), item.get("final_xy")
        if not isinstance(start, list) or not isinstance(end, list) or len(start) != 2 or len(end) != 2:
            low_motion_episode_ids.append(int(item.get("episode_id", -1)))
            continue
        displacement = sum((float(a) - float(b)) ** 2 for a, b in zip(start, end)) ** 0.5
        if displacement < thresholds.min_search_displacement_m:
            low_motion_episode_ids.append(int(item.get("episode_id", -1)))

    gates = {
        "min_episode_count": len(episodes) >= int(thresholds.min_episodes),
        "min_layout_count": len(layout_ids) >= int(thresholds.min_layouts),
        "min_terrain_profile_count": len(terrain_profiles) >= int(thresholds.min_terrain_profiles),
        "min_language_count": len(task_languages) >= int(thresholds.min_languages),
        "npz_matches_episode_json": npz_count == len(episodes),
        "search_balanced_anchor_and_varied": (
            not search_episodes
            or (
                anchor_search_count >= int(thresholds.min_search_anchor_episodes)
                and varied_search_count >= int(thresholds.min_search_varied_episodes)
            )
            or (
                int(thresholds.min_search_structured_episodes) > 0
                and structured_search_count >= int(thresholds.min_search_structured_episodes)
            )
        ),
        "search_discovery_present": (
            not search_episodes
            or len(search_discoveries) >= int(thresholds.min_target_discovered)
        ),
        "search_success_present": (
            not search_episodes
            or len(search_successes) >= int(thresholds.min_search_successes)
        ),
        "hidden_search_pre_reach_discovery_present": (
            not hidden_search_episodes or len(search_pre_reach_discoveries) > 0
        ),
        "hidden_search_success_present": (
            not hidden_search_episodes or len(hidden_search_successes) > 0
        ),
        "hidden_search_clearance_threshold": (
            not hidden_search_episodes
            or (min(clearances) >= float(thresholds.min_search_clearance_m))
        ),
        "stability_records_present": stability_records_present,
        "all_episodes_stable": stability_records_present and not unstable_episodes,
        "all_search_episodes_move": not low_motion_episode_ids,
    }

    return {
        "schema": "m20pro_mujoco_vla_distribution_audit_v1",
        "dataset": str(dataset_root),
        "episode_count": len(episodes),
        "npz_count": npz_count,
        "collection_modes": dict(collection_modes),
        "terrain_profiles": dict(terrain_profiles),
        "task_languages": dict(task_languages),
        "search_start_modes": dict(search_start_modes),
        "search_start_variants": dict(search_start_variants),
        "search_policies": dict(search_policies),
        "target_labels": dict(target_labels),
        "layout_ids": layout_ids,
        "layout_count": len(layout_ids),
        "search_episode_count": len(search_episodes),
        "hidden_search_episode_count": len(hidden_search_episodes),
        "search_success_count": len(search_successes),
        "search_discovery_count": len(search_discoveries),
        "search_pre_reach_discovery_count": len(search_pre_reach_discoveries),
        "hidden_search_success_count": len(hidden_search_successes),
        "search_anchor_count": anchor_search_count,
        "search_varied_count": varied_search_count,
        "search_structured_count": structured_search_count,
        "min_obstacle_clearance_m": min(clearances) if clearances else None,
        "mean_obstacle_clearance_m": (sum(clearances) / len(clearances)) if clearances else None,
        "unstable_episode_ids": unstable_episodes,
        "low_motion_episode_ids": low_motion_episode_ids,
        "thresholds": {
            "min_episodes": int(thresholds.min_episodes),
            "min_layouts": int(thresholds.min_layouts),
            "min_terrain_profiles": int(thresholds.min_terrain_profiles),
            "min_languages": int(thresholds.min_languages),
            "min_search_anchor_episodes": int(thresholds.min_search_anchor_episodes),
            "min_search_varied_episodes": int(thresholds.min_search_varied_episodes),
            "min_search_successes": int(thresholds.min_search_successes),
            "min_target_discovered": int(thresholds.min_target_discovered),
            "min_search_clearance_m": float(thresholds.min_search_clearance_m),
            "min_search_displacement_m": float(thresholds.min_search_displacement_m),
            "min_base_height_m": float(thresholds.min_base_height_m),
            "max_abs_roll_deg": float(thresholds.max_abs_roll_deg),
            "max_abs_pitch_deg": float(thresholds.max_abs_pitch_deg),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }

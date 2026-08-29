#!/usr/bin/env python3

"""Convert audited M20 ObjectNav episodes to LeRobot.

The SmolVLA base checkpoint accepts a 32-dimensional state.  v2 uses eight
body-frame proprioceptive values and compresses the 72-beam indoor LiDAR into
24 sector minimums; absolute world position is deliberately excluded.  Target
poses are used only to repair legacy expert stop supervision during conversion
and are never included in policy observations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = Path(
    os.environ.get(
        "M20PRO_VLA_DATA_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)
PROPRIO_DIM = 8
LIDAR_SECTORS = 24
EXPECTED_V5_SCENARIO_IDS = {
    "train_0000", "train_0001", "train_0002", "train_0004", "train_0007",
    "train_0009", "train_0010", "train_0011", "train_0013", "train_0014",
    "train_0015", "train_0016", "train_0027", "train_0036", "train_0040",
    "train_0054", "train_0056", "train_0065", "train_0066", "train_0067",
    "train_0068", "train_0069", "train_0070", "train_0071", "train_0073",
    "train_0076", "train_0077", "train_0080", "train_0090",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ROOT / "datasets/m20_visible_objectnav_v4_stop08_source/train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT / "datasets/m20_visible_objectnav_lerobot_v5_camera12_stop08",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_ROOT / "logs/m20_smolvla_data_audit_v4_stop08_source.json",
    )
    parser.add_argument(
        "--repo-id", default="m20pro_visible_objectnav_v5_camera12_stop08"
    )
    parser.add_argument(
        "--task-type-filter",
        choices=("visible_object_navigation", "hidden_object_search"),
        default="visible_object_navigation",
        help="Select whether to convert visible navigation or hidden-search episodes.",
    )
    parser.add_argument(
        "--candidate-split",
        default="train",
        help="Audit split to convert. Visible defaults to train; hidden-search uses hidden_train.",
    )
    parser.add_argument(
        "--require-audit-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the audit readiness gate before conversion.",
    )
    parser.add_argument(
        "--require-exact-scenario-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the audit candidate set to match the canonical visible source ids.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--stop-label-radius",
        type=float,
        default=0.8,
        help="Canonical success radius used to derive latched stop labels.",
    )
    parser.add_argument(
        "--stop-tail-frames",
        type=int,
        default=20,
        help="Keep this many frames from the first stop label onward.",
    )
    parser.add_argument("--stop-visibility-epsilon", type=float, default=1.0e-4)
    parser.add_argument(
        "--collection-lock",
        type=Path,
        default=DEFAULT_ROOT / "locks/m20_visible_objectnav_v4_stop08_source.lock",
        help="Collection lock shared with the source batch recorder.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Source collection is still running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def lidar_sector_minimums(lidar: np.ndarray) -> np.ndarray:
    """Return 24 finite minimum ranges from a 72-beam scan."""
    lidar = np.asarray(lidar, dtype=np.float32)
    if lidar.ndim != 1 or lidar.size % LIDAR_SECTORS != 0:
        raise ValueError(f"Expected a flat 72-beam LiDAR scan, got {lidar.shape}")
    beams_per_sector = lidar.size // LIDAR_SECTORS
    clean = np.nan_to_num(lidar, nan=20.0, posinf=20.0, neginf=0.0)
    return clean.reshape(LIDAR_SECTORS, beams_per_sector).min(axis=1)


def make_state(state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
    proprio = np.asarray(state, dtype=np.float32)
    if proprio.shape != (PROPRIO_DIM,):
        raise ValueError(f"Expected invariant proprio shape {(PROPRIO_DIM,)}, got {proprio.shape}")
    compressed_lidar = lidar_sector_minimums(lidar)
    result = np.concatenate((proprio, compressed_lidar)).astype(np.float32, copy=False)
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid converted state: shape={result.shape}")
    return result


def retained_frame_count(action: np.ndarray, stop_tail_frames: int) -> int:
    """Trim long post-arrival tails while retaining a stable stop window."""
    if stop_tail_frames < 0:
        raise ValueError(f"stop_tail_frames must be non-negative, got {stop_tail_frames}")
    stop_indices = np.flatnonzero(np.asarray(action)[:, 3] > 0.5)
    if len(stop_indices) == 0:
        return int(action.shape[0])
    return min(int(action.shape[0]), int(stop_indices[0]) + stop_tail_frames)


def canonicalize_stop_labels(
    action: np.ndarray,
    state: np.ndarray,
    target_xy: np.ndarray,
    success_radius: float,
) -> tuple[np.ndarray, dict]:
    """Replace legacy pretrigger labels with a latched in-radius stop label."""
    action = np.asarray(action, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 6:
        raise ValueError(f"Expected six-dimensional high-level actions, got {action.shape}")
    if state.ndim != 2 or state.shape[0] != action.shape[0] or state.shape[1] < 2:
        raise ValueError(f"State/action length or shape mismatch: {state.shape}, {action.shape}")
    if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
        raise ValueError(f"Expected finite target_xy shape (2,), got {target_xy}")
    if success_radius <= 0.0:
        raise ValueError(f"success_radius must be positive, got {success_radius}")

    distances = np.linalg.norm(state[:, :2] - target_xy, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Non-finite target distance while canonicalizing stop labels")
    source_stop = action[:, 3] > 0.5
    inside = distances <= success_radius
    canonical_stop = np.maximum.accumulate(inside)
    canonical = action.copy()
    canonical[:, 3] = canonical_stop.astype(np.float32)

    source_indices = np.flatnonzero(source_stop)
    inside_indices = np.flatnonzero(inside)
    canonical_indices = np.flatnonzero(canonical_stop)
    first_inside = int(inside_indices[0]) if len(inside_indices) else -1
    first_source_stop = int(source_indices[0]) if len(source_indices) else -1
    first_canonical_stop = int(canonical_indices[0]) if len(canonical_indices) else -1
    source_early_stop = source_stop & ~canonical_stop
    stop_motion_conflicts = canonical_stop & (
        np.linalg.norm(canonical[:, :3], axis=1) > 1.0e-4
    )
    diagnostics = {
        "source_first_stop_frame": first_source_stop,
        "source_first_stop_distance_m": (
            float(distances[first_source_stop]) if first_source_stop >= 0 else None
        ),
        "pre_action_first_inside_frame": first_inside,
        "canonical_first_stop_frame": first_canonical_stop,
        "canonical_first_stop_distance_m": (
            float(distances[first_canonical_stop])
            if first_canonical_stop >= 0
            else None
        ),
        "source_stop_frames": int(source_stop.sum()),
        "canonical_stop_frames": int(canonical_stop.sum()),
        "source_early_stop_frames": int(source_early_stop.sum()),
        "stop_motion_conflict_frames": int(stop_motion_conflicts.sum()),
        "relabelled_stop_frames": int(np.count_nonzero(source_stop != canonical_stop)),
    }
    return canonical, diagnostics


def eligible_paths(
    input_root: Path,
    audit_path: Path,
    *,
    candidate_split: str,
    task_type_filter: str,
    require_audit_ready: bool,
    require_exact_scenario_set: bool,
) -> tuple[list[dict], str]:
    audit_bytes = audit_path.read_bytes()
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    audit = json.loads(audit_bytes.decode("utf-8"))
    audit_root_value = audit.get("input_root")
    if not audit_root_value:
        raise ValueError(f"Audit does not declare input_root: {audit_path}")
    audit_root = Path(audit_root_value).resolve()
    requested_input_root = input_root.resolve()
    train_root = (
        audit_root if audit_root.name == "train" else (audit_root / "train").resolve()
    )
    if requested_input_root in {audit_root, train_root}:
        input_root = train_root
    else:
        raise ValueError(
            f"Audit/input root mismatch: audit={audit_root}, input={requested_input_root}"
        )

    if require_audit_ready:
        if task_type_filter == "hidden_object_search":
            required_gates = (
                "minimum_train_scenes_8",
                "minimum_object_categories_12",
                "minimum_instruction_templates_24",
                "all_hidden_candidates_valid_and_timestamp_aligned",
                "search_required_and_occluded_at_start",
                "scene_geometry_visible_to_lidar",
                "six_dimensional_action_labels_present",
                "hidden_search_success_present",
            )
            gate_source = audit.get("hidden_object_search_gates", {})
            ready_key = "ready_for_hidden_object_search_finetune"
        else:
            required_gates = (
                "minimum_train_scenes_8",
                "minimum_object_categories_12",
                "minimum_instruction_templates_24",
                "all_candidates_valid_and_timestamp_aligned",
                "scene_geometry_visible_to_lidar",
                "six_dimensional_action_labels_present",
            )
            gate_source = audit.get("visible_objectnav_gates", {})
            ready_key = "ready_for_visible_objectnav_finetune"
        failed_gates = [
            name for name in required_gates if gate_source.get(name) is not True
        ]
        if audit.get(ready_key) is not True or failed_gates:
            raise RuntimeError(
                "Source audit is not ready for ObjectNav fine-tuning: "
                + ", ".join(failed_gates)
            )

    candidates = [
        item
        for item in audit.get("episodes", [])
        if item.get("smolvla_candidate")
        and item.get("split") == candidate_split
        and item.get("task_type") == task_type_filter
    ]
    rejected = [
        item["path"]
        for item in candidates
        if not (
            item.get("smolvla_eligible")
            or item.get("smolvla_stop_migration_eligible")
        )
    ]
    if rejected:
        raise RuntimeError(
            "Audit contains ineligible training candidates: " + ", ".join(rejected)
        )
    scenario_ids = [item.get("scenario_episode_id") for item in candidates]
    missing_scenario_ids = [
        item["path"]
        for item, scenario_id in zip(candidates, scenario_ids)
        if not scenario_id
    ]
    if missing_scenario_ids:
        raise ValueError(
            "Audit candidates are missing scenario episode ids: "
            + ", ".join(missing_scenario_ids)
        )
    duplicate_scenario_ids = sorted(
        {
            scenario_id
            for scenario_id in scenario_ids
            if scenario_ids.count(scenario_id) > 1
        }
    )
    if duplicate_scenario_ids:
        raise ValueError(
            "Audit contains duplicate scenario episode ids: "
            + ", ".join(duplicate_scenario_ids)
        )
    actual_scenario_ids = set(scenario_ids)
    if require_exact_scenario_set and task_type_filter == "visible_object_navigation":
        if actual_scenario_ids != EXPECTED_V5_SCENARIO_IDS:
            missing = sorted(EXPECTED_V5_SCENARIO_IDS - actual_scenario_ids)
            unexpected = sorted(actual_scenario_ids - EXPECTED_V5_SCENARIO_IDS)
            raise ValueError(
                "Audit scenario set does not match the v5 source set: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif not actual_scenario_ids:
        raise RuntimeError("No audited SmolVLA-eligible episodes found")

    entries = []
    for item in sorted(candidates, key=lambda value: value["path"]):
        path = audit_root / item["path"]
        if path.parent.resolve() != input_root:
            raise ValueError(f"Audited episode is outside input root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Audited episode is missing: {path}")
        expected_sha256 = item.get("sha256")
        if not expected_sha256:
            raise ValueError(f"Audit is missing source SHA-256: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Audited episode changed: {path} ({actual_sha256} != {expected_sha256})"
            )
        entries.append(
            {
                "path": path,
                "sha256": expected_sha256,
                "scenario_episode_id": item["scenario_episode_id"],
            }
        )

    if not entries:
        raise RuntimeError("No audited SmolVLA-eligible episodes found")
    discovered = {path.absolute() for path in input_root.glob("*.h5")}
    audited = {entry["path"].absolute() for entry in entries}
    unaudited = sorted(discovered - audited)
    if unaudited:
        raise RuntimeError(
            "Input root contains unaudited HDF5 episodes: "
            + ", ".join(str(path) for path in unaudited)
        )
    return entries, audit_sha256


def episode_conversion_plan(h5: h5py.File, args: argparse.Namespace) -> dict:
    attrs = {key: _json_value(value) for key, value in h5.attrs.items()}
    try:
        source_success_radius = float(attrs["success_radius"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Episode is missing a finite success_radius") from exc
    if not np.isfinite(source_success_radius) or not np.isclose(
        source_success_radius,
        args.stop_label_radius,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Episode/audit conversion radius mismatch: "
            f"{source_success_radius} != {args.stop_label_radius}"
        )
    source_action = np.asarray(h5["high_level_action"], dtype=np.float32)
    source_state = np.asarray(h5["observation/state"], dtype=np.float32)
    action, stop_diagnostics = canonicalize_stop_labels(
        source_action,
        source_state,
        attrs.get("target_xy"),
        args.stop_label_radius,
    )
    frame_count = retained_frame_count(action, args.stop_tail_frames)
    first_inside = stop_diagnostics["pre_action_first_inside_frame"]
    successful_expert = bool(attrs.get("success", False))
    if successful_expert and first_inside < 0:
        raise ValueError("Successful expert never enters the success radius")
    if successful_expert and stop_diagnostics["canonical_first_stop_frame"] < 0:
        raise ValueError("Successful expert has no canonical stop label")
    if stop_diagnostics["stop_motion_conflict_frames"]:
        raise ValueError(
            "Canonical stop overlaps non-zero motion in "
            f"{stop_diagnostics['stop_motion_conflict_frames']} frames"
        )
    if first_inside >= 0 and frame_count <= first_inside:
        raise ValueError("Conversion trims before target reach")

    front_fraction = np.asarray(
        h5["observation/front_target_pixel_fraction"], dtype=np.float32
    )
    rear_fraction = np.asarray(
        h5["observation/rear_target_pixel_fraction"], dtype=np.float32
    )
    if front_fraction.shape != (len(action),) or rear_fraction.shape != (len(action),):
        raise ValueError("Target visibility arrays do not match action length")
    visibility = np.maximum(front_fraction, rear_fraction)
    retained_stop = action[:frame_count, 3] > 0.5
    retained_invisible_stop = retained_stop & (
        visibility[:frame_count] <= args.stop_visibility_epsilon
    )
    retained_stop_frames = int(retained_stop.sum())
    retained_invisible_stop_frames = int(retained_invisible_stop.sum())
    if successful_expert and retained_stop_frames == 0:
        raise ValueError("Successful expert retains no stop frames")
    if successful_expert and retained_stop_frames != args.stop_tail_frames:
        raise ValueError(
            "Successful expert does not retain a complete stop tail: "
            f"{retained_stop_frames}/{args.stop_tail_frames}"
        )
    if retained_invisible_stop_frames:
        raise ValueError(
            f"Canonical stop is visually unobservable in "
            f"{retained_invisible_stop_frames}/{retained_stop_frames} retained frames"
        )

    return {
        "attrs": attrs,
        "action": action,
        "frame_count": frame_count,
        "first_inside": first_inside,
        "stop_diagnostics": stop_diagnostics,
        "retained_stop_frames": retained_stop_frames,
        "retained_invisible_stop_frames": retained_invisible_stop_frames,
        "minimum_retained_stop_visibility": (
            float(visibility[:frame_count][retained_stop].min())
            if retained_stop_frames
            else None
        ),
        "camera_focal_length_mm": float(attrs.get("camera_focal_length", -1.0)),
        "camera_pitch_deg": float(attrs.get("camera_pitch_deg", -1000.0)),
    }


def convert(args: argparse.Namespace) -> dict:
    if args.stop_label_radius <= 0.0:
        raise ValueError("--stop-label-radius must be positive")
    if args.stop_tail_frames <= 0:
        raise ValueError("--stop-tail-frames must be positive")
    if args.stop_visibility_epsilon < 0.0:
        raise ValueError("--stop-visibility-epsilon must be non-negative")
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    if args.output_root.exists() and not args.force:
        raise FileExistsError(
            f"Output exists; pass --force to replace: {args.output_root}"
        )

    audited_entries, source_audit_sha256 = eligible_paths(
        args.input_root,
        args.audit,
        candidate_split=args.candidate_split,
        task_type_filter=args.task_type_filter,
        require_audit_ready=args.require_audit_ready,
        require_exact_scenario_set=args.require_exact_scenario_set,
    )
    entries = audited_entries
    if args.max_episodes > 0:
        entries = entries[: args.max_episodes]

    front_shape = None
    camera_configs = set()
    for entry in entries:
        path = entry["path"]
        try:
            with h5py.File(path, "r") as h5:
                plan = episode_conversion_plan(h5, args)
                episode_front_shape = tuple(
                    h5["observation/front_rgb"].shape[1:]
                )
                rear_shape = tuple(h5["observation/rear_rgb"].shape[1:])
                if episode_front_shape != rear_shape:
                    raise ValueError(
                        f"Front/rear camera shape mismatch: {episode_front_shape}, {rear_shape}"
                    )
                if front_shape is None:
                    front_shape = episode_front_shape
                elif episode_front_shape != front_shape:
                    raise ValueError(
                        f"Camera shape differs across episodes: {episode_front_shape}, {front_shape}"
                    )
                camera_configs.add(
                    (
                        plan["camera_focal_length_mm"],
                        plan["camera_pitch_deg"],
                    )
                )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"Preflight failed for {path}: {exc}") from exc
    if len(camera_configs) != 1:
        raise ValueError(f"Mixed camera intrinsics/extrinsics: {camera_configs}")
    camera_focal_length, _ = next(iter(camera_configs))
    if not np.isfinite(camera_focal_length) or not np.isclose(
        camera_focal_length, 12.0, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            f"Expected unified 12 mm cameras, got {camera_focal_length} mm"
        )
    if front_shape is None:
        raise RuntimeError("No episodes selected after preflight")
    if front_shape != rear_shape or len(front_shape) != 3 or front_shape[-1] != 3:
        raise ValueError(f"Unexpected camera shapes: {front_shape}, {rear_shape}")

    build_root = args.output_root.with_name(
        f".{args.output_root.name}.part-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    if build_root.exists():
        raise FileExistsError(f"Temporary conversion path already exists: {build_root}")

    action_names = ["forward_mps", "lateral_mps", "yaw_rps", "stop", "search", "parkour"]
    state_names = [f"proprio_{index:02d}" for index in range(PROPRIO_DIM)]
    state_names.extend(f"lidar_sector_{index:02d}" for index in range(LIDAR_SECTORS))
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (32,),
            "names": state_names,
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": front_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": rear_shape,
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": action_names,
        },
    }
    manifest = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=50,
            root=build_root,
            robot_type="m20pro_sim",
            features=features,
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        for episode_index, entry in enumerate(entries):
            path = entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Source changed after preflight: {path}")
            with h5py.File(path, "r") as h5:
                plan = episode_conversion_plan(h5, args)
                attrs = plan["attrs"]
                if attrs.get("scenario_episode_id") != entry["scenario_episode_id"]:
                    raise ValueError(
                        "Scenario episode id changed after audit: "
                        f"{attrs.get('scenario_episode_id')} != "
                        f"{entry['scenario_episode_id']}"
                    )
                front = h5["observation/front_rgb"]
                rear = h5["observation/rear_rgb"]
                lidar = h5["observation/lidar"]
                action = plan["action"]
                stop_diagnostics = plan["stop_diagnostics"]
                smolvla_proprio = h5["observation/smolvla_proprio"]
                frame_count = plan["frame_count"]
                first_inside = plan["first_inside"]
                task = str(attrs["task_text"])
                for frame_index in range(frame_count):
                    frame = {
                        "observation.state": make_state(
                            smolvla_proprio[frame_index], lidar[frame_index]
                        ),
                        "observation.images.camera1": np.asarray(front[frame_index], dtype=np.uint8),
                        "observation.images.camera2": np.asarray(rear[frame_index], dtype=np.uint8),
                        "action": action[frame_index],
                        "task": task,
                    }
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                manifest.append(
                    {
                        "episode_index": episode_index,
                        "source": str(path),
                        "source_sha256": entry["sha256"],
                        "scenario_episode_id": attrs.get("scenario_episode_id"),
                        "scene_id": attrs.get("scene_id"),
                        "object_category": attrs.get("object_category"),
                        "instruction_template_id": attrs.get("instruction_template_id"),
                        "task_text": task,
                        "source_frames": int(action.shape[0]),
                        "frames": frame_count,
                        "success_radius_m": args.stop_label_radius,
                        **stop_diagnostics,
                        "retained_stop_frames": plan["retained_stop_frames"],
                        "retained_invisible_stop_frames": plan[
                            "retained_invisible_stop_frames"
                        ],
                        "minimum_retained_stop_visibility": plan[
                            "minimum_retained_stop_visibility"
                        ],
                        "trimmed_before_target_reach": bool(
                            first_inside >= 0 and frame_count <= first_inside
                        ),
                        "dagger_partial_reached_target": bool(
                            attrs.get("smolvla_dagger_partial", False)
                            and first_inside >= 0
                        ),
                    }
                )
            print(
                f"[M20PRO-CONVERT] {episode_index + 1}/{len(entries)} {path.name}",
                flush=True,
            )
        dataset.finalize()
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    if sha256_file(args.audit) != source_audit_sha256:
        shutil.rmtree(build_root, ignore_errors=True)
        raise RuntimeError("Source audit changed during conversion")
    for entry in audited_entries:
        if sha256_file(entry["path"]) != entry["sha256"]:
            shutil.rmtree(build_root, ignore_errors=True)
            raise RuntimeError(f"Source changed during conversion: {entry['path']}")
    source_root = entries[0]["path"].parent
    discovered_sources = {path.resolve() for path in source_root.glob("*.h5")}
    audited_sources = {entry["path"].resolve() for entry in audited_entries}
    if discovered_sources != audited_sources:
        shutil.rmtree(build_root, ignore_errors=True)
        raise RuntimeError("Source directory changed during conversion")

    coverage = {
        "scenario_episode_ids": sorted(
            str(item["scenario_episode_id"]) for item in manifest
        ),
        "scene_ids": sorted(str(item["scene_id"]) for item in manifest),
        "object_categories": sorted(
            str(item["object_category"]) for item in manifest
        ),
        "instruction_template_ids": sorted(
            str(item["instruction_template_id"]) for item in manifest
        ),
    }
    for key in ("scene_ids", "object_categories", "instruction_template_ids"):
        coverage[key] = sorted(set(coverage[key]))
    coverage.update(
        {
            "unique_scenario_episodes": len(
                set(coverage["scenario_episode_ids"])
            ),
            "unique_scenes": len(coverage["scene_ids"]),
            "unique_object_categories": len(coverage["object_categories"]),
            "unique_instruction_templates": len(
                coverage["instruction_template_ids"]
            ),
        }
    )
    conversion = {
        "schema": (
            "m20pro_hidden_objectnav_lerobot_v1"
            if args.task_type_filter == "hidden_object_search"
            else "m20pro_visible_objectnav_lerobot_v3"
        ),
        "source_audit": str(args.audit),
        "source_audit_sha256": source_audit_sha256,
        "repo_id": args.repo_id,
        "task_type_filter": args.task_type_filter,
        "candidate_split": args.candidate_split,
        "episode_count": len(manifest),
        "frames": sum(item["frames"] for item in manifest),
        "fps": 50,
        "coverage": coverage,
        "stop_tail_frames": args.stop_tail_frames,
        "sensor_config": {
            "camera_focal_length_mm": next(iter(camera_configs))[0],
            "camera_pitch_deg": next(iter(camera_configs))[1],
            "image_shape": list(front_shape),
        },
        "stop_label_policy": {
            "version": "target_radius_latched_v1",
            "success_radius_m": args.stop_label_radius,
            "robot_position_source": "observation/state[:2] (supervision only)",
            "target_position_source": "HDF5 target_xy (supervision only)",
            "inference_uses_privileged_target_pose": False,
        },
        "stop_label_audit": {
            "source_early_stop_frames": sum(
                item["source_early_stop_frames"] for item in manifest
            ),
            "relabelled_stop_frames": sum(
                item["relabelled_stop_frames"] for item in manifest
            ),
            "retained_stop_frames": sum(
                item["retained_stop_frames"] for item in manifest
            ),
            "retained_invisible_stop_frames": sum(
                item["retained_invisible_stop_frames"] for item in manifest
            ),
            "canonical_stop_visibility_valid": all(
                item["retained_invisible_stop_frames"] == 0
                for item in manifest
            ),
            "stop_motion_conflict_frames": sum(
                item["stop_motion_conflict_frames"] for item in manifest
            ),
            "trimmed_before_target_reach_episodes": sum(
                item["trimmed_before_target_reach"] for item in manifest
            ),
        },
        "state": {
            "shape": [32],
            "proprio_indices": list(range(PROPRIO_DIM)),
            "lidar_input": "72 beams compressed to 24 sector minimums",
        },
        "action": {"shape": [6], "names": action_names},
        "camera_keys": ["observation.images.camera1", "observation.images.camera2"],
        "episodes": manifest,
    }
    (build_root / "m20pro_conversion_manifest.json").write_text(
        json.dumps(conversion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    backup_root = None
    try:
        if args.output_root.exists():
            if not args.force:
                raise FileExistsError(
                    f"Output appeared during conversion: {args.output_root}"
                )
            backup_root = args.output_root.with_name(
                f".{args.output_root.name}.backup-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            os.replace(args.output_root, backup_root)
        os.replace(build_root, args.output_root)
    except Exception:
        if backup_root is not None and backup_root.exists() and not args.output_root.exists():
            os.replace(backup_root, args.output_root)
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)
    return conversion


def main() -> None:
    args = parse_args()
    with exclusive_lock(args.collection_lock):
        result = convert(args)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema",
                    "repo_id",
                    "episode_count",
                    "frames",
                    "stop_label_policy",
                    "stop_label_audit",
                    "state",
                    "action",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

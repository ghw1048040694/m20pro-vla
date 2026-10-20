#!/usr/bin/env python3
"""Train the first high-level M20 VLA policy on MuJoCo episodes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m20pro_vla.data import (
    DEFAULT_HISTORY_PIXEL_THRESHOLD,
    HISTORY_FEATURE_LABELS,
    M20MuJoCoDistributionThresholds,
    audit_m20_mujoco_dataset,
    build_visual_history_feature_trace,
    first_visual_target_visible_step,
    target_color_mask,
)
from m20pro_vla.policies import (
    M20MuJoCoVLA,
    PHASE_LABELS,
    TEXT_ENCODING,
    TEXT_TOKEN_LENGTH,
    VISUAL_GEOMETRY_LABELS,
    encode_text,
)
from m20pro_vla.utils import (
    ObservationAugmentationConfig,
    RGBAugmentationConfig,
    apply_observation_augmentation,
)

TARGET_LABEL_TO_ID = {
    "red cube": 0,
    "green cylinder": 1,
    "yellow box": 2,
}
PHASE_LABEL_TO_ID = {label: index for index, label in enumerate(PHASE_LABELS)}


def visual_geometry_label(
    front_rgb: np.ndarray,
    rear_rgb: np.ndarray,
    target_label: str,
    *,
    pixel_threshold: int = 5,
) -> np.ndarray:
    """Return learner-only target geometry labels from policy RGB frames.

    Format is ``[visible, x_offset, area, rear_source]``.  The labels are
    derived only from the same low-resolution front/rear RGB streams that the
    policy receives, not from target XY or simulator object IDs at runtime.
    """
    front_mask = target_color_mask(front_rgb, target_label)
    rear_mask = target_color_mask(rear_rgb, target_label)
    front_count = int(front_mask.sum())
    rear_count = int(rear_mask.sum())
    total = front_count + rear_count
    if total < int(pixel_threshold):
        return np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    use_rear = rear_count > front_count
    mask = rear_mask if use_rear else front_mask
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    x_offset = (float(xs.mean()) - 0.5 * float(width - 1)) / max(1.0, 0.5 * float(width))
    area = min(1.0, float(np.log1p(total) / np.log1p(width * height)))
    return np.asarray(
        (
            1.0,
            float(np.clip(x_offset, -1.0, 1.0)),
            area,
            1.0 if use_rear else 0.0,
        ),
        dtype=np.float32,
    )


def phase_label_for_sample(meta: dict, action: np.ndarray, step: int, *, visual_first_visible_step: int | None = None) -> int:
    """Return explicit hidden-search phase label for one frame.

    ``search`` covers all pre-discovery frames, ``approach`` starts once the
    target is visible/discovered, and ``stop`` starts at the expert stop/hold
    tail.  For non-search episodes we still separate moving vs stop frames.
    """
    stop_action = bool(float(action[3]) > 0.5)
    if stop_action:
        return PHASE_LABEL_TO_ID["stop"]
    if str(meta.get("collection_mode", "")) != "search":
        return PHASE_LABEL_TO_ID["approach"]
    reached_step = int(meta.get("target_reached_step", -1))
    if reached_step >= 0 and step >= reached_step:
        return PHASE_LABEL_TO_ID["stop"]
    first_visible = (
        int(visual_first_visible_step)
        if visual_first_visible_step is not None
        else int(meta.get("target_first_visible_step", -1))
    )
    if first_visible >= 0 and step >= first_visible:
        return PHASE_LABEL_TO_ID["approach"]
    return PHASE_LABEL_TO_ID["search"]

TERRAIN_WEIGHT_MULTIPLIER = {
    "flat": 1.0,
    "slope": 1.35,
    "bumps": 1.50,
    "step": 1.65,
}


def episode_training_weight(meta: dict) -> float:
    terrain_profile = str(meta.get("terrain_profile", "flat"))
    terrain_multiplier = float(TERRAIN_WEIGHT_MULTIPLIER.get(terrain_profile, 1.0))
    search_multiplier = 1.0
    if str(meta.get("collection_mode", "")) == "search":
        if not bool(meta.get("initial_target_visible", True)):
            search_multiplier *= 1.15
        if bool(meta.get("target_discovered_before_reach", False)):
            search_multiplier *= 1.08
        heading_offset = abs(float(meta.get("search_start_yaw_offset_rad", 0.0)))
        if heading_offset > 0.0:
            search_multiplier *= 1.0 + min(0.12, heading_offset / 0.72 * 0.12)
        min_clearance = meta.get("min_obstacle_clearance")
        if min_clearance is not None:
            clearance_bonus = float(np.clip((0.20 - float(min_clearance)) / 0.20, 0.0, 0.28))
            search_multiplier *= 1.0 + clearance_bonus
    if bool(meta.get("success", False)):
        return float(np.clip(terrain_multiplier * search_multiplier, 0.15, 1.35))
    if str(meta.get("collection_mode", "")) == "search":
        min_distance = float(meta.get("min_target_distance", 2.0))
        weight = 0.25 + 0.75 * math.exp(-min_distance / 2.5)
        if not bool(meta.get("target_discovered", False)):
            weight *= 0.75
        weight *= search_multiplier
        return float(np.clip(weight * terrain_multiplier, 0.15, 1.35))
    return 0.0


def sample_training_multiplier(meta: dict, step: int, *, visual_first_visible_step: int | None = None) -> float:
    """Weight hidden-search phases for strict closed-loop training.

    The strict gate exposed a second failure mode after dense success
    fine-tuning: the stop head can trigger before the target has ever become
    visible.  Pre-discovery frames therefore need to remain strong negative
    stop examples, while the final approach window still gets the largest
    positive stop/slowdown weight.
    """
    if str(meta.get("collection_mode", "")) != "search":
        return 1.0
    multiplier = 0.90
    discovered_step = (
        int(visual_first_visible_step)
        if visual_first_visible_step is not None
        else int(meta.get("target_first_visible_step", -1))
    )
    reached_step = int(meta.get("target_reached_step", -1))
    success = bool(meta.get("success", False))
    if discovered_step >= 0:
        if step < discovered_step:
            multiplier *= 1.60
        else:
            multiplier *= 1.25
    else:
        multiplier *= 0.55
    if success and reached_step >= 0:
        close_tail_start = max(0, reached_step - 160)
        if step >= reached_step:
            multiplier *= 3.20
        elif step >= close_tail_start:
            multiplier *= 2.80
        elif discovered_step >= 0 and step >= discovered_step:
            multiplier *= 1.35
    elif success and discovered_step >= 0 and step >= discovered_step:
        multiplier *= 1.20
    if bool(meta.get("target_discovered_before_reach", False)):
        multiplier *= 1.05
    return float(np.clip(multiplier, 0.20, 4.00))


def visual_geometry_loss(
    visual_prediction: torch.Tensor,
    visual_target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample auxiliary loss for RGB target geometry."""
    visible_target = visual_target[:, 0]
    visible_loss = nn.functional.binary_cross_entropy_with_logits(
        visual_prediction[:, 0],
        visible_target,
        reduction="none",
    )
    visible_weight = visible_target
    offset_loss = nn.functional.smooth_l1_loss(
        torch.tanh(visual_prediction[:, 1]),
        visual_target[:, 1],
        reduction="none",
    )
    area_loss = nn.functional.smooth_l1_loss(
        torch.sigmoid(visual_prediction[:, 2]),
        visual_target[:, 2],
        reduction="none",
    )
    source_loss = nn.functional.binary_cross_entropy_with_logits(
        visual_prediction[:, 3],
        visual_target[:, 3],
        reduction="none",
    )
    return visible_loss + visible_weight * (1.50 * offset_loss + area_loss + 0.50 * source_loss)


def close_approach_sample_weight(meta: dict, phase_id: int, history: np.ndarray) -> float:
    """Extra supervision weight for the failure mode found in closed loop.

    This uses only learner-available history features.  It emphasizes the
    segment after the target has appeared in policy RGB, including brief
    target-loss periods after discovery, because the current model discovers
    targets but fails to keep approaching them to the strict radius.
    """
    del meta
    values = np.asarray(history, dtype=np.float32)
    current_visible = bool(values[0] >= 0.5)
    current_area = float(values[2])
    seen_latch = bool(values[4] >= 0.5)
    last_visible_decay = float(values[5])
    recently_lost = float(values[6])
    peak_area = float(values[9])
    if int(phase_id) == PHASE_LABEL_TO_ID["stop"]:
        base = 1.80
    elif int(phase_id) == PHASE_LABEL_TO_ID["approach"]:
        base = 1.20
    elif seen_latch and recently_lost > 0.10:
        base = 0.75
    else:
        base = 0.0
    if current_visible:
        base += 0.45 + 0.80 * min(1.0, current_area / 0.55)
    elif seen_latch:
        base += 0.35 * min(1.0, last_visible_decay)
    if peak_area >= 0.25 and int(phase_id) != PHASE_LABEL_TO_ID["search"]:
        base += 0.35
    return float(np.clip(base, 0.0, 3.50))


def split_paths_by_layout(
    paths: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path], str, list[int]]:
    metadata = {path: json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in paths}
    layout_groups: dict[int, list[Path]] = {}
    for path, meta in metadata.items():
        if meta.get("layout_id") is not None:
            layout_groups.setdefault(int(meta["layout_id"]), []).append(path)
    if layout_groups and sum(len(group) for group in layout_groups.values()) == len(paths) and len(layout_groups) >= 2:
        rng = random.Random(seed)
        layout_ids = sorted(layout_groups)
        rng.shuffle(layout_ids)
        val_layout_count = max(1, int(round(len(layout_ids) * val_fraction)))
        val_layouts = set(layout_ids[:val_layout_count])
        val_paths = [path for layout_id in layout_ids if layout_id in val_layouts for path in sorted(layout_groups[layout_id])]
        train_paths = [path for layout_id in layout_ids if layout_id not in val_layouts for path in sorted(layout_groups[layout_id])]
        return train_paths, val_paths, "layout_id", sorted(val_layouts)
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count], "episode", []


class EpisodeFrames(Dataset):
    def __init__(
        self,
        paths: list[Path],
        stride: int = 2,
        episode_weights: dict[Path, float] | None = None,
        *,
        history_pixel_threshold: int = DEFAULT_HISTORY_PIXEL_THRESHOLD,
    ):
        self.samples = []
        self.sample_weights = []
        self.episodes = []
        for path in paths:
            # Load compressed npz episodes once. Keeping NpzFile handles here makes
            # every sampled frame re-decode RGB from zip storage, which dominates
            # small hidden-search training runs.
            with np.load(path) as raw:
                arrays = {name: raw[name].copy() for name in raw.files}
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            visual_first_visible = first_visual_target_visible_step(
                arrays["front_rgb"],
                arrays["rear_rgb"],
                meta["target_label"],
                pixel_threshold=int(history_pixel_threshold),
            )
            meta = {
                **meta,
                "visual_target_first_visible_step": int(visual_first_visible),
                "geometric_target_first_visible_step": int(meta.get("target_first_visible_step", -1)),
            }
            executed_actions = arrays.get("executed_action")
            arrays["history"] = build_visual_history_feature_trace(
                arrays["front_rgb"],
                arrays["rear_rgb"],
                arrays["action"],
                meta["target_label"],
                executed_actions=executed_actions,
                pixel_threshold=int(history_pixel_threshold),
            )
            self.episodes.append((arrays, meta))
            weight = float((episode_weights or {}).get(path, 1.0))
            step_range = range(0, len(arrays["action"]), stride)
            self.samples.extend((len(self.episodes) - 1, step) for step in step_range)
            self.sample_weights.extend(
                weight * sample_training_multiplier(meta, step, visual_first_visible_step=visual_first_visible)
                for step in step_range
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_id, step = self.samples[index]
        arrays, meta = self.episodes[episode_id]
        rgb = np.concatenate((arrays["front_rgb"][step], arrays["rear_rgb"][step]), axis=-1).transpose(2, 0, 1)
        phase_id = phase_label_for_sample(
            meta,
            arrays["action"][step],
            step,
            visual_first_visible_step=int(meta.get("visual_target_first_visible_step", -1)),
        )
        history = arrays["history"][step]
        return {
            "rgb": torch.from_numpy(rgb.copy()).float().div_(255.0),
            "lidar": torch.from_numpy(arrays["lidar"][step].copy()).float().div_(10.0),
            "proprio": torch.from_numpy(arrays["proprio"][step].copy()).float(),
            "history": torch.from_numpy(history.copy()).float(),
            "language": torch.from_numpy(encode_text(meta["task_text"])),
            "action": torch.from_numpy(arrays["action"][step].copy()).float(),
            "phase_id": torch.tensor(
                phase_id,
                dtype=torch.long,
            ),
            "target_id": torch.tensor(TARGET_LABEL_TO_ID[meta["target_label"]], dtype=torch.long),
            "visual_geometry": torch.from_numpy(
                visual_geometry_label(
                    arrays["front_rgb"][step],
                    arrays["rear_rgb"][step],
                    meta["target_label"],
                )
            ),
            "close_approach_weight": torch.tensor(
                close_approach_sample_weight(meta, phase_id, history),
                dtype=torch.float32,
            ),
            "sample_weight": torch.tensor(self.sample_weights[index], dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(".runtime/datasets/m20_mujoco_vla_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/checkpoints/m20_mujoco_vla_v1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
    parser.add_argument("--phase-loss-weight", type=float, default=0.8)
    parser.add_argument("--visual-geometry-loss-weight", type=float, default=0.75)
    parser.add_argument(
        "--close-approach-loss-weight",
        type=float,
        default=0.80,
        help="Extra action/stop imitation weight on post-discovery close-approach samples.",
    )
    parser.add_argument(
        "--history-pixel-threshold",
        type=int,
        default=DEFAULT_HISTORY_PIXEL_THRESHOLD,
        help="Target-color pixel threshold used to build learner-only visual history features.",
    )
    parser.add_argument(
        "--rgb-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply mild RGB domain randomization during training only.",
    )
    parser.add_argument("--rgb-brightness-jitter", type=float, default=0.08)
    parser.add_argument("--rgb-contrast-jitter", type=float, default=0.08)
    parser.add_argument("--rgb-noise-std", type=float, default=0.01)
    parser.add_argument("--rgb-cutout-prob", type=float, default=0.20)
    parser.add_argument("--rgb-cutout-size", type=int, default=8)
    parser.add_argument("--lidar-noise-std", type=float, default=0.01)
    parser.add_argument("--lidar-dropout-prob", type=float, default=0.05)
    parser.add_argument("--proprio-noise-std", type=float, default=0.005)
    parser.add_argument("--require-distribution-gate", action="store_true")
    parser.add_argument("--min-distribution-episodes", type=int, default=4)
    parser.add_argument("--min-distribution-layouts", type=int, default=2)
    parser.add_argument("--min-distribution-terrain-profiles", type=int, default=2)
    parser.add_argument("--min-distribution-languages", type=int, default=2)
    parser.add_argument("--min-distribution-search-anchor-episodes", type=int, default=1)
    parser.add_argument("--min-distribution-search-varied-episodes", type=int, default=1)
    parser.add_argument(
        "--min-distribution-search-structured-episodes",
        type=int,
        default=0,
        help=(
            "Structured scenes (S2/S3) replace the anchor/varied start-pose curriculum with a "
            "resampled building, so they report search_start_variant='structured'. When this is "
            "positive the anchor/varied gate may be satisfied by that count instead. 0 (default) "
            "disables the clause and leaves the historical gate untouched."
        ),
    )
    parser.add_argument("--min-distribution-search-successes", type=int, default=1)
    parser.add_argument("--min-distribution-target-discovered", type=int, default=1)
    parser.add_argument("--min-distribution-search-clearance-m", type=float, default=0.10)
    parser.add_argument("--min-distribution-search-displacement-m", type=float, default=0.50)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize the VLA from before fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        args.rgb_brightness_jitter < 0.0
        or args.rgb_contrast_jitter < 0.0
        or args.rgb_noise_std < 0.0
        or not 0.0 <= args.rgb_cutout_prob <= 1.0
        or args.rgb_cutout_size <= 0
        or args.lidar_noise_std < 0.0
        or not 0.0 <= args.lidar_dropout_prob <= 1.0
        or args.proprio_noise_std < 0.0
        or args.visual_geometry_loss_weight < 0.0
        or args.close_approach_loss_weight < 0.0
        or args.history_pixel_threshold <= 0
    ):
        raise ValueError("Invalid observation augmentation parameters")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    paths = []
    episode_weights: dict[Path, float] = {}
    for path in sorted(args.dataset.glob("episode_*.npz")):
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        weight = episode_training_weight(meta)
        if weight > 0.0:
            paths.append(path)
            episode_weights[path] = weight
    if len(paths) < 4:
        raise RuntimeError(f"Need at least 4 trainable MuJoCo episodes, found {len(paths)} in {args.dataset}")
    distribution_thresholds = M20MuJoCoDistributionThresholds(
        min_episodes=args.min_distribution_episodes,
        min_layouts=args.min_distribution_layouts,
        min_terrain_profiles=args.min_distribution_terrain_profiles,
        min_languages=args.min_distribution_languages,
        min_search_anchor_episodes=args.min_distribution_search_anchor_episodes,
        min_search_varied_episodes=args.min_distribution_search_varied_episodes,
        min_search_structured_episodes=args.min_distribution_search_structured_episodes,
        min_search_successes=args.min_distribution_search_successes,
        min_target_discovered=args.min_distribution_target_discovered,
        min_search_clearance_m=args.min_distribution_search_clearance_m,
        min_search_displacement_m=args.min_distribution_search_displacement_m,
    )
    distribution_audit = audit_m20_mujoco_dataset(args.dataset, thresholds=distribution_thresholds)
    if args.require_distribution_gate and not distribution_audit["passed"]:
        raise RuntimeError(f"Dataset distribution gate failed: {json.dumps(distribution_audit['gates'], sort_keys=True)}")
    train_paths, val_paths, split_unit, val_layouts = split_paths_by_layout(
        paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_episode_weights = [float(episode_weights[path]) for path in train_paths]
    val_episode_weights = [float(episode_weights[path]) for path in val_paths]
    train_set = EpisodeFrames(
        train_paths,
        args.stride,
        episode_weights,
        history_pixel_threshold=int(args.history_pixel_threshold),
    )
    val_set = EpisodeFrames(
        val_paths,
        args.stride,
        episode_weights,
        history_pixel_threshold=int(args.history_pixel_threshold),
    )
    first_front_rgb = train_set.episodes[0][0]["front_rgb"]
    policy_input_resolution = {
        "width": int(first_front_rgb.shape[2]),
        "height": int(first_front_rgb.shape[1]),
    }
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = M20MuJoCoVLA().to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {args.init_checkpoint}")
        init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(init_payload["model_state_dict"], strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "init_missing_keys": missing,
                    "init_unexpected_keys": unexpected,
                },
                indent=2,
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation_augmentation = ObservationAugmentationConfig(
        rgb=RGBAugmentationConfig(
            enabled=bool(args.rgb_augmentation),
            brightness_jitter=float(args.rgb_brightness_jitter),
            contrast_jitter=float(args.rgb_contrast_jitter),
            noise_std=float(args.rgb_noise_std),
            cutout_prob=float(args.rgb_cutout_prob),
            cutout_size=int(args.rgb_cutout_size),
        ),
        lidar_noise_std=float(args.lidar_noise_std),
        lidar_dropout_prob=float(args.lidar_dropout_prob),
        proprio_noise_std=float(args.proprio_noise_std),
    )
    motion_loss_weights = torch.tensor([1.0, 0.75, 3.0], device=device, dtype=torch.float32)
    motion_loss_weight_normalizer = motion_loss_weights.mean().clamp_min(1.0e-6)
    config = {"schema": "m20pro_mujoco_vla_train_v4_history_close_approach", "vision": model.vision, "dataset": str(args.dataset), "train_episodes": [str(p) for p in train_paths], "val_episodes": [str(p) for p in val_paths], "train_episode_weights": train_episode_weights, "val_episode_weights": val_episode_weights, "split_unit": split_unit, "validation_layout_ids": sorted(val_layouts), "policy_input_resolution": policy_input_resolution, "target_loss_weight": args.target_loss_weight, "phase_loss_weight": args.phase_loss_weight, "visual_geometry_loss_weight": args.visual_geometry_loss_weight, "close_approach_loss_weight": args.close_approach_loss_weight, "motion_loss_weights": motion_loss_weights.detach().cpu().tolist(), "target_labels": list(TARGET_LABEL_TO_ID), "phase_labels": list(PHASE_LABELS), "visual_geometry_labels": list(VISUAL_GEOMETRY_LABELS), "history_feature_labels": list(HISTORY_FEATURE_LABELS), "history_pixel_threshold": int(args.history_pixel_threshold), "phase_label_contract": "search->approach transition uses visual target visibility in policy front/rear RGB, not geometric line-of-sight metadata", "visual_geometry_contract": "[visible, x_offset, area, rear_source] from policy front/rear RGB target-color pixels; no target XY runtime input", "history_contract": "short memory from policy RGB target pixels + previous executed body command; no target_xy/object_id/semantic mask at runtime", "text_encoding": TEXT_ENCODING, "text_token_length": TEXT_TOKEN_LENGTH, "device": str(device), "seed": args.seed}
    config["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint is not None else ""
    config["distribution_audit"] = distribution_audit
    config["distribution_gate_required"] = bool(args.require_distribution_gate)
    config["distribution_gate_thresholds"] = {
        "min_episodes": args.min_distribution_episodes,
        "min_layouts": args.min_distribution_layouts,
        "min_terrain_profiles": args.min_distribution_terrain_profiles,
        "min_languages": args.min_distribution_languages,
        "min_search_anchor_episodes": args.min_distribution_search_anchor_episodes,
        "min_search_varied_episodes": args.min_distribution_search_varied_episodes,
        "min_search_structured_episodes": args.min_distribution_search_structured_episodes,
        "min_search_successes": args.min_distribution_search_successes,
        "min_target_discovered": args.min_distribution_target_discovered,
        "min_search_clearance_m": args.min_distribution_search_clearance_m,
    }
    config["observation_augmentation"] = {
        "rgb": {
            "enabled": observation_augmentation.rgb.enabled,
            "brightness_jitter": observation_augmentation.rgb.brightness_jitter,
            "contrast_jitter": observation_augmentation.rgb.contrast_jitter,
            "noise_std": observation_augmentation.rgb.noise_std,
            "cutout_prob": observation_augmentation.rgb.cutout_prob,
            "cutout_size": observation_augmentation.rgb.cutout_size,
        },
        "lidar_noise_std": observation_augmentation.lidar_noise_std,
        "lidar_dropout_prob": observation_augmentation.lidar_dropout_prob,
        "proprio_noise_std": observation_augmentation.proprio_noise_std,
    }
    config["episode_weighting"] = {
        "success": 1.0,
        "terrain_weight_multiplier": TERRAIN_WEIGHT_MULTIPLIER,
        "search_discovered_base": 0.25,
        "search_distance_tau_m": 2.5,
        "search_undiscovered_multiplier": 0.75,
        "search_hidden_start_multiplier": 1.15,
        "search_discovered_before_reach_multiplier": 1.08,
        "search_yaw_offset_bonus_max": 0.12,
        "search_min_clearance_bonus_max": 0.28,
        "weight_floor": 0.15,
        "weight_ceiling": 1.35,
    }
    config["sample_weighting"] = {
        "search_pre_discovery_multiplier": 1.60,
        "search_post_discovery_multiplier": 1.25,
        "search_success_close_tail_window_steps": 160,
        "search_success_close_tail_multiplier": 2.80,
        "search_success_stop_tail_multiplier": 3.20,
        "search_success_post_discovery_multiplier": 1.35,
        "search_hidden_failure_multiplier": 0.55,
        "search_step_base": 0.90,
        "sample_floor": 0.20,
        "sample_ceiling": 4.00,
        "close_approach_weight_max": 3.50,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "distribution_audit.json").write_text(json.dumps(distribution_audit, indent=2) + "\n", encoding="utf-8")
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_total = 0.0; train_count = 0.0; train_metric_count = 0; train_target_correct = 0; train_phase_correct = 0; train_visual_visible_correct = 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            sample_weight = batch.pop("sample_weight")
            close_approach_weight = batch.pop("close_approach_weight")
            train_metric_count += int(batch["target_id"].numel())
            batch["rgb"], batch["lidar"], batch["proprio"] = apply_observation_augmentation(
                rgb=batch["rgb"],
                lidar=batch["lidar"],
                proprio=batch["proprio"],
                config=observation_augmentation,
            )
            latent = model.encode(batch["rgb"], batch["lidar"], batch["proprio"], batch["language"], history=batch["history"])
            target_logits = model.target_head(latent)
            phase_logits = model.phase_head(latent)
            visual_logits = model.visual_geometry_head(latent)
            target_context = model.target_context(batch["target_id"])
            prediction = model.action_from_latent(latent, target_context, phase_logits=phase_logits, history=batch["history"])
            motion_error = nn.functional.smooth_l1_loss(prediction[:, :3], batch["action"][:, :3], reduction="none")
            motion_loss = (motion_error * motion_loss_weights).sum(dim=1) / motion_loss_weight_normalizer
            stop_loss = nn.functional.binary_cross_entropy_with_logits(prediction[:, 3], batch["action"][:, 3], reduction="none")
            target_loss = nn.functional.cross_entropy(target_logits, batch["target_id"], reduction="none")
            phase_loss = nn.functional.cross_entropy(phase_logits, batch["phase_id"], reduction="none")
            visual_loss = visual_geometry_loss(visual_logits, batch["visual_geometry"])
            per_sample_loss = (
                motion_loss
                + stop_loss
                + args.target_loss_weight * target_loss
                + args.phase_loss_weight * phase_loss
                + args.visual_geometry_loss_weight * visual_loss
            )
            if args.close_approach_loss_weight > 0.0:
                close_loss = motion_loss + 0.75 * stop_loss
                per_sample_loss = per_sample_loss + args.close_approach_loss_weight * close_approach_weight * close_loss
            weight_sum = sample_weight.sum().clamp_min(1.0e-6)
            loss = (per_sample_loss * sample_weight).sum() / weight_sum
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_total += float(loss.detach()) * float(weight_sum.detach()); train_count += float(weight_sum.detach())
            train_target_correct += int((target_logits.argmax(dim=1) == batch["target_id"]).sum())
            train_phase_correct += int((phase_logits.argmax(dim=1) == batch["phase_id"]).sum())
            train_visual_visible_correct += int(((visual_logits[:, 0] > 0.0) == (batch["visual_geometry"][:, 0] > 0.5)).sum())
        model.eval(); val_total = 0.0; val_count = 0.0; val_metric_count = 0; stop_correct = 0; val_target_correct = 0; val_phase_correct = 0; val_visual_visible_correct = 0; val_visual_offset_abs = 0.0; val_visual_area_abs = 0.0; val_visual_visible_count = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sample_weight = batch.pop("sample_weight")
                close_approach_weight = batch.pop("close_approach_weight")
                val_metric_count += int(batch["target_id"].numel())
                latent = model.encode(batch["rgb"], batch["lidar"], batch["proprio"], batch["language"], history=batch["history"])
                target_logits = model.target_head(latent)
                phase_logits = model.phase_head(latent)
                visual_logits = model.visual_geometry_head(latent)
                target_context = model.target_context(batch["target_id"])
                prediction = model.action_from_latent(latent, target_context, phase_logits=phase_logits, history=batch["history"])
                motion_error = nn.functional.smooth_l1_loss(prediction[:, :3], batch["action"][:, :3], reduction="none")
                motion_loss = (motion_error * motion_loss_weights).sum(dim=1) / motion_loss_weight_normalizer
                stop_loss = nn.functional.binary_cross_entropy_with_logits(prediction[:, 3], batch["action"][:, 3], reduction="none")
                target_loss = nn.functional.cross_entropy(target_logits, batch["target_id"], reduction="none")
                phase_loss = nn.functional.cross_entropy(phase_logits, batch["phase_id"], reduction="none")
                visual_loss = visual_geometry_loss(visual_logits, batch["visual_geometry"])
                per_sample_loss = (
                    motion_loss
                    + stop_loss
                    + args.target_loss_weight * target_loss
                    + args.phase_loss_weight * phase_loss
                    + args.visual_geometry_loss_weight * visual_loss
                )
                if args.close_approach_loss_weight > 0.0:
                    close_loss = motion_loss + 0.75 * stop_loss
                    per_sample_loss = per_sample_loss + args.close_approach_loss_weight * close_approach_weight * close_loss
                weight_sum = sample_weight.sum().clamp_min(1.0e-6)
                loss = (per_sample_loss * sample_weight).sum() / weight_sum
                val_total += float(loss) * float(weight_sum.detach()); val_count += float(weight_sum.detach())
                stop_correct += int(((prediction[:, 3] > 0.0) == (batch["action"][:, 3] > 0.5)).sum())
                val_target_correct += int((target_logits.argmax(dim=1) == batch["target_id"]).sum())
                val_phase_correct += int((phase_logits.argmax(dim=1) == batch["phase_id"]).sum())
                val_visual_visible_correct += int(((visual_logits[:, 0] > 0.0) == (batch["visual_geometry"][:, 0] > 0.5)).sum())
                visible_mask = batch["visual_geometry"][:, 0] > 0.5
                if bool(visible_mask.any()):
                    val_visual_visible_count += float(visible_mask.sum().item())
                    val_visual_offset_abs += float(
                        torch.abs(torch.tanh(visual_logits[visible_mask, 1]) - batch["visual_geometry"][visible_mask, 1]).sum().item()
                    )
                    val_visual_area_abs += float(
                        torch.abs(torch.sigmoid(visual_logits[visible_mask, 2]) - batch["visual_geometry"][visible_mask, 2]).sum().item()
                    )
        scheduler.step(); row = {"epoch": epoch, "train_loss": train_total / train_count, "val_loss": val_total / val_count, "val_stop_accuracy": stop_correct / val_metric_count, "lr": scheduler.get_last_lr()[0]}
        row["train_target_accuracy"] = train_target_correct / train_metric_count
        row["val_target_accuracy"] = val_target_correct / val_metric_count
        row["train_phase_accuracy"] = train_phase_correct / train_metric_count
        row["val_phase_accuracy"] = val_phase_correct / val_metric_count
        row["train_visual_visible_accuracy"] = train_visual_visible_correct / train_metric_count
        row["val_visual_visible_accuracy"] = val_visual_visible_correct / val_metric_count
        row["val_visual_offset_mae_visible"] = (
            val_visual_offset_abs / val_visual_visible_count if val_visual_visible_count > 0.0 else None
        )
        row["val_visual_area_mae_visible"] = (
            val_visual_area_abs / val_visual_visible_count if val_visual_visible_count > 0.0 else None
        )
        history.append(row)
        print(f"[M20-MUJOCO-VLA] epoch={epoch:03d}/{args.epochs} train={row['train_loss']:.5f} val={row['val_loss']:.5f} stop_acc={row['val_stop_accuracy']:.3f} target_acc={row['val_target_accuracy']:.3f} phase_acc={row['val_phase_accuracy']:.3f} vis_acc={row['val_visual_visible_accuracy']:.3f}", flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_loss": best}, args.output_dir / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": args.epochs}, args.output_dir / "last.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output_dir / "best.pt"), "train_episodes": len(train_paths), "val_episodes": len(val_paths), "train_frames": len(train_set), "val_frames": len(val_set), "best_val_loss": best}, indent=2))


if __name__ == "__main__":
    main()

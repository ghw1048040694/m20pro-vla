#!/usr/bin/env python3
"""Create a symlinked MuJoCo M20 dataset view after metadata filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-canonical-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-base-height", type=float, default=0.45)
    parser.add_argument("--min-obstacle-clearance", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_passes(meta: dict, args: argparse.Namespace) -> bool:
    if bool(args.require_success) and not bool(meta.get("success", False)):
        return False
    if bool(args.require_canonical_success) and not bool(meta.get("canonical_success", False)):
        return False
    if float(meta.get("min_base_height", 0.0)) < float(args.min_base_height):
        return False
    if args.min_obstacle_clearance is not None:
        clearance = meta.get("min_obstacle_clearance")
        if clearance is None or float(clearance) < float(args.min_obstacle_clearance):
            return False
    return True


def main() -> None:
    args = parse_args()
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
        for path in args.output_dir.glob("episode_*.*"):
            path.unlink()
        (args.output_dir / "dataset_summary.json").unlink(missing_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    rejected = []
    for meta_path in sorted(args.dataset.glob("episode_*.json")):
        npz_path = meta_path.with_suffix(".npz")
        if not npz_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if episode_passes(meta, args):
            output_index = len(kept)
            out_json = args.output_dir / f"episode_{output_index:04d}.json"
            out_npz = args.output_dir / f"episode_{output_index:04d}.npz"
            out_json.symlink_to(meta_path.resolve())
            out_npz.symlink_to(npz_path.resolve())
            kept.append({**meta, "source_episode_json": str(meta_path), "filtered_episode_id": output_index})
        else:
            rejected.append(meta)

    if not kept:
        raise RuntimeError("Filter produced an empty dataset")
    summary = {
        "schema": "m20pro_mujoco_vla_filtered_dataset_v1",
        "source_dataset": str(args.dataset),
        "filters": {
            "require_success": bool(args.require_success),
            "require_canonical_success": bool(args.require_canonical_success),
            "min_base_height": float(args.min_base_height),
            "min_obstacle_clearance": args.min_obstacle_clearance,
        },
        "episodes": kept,
        "source_episode_count": len(kept) + len(rejected),
        "kept_episode_count": len(kept),
        "rejected_episode_count": len(rejected),
        "success_count": sum(bool(item.get("success", False)) for item in kept),
        "canonical_success_count": sum(bool(item.get("canonical_success", False)) for item in kept),
        "target_discovered_count": sum(bool(item.get("target_discovered", False)) for item in kept),
        "policy_input_resolution": kept[0].get("policy_input_resolution", {}),
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("source_episode_count", "kept_episode_count", "rejected_episode_count", "success_count")}, indent=2))


if __name__ == "__main__":
    main()

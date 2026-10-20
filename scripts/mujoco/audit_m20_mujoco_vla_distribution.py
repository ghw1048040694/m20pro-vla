#!/usr/bin/env python3
"""Audit MuJoCo M20 VLA dataset distribution coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m20pro_vla.data import M20MuJoCoDistributionThresholds, audit_m20_mujoco_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min-episodes", type=int, default=4)
    parser.add_argument("--min-layouts", type=int, default=2)
    parser.add_argument("--min-terrain-profiles", type=int, default=2)
    parser.add_argument("--min-languages", type=int, default=2)
    parser.add_argument("--min-search-anchor-episodes", type=int, default=1)
    parser.add_argument("--min-search-varied-episodes", type=int, default=1)
    parser.add_argument(
        "--min-search-structured-episodes",
        type=int,
        default=0,
        help="Structured (S2/S3) episodes instead of the anchor/varied split; 0 disables the clause.",
    )
    parser.add_argument("--min-search-successes", type=int, default=1)
    parser.add_argument("--min-target-discovered", type=int, default=1)
    parser.add_argument("--min-search-clearance-m", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = M20MuJoCoDistributionThresholds(
        min_episodes=args.min_episodes,
        min_layouts=args.min_layouts,
        min_terrain_profiles=args.min_terrain_profiles,
        min_languages=args.min_languages,
        min_search_anchor_episodes=args.min_search_anchor_episodes,
        min_search_varied_episodes=args.min_search_varied_episodes,
        min_search_structured_episodes=args.min_search_structured_episodes,
        min_search_successes=args.min_search_successes,
        min_target_discovered=args.min_target_discovered,
        min_search_clearance_m=args.min_search_clearance_m,
    )
    report = audit_m20_mujoco_dataset(args.dataset, thresholds=thresholds)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

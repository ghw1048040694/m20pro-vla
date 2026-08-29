#!/usr/bin/env python3

"""Write the canonical M20 ObjectNav scenario manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m20_indoor_scenarios import build_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--include-hidden-search",
        action="store_true",
        help="Build the hidden-search variant with occluding geometry and occluded targets.",
    )
    args = parser.parse_args()
    if args.output is None:
        output_name = (
            "configs/m20_hidden_objectnav_scenarios_v1.json"
            if args.include_hidden_search
            else "configs/m20_visible_objectnav_scenarios_v1.json"
        )
        args.output = Path(__file__).resolve().parents[1] / output_name
    manifest = build_manifest(args.seed, include_hidden_search=args.include_hidden_search)
    summary = validate_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

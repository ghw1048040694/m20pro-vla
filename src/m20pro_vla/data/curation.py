"""Build a space-efficient, audited view of trainable M20 episodes."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from .distribution import M20MuJoCoDistributionThresholds, audit_m20_mujoco_dataset


def curate_episode_view(
    *,
    source: Path,
    recovery: Path,
    output: Path,
    thresholds: M20MuJoCoDistributionThresholds,
) -> dict:
    source, recovery, output = Path(source), Path(recovery), Path(output)
    if output.exists():
        raise FileExistsError(f"Curated dataset already exists: {output}")
    if not source.is_dir() or not recovery.is_dir():
        raise FileNotFoundError("Source and accepted recovery directories must exist")

    selected: list[tuple[Path, Path]] = []
    excluded: list[int] = []
    seen: set[int] = set()
    for directory in (source, recovery):
        for json_path in sorted(directory.glob("episode_*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.is_file():
                raise FileNotFoundError(f"Episode arrays missing: {npz_path}")
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            episode_id = int(metadata["episode_id"])
            if episode_id in seen:
                raise ValueError(f"Duplicate episode ID: {episode_id}")
            seen.add(episode_id)
            if metadata.get("quality_passed") is False:
                excluded.append(episode_id)
                continue
            if metadata.get("collection_mode") == "search":
                start, end = metadata["initial_xy"], metadata["final_xy"]
                if math.dist(start, end) < thresholds.min_search_displacement_m:
                    excluded.append(episode_id)
                    continue
            if directory == recovery and metadata.get("collection_mode") != "failure_recovery":
                raise ValueError(f"Unexpected recovery mode: {json_path}")
            selected.append((json_path, npz_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for json_path, npz_path in selected:
            (temporary / json_path.name).symlink_to(json_path.resolve())
            (temporary / npz_path.name).symlink_to(npz_path.resolve())
        audit = audit_m20_mujoco_dataset(temporary, thresholds=thresholds)
        if not audit["passed"]:
            raise ValueError(
                "Curated dataset failed audit: "
                + json.dumps({"gates": audit["gates"], "low_motion_ids": audit["low_motion_episode_ids"]})
            )
        audit["dataset"] = str(output)
        report = {
            "schema": "m20pro_curated_episode_view_v1",
            "source": str(source),
            "recovery": str(recovery),
            "output": str(output),
            "selected_episode_count": len(selected),
            "excluded_episode_ids": sorted(excluded),
            "audit": audit,
        }
        (temporary / "curation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

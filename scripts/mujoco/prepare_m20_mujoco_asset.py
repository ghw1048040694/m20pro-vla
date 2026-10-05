#!/usr/bin/env python3
"""Convert the ROS M20 URDF into a MuJoCo-readable MJCF asset.

The source URDF is retained unchanged. This script only resolves its ROS
``package://`` mesh URIs into the local mesh directory, lets MuJoCo compile the
URDF, and exports the resulting MJCF plus an auditable topology report.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_URDF = WORKSPACE / "src/m20pro_description/urdf/M20.urdf"
MESH_DIR = WORKSPACE / "src/m20pro_description/meshes"
DEFAULT_OUTPUT_DIR = WORKSPACE / ".runtime/mujoco_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_urdf(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        raise ValueError("M20 URDF has no MuJoCo compiler section")
    compiler.set("meshdir", str(MESH_DIR))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("Encountered a mesh without a filename")
        resolved = MESH_DIR / Path(filename).name
        if not resolved.is_file():
            raise FileNotFoundError(f"M20 mesh is missing: {resolved}")
        mesh.set("filename", resolved.name)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"M20 URDF is missing: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_urdf = output_dir / "M20_resolved.urdf"
    output_mjcf = output_dir / "M20_compiled.xml"
    report_path = output_dir / "M20_compiled_report.json"
    normalize_urdf(source, normalized_urdf)

    model = mujoco.MjModel.from_xml_path(str(normalized_urdf))
    mujoco.mj_saveLastXML(str(output_mjcf), model)
    report = {
        "schema": "m20pro_mujoco_asset_compile_v1",
        "source_urdf": str(source),
        "normalized_urdf": str(normalized_urdf),
        "compiled_mjcf": str(output_mjcf),
        "mujoco_version": mujoco.__version__,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)],
        "note": "URDF compilation validates geometry/topology only. A floating base and actuators are required before dynamics training.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

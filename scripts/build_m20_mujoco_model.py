#!/usr/bin/env python3
"""Add dynamics-critical MuJoCo elements to the compiled M20 MJCF asset."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
ASSET_DIR = WORKSPACE / ".runtime/mujoco_assets"
DEFAULT_INPUT = ASSET_DIR / "M20_compiled.xml"
DEFAULT_OUTPUT = ASSET_DIR / "M20_floating_actuated.xml"

LEG_JOINTS = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
)
WHEEL_JOINTS = ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=480)
    return parser.parse_args()


def add_dynamics_elements(root: ET.Element) -> None:
    root.insert(1, ET.Element("option", {"timestep": "0.0025", "integrator": "implicitfast"}))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Compiled MJCF has no worldbody")
    base = worldbody.find("./body[@name='base_link']")
    if base is None:
        raise ValueError("Compiled MJCF has no base_link body")

    worldbody.insert(0, ET.Element("geom", {
        "name": "ground",
        "type": "plane",
        "size": "0 0 0.1",
        "friction": "1.2 0.005 0.0001",
        "rgba": "0.18 0.20 0.23 1",
    }))
    base.set("pos", "0 0 0.60")
    base.insert(1, ET.Element("freejoint", {"name": "base_free"}))

    joint_ranges = {
        joint.get("name"): joint.get("range")
        for joint in root.findall(".//joint")
        if joint.get("name") and joint.get("range")
    }
    actuator = ET.Element("actuator")
    for joint in LEG_JOINTS:
        if joint not in joint_ranges:
            raise ValueError(f"Missing compiled joint range: {joint}")
        actuator.append(ET.Element("position", {
            "name": f"{joint}_position",
            "joint": joint,
            "kp": "80",
            "ctrllimited": "true",
            "ctrlrange": joint_ranges[joint],
            "forcelimited": "true",
            "forcerange": "-76.4 76.4",
        }))
    for joint in WHEEL_JOINTS:
        actuator.append(ET.Element("velocity", {
            "name": f"{joint}_velocity",
            "joint": joint,
            "kv": "0.3",
            "ctrllimited": "true",
            "ctrlrange": "-20 20",
            "forcelimited": "true",
            "forcerange": "-21.6 21.6",
        }))
    root.append(actuator)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Compile the M20 URDF first: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_path)
    root = tree.getroot()
    add_dynamics_elements(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    data = mujoco.MjData(model)
    if model.njnt != 17 or model.nu != 16 or model.nq != 23 or model.nv != 22:
        raise RuntimeError(
            f"Unexpected floating M20 topology: njnt={model.njnt} nq={model.nq} nv={model.nv} nu={model.nu}"
        )
    data.ctrl[:len(LEG_JOINTS)] = data.qpos[7:7 + len(LEG_JOINTS)]
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    base_pos = data.qpos[:3]
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Non-finite state after the M20 MuJoCo dynamics smoke test")

    report = {
        "schema": "m20pro_mujoco_floating_asset_v1",
        "input_mjcf": str(input_path),
        "output_mjcf": str(output_path),
        "mujoco_version": mujoco.__version__,
        "njnt": model.njnt,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "control_hz": 1.0 / model.opt.timestep,
        "smoke_steps": args.steps,
        "final_base_xyz": [float(value) for value in base_pos],
        "finite_state": True,
        "note": "Dynamics topology passed. This is not a locomotion or VLA success result.",
    }
    report_path = output_path.with_name("M20_floating_actuated_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

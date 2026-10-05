#!/usr/bin/env python3
"""Add dynamics-critical MuJoCo elements to the compiled M20 MJCF asset."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
ASSET_DIR = WORKSPACE / ".runtime/mujoco_assets"
DEFAULT_INPUT = ASSET_DIR / "M20_compiled.xml"
DEFAULT_OUTPUT = ASSET_DIR / "M20_floating_actuated.xml"
DEFAULT_CONTRACT = WORKSPACE / "configs/m20pro_low_level_v1.yaml"

LEG_KP = 80.0
DEFAULT_LEG_KV = 0.0
DEFAULT_WHEEL_KV = 0.3
DEFAULT_WHEEL_GROUND_FRICTION = 1.2

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
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--steps", type=int, default=480)
    return parser.parse_args()


def load_dynamics_contract(contract_path: Path) -> tuple[dict[str, float], float, float]:
    """Read the leg/wheel actuator gains from the single low-level contract.

    The contract YAML is the only authority for bottom-controller gains. Keeping
    the generator on the same file removes the class of drift where the contract
    documents one damping value while the built asset silently carries another.

    ``leg_position_actuator_kv`` accepts either one number applied to all twelve
    leg joints, or a mapping with ``hipx``/``hipy``/``knee`` entries so a single
    joint group can be damped without stiffening the whole stance.
    """
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing low-level contract: {contract_path}")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    feedback = contract.get("feedback") or {}
    raw_leg_kv = feedback.get("leg_position_actuator_kv", DEFAULT_LEG_KV)
    wheel_kv = feedback.get("wheel_velocity_actuator_kv", DEFAULT_WHEEL_KV)
    wheel_ground_friction = feedback.get(
        "wheel_ground_sliding_friction", DEFAULT_WHEEL_GROUND_FRICTION
    )

    def _check(name: str, value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
        return float(value)

    if isinstance(raw_leg_kv, dict):
        unknown = set(raw_leg_kv) - {"hipx", "hipy", "knee"}
        if unknown:
            raise ValueError(f"Unknown leg joint group(s) in leg_position_actuator_kv: {sorted(unknown)}")
        leg_kv = {
            group: _check(f"leg_position_actuator_kv.{group}", raw_leg_kv.get(group, DEFAULT_LEG_KV))
            for group in ("hipx", "hipy", "knee")
        }
    else:
        value = _check("leg_position_actuator_kv", raw_leg_kv)
        leg_kv = {group: value for group in ("hipx", "hipy", "knee")}
    return (
        leg_kv,
        _check("wheel_velocity_actuator_kv", wheel_kv),
        _check("wheel_ground_sliding_friction", wheel_ground_friction),
    )


def leg_kv_for(joint: str, leg_kv: dict[str, float]) -> float:
    for group in ("hipx", "hipy", "knee"):
        if joint.endswith(f"{group}_joint"):
            return leg_kv[group]
    raise ValueError(f"Cannot map leg joint to a damping group: {joint}")


def add_dynamics_elements(
    root: ET.Element,
    *,
    leg_kv: dict[str, float],
    wheel_kv: float,
    wheel_ground_friction: float,
) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Compiled MJCF has no compiler")
    compiler.set("meshdir", str(WORKSPACE / "src" / "m20pro_description" / "meshes"))
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
        "friction": f"{wheel_ground_friction:g} 0.005 0.0001",
        "rgba": "0.18 0.20 0.23 1",
    }))
    base.set("pos", "0 0 0.60")
    base.insert(1, ET.Element("freejoint", {"name": "base_free"}))
    for wheel_name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"):
        wheel = base.find(f".//body[@name='{wheel_name}']")
        if wheel is None:
            raise ValueError(f"Compiled MJCF has no {wheel_name} body")
        for geom in wheel.findall("geom"):
            if geom.get("contype", "1") != "0":
                geom.set("friction", f"{wheel_ground_friction:g} 0.005 0.0001")

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
            "kp": f"{LEG_KP:g}",
            "kv": f"{leg_kv_for(joint, leg_kv):g}",
            "ctrllimited": "true",
            "ctrlrange": joint_ranges[joint],
            "forcelimited": "true",
            "forcerange": "-76.4 76.4",
        }))
    for joint in WHEEL_JOINTS:
        actuator.append(ET.Element("velocity", {
            "name": f"{joint}_velocity",
            "joint": joint,
            "kv": f"{wheel_kv:g}",
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
    contract_path = args.contract.resolve()
    leg_kv, wheel_kv, wheel_ground_friction = load_dynamics_contract(contract_path)
    add_dynamics_elements(
        root,
        leg_kv=leg_kv,
        wheel_kv=wheel_kv,
        wheel_ground_friction=wheel_ground_friction,
    )
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
        "low_level_contract": str(contract_path),
        "leg_position_actuator_kp": LEG_KP,
        "leg_position_actuator_kv": leg_kv,
        "leg_position_actuator_kv_uniform": (
            next(iter(leg_kv.values())) if len(set(leg_kv.values())) == 1 else None
        ),
        "wheel_velocity_actuator_kv": wheel_kv,
        "wheel_ground_sliding_friction": wheel_ground_friction,
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

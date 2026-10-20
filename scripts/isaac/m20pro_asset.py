"""Stage and import the canonical M20 Pro URDF with Isaac Sim 6.x."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


LEGS = ("fl", "fr", "hl", "hr")
JOINTS = {
    f"{leg}_{joint}_joint"
    for leg in LEGS
    for joint in ("hipx", "hipy", "knee", "wheel")
}


def inspect_urdf(package_dir: Path) -> dict:
    urdf = package_dir / "urdf" / "M20.urdf"
    root = ET.parse(urdf).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}
    if set(joints) != JOINTS or len(links) != 17:
        raise ValueError(f"Expected 16 M20 joints and 17 links; found {len(joints)} joints and {len(links)} links")
    if "base_link" not in links:
        raise ValueError("Missing base_link")
    for name, joint in joints.items():
        expected_type = "continuous" if name.endswith("wheel_joint") else "revolute"
        if joint.get("type") != expected_type:
            raise ValueError(f"{name} must be {expected_type}")
        limit = joint.find("limit")
        if limit is None or float(limit.get("effort", "0")) <= 0 or float(limit.get("velocity", "0")) <= 0:
            raise ValueError(f"{name} has invalid effort/velocity limits")
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None or float(inertial.find("mass").get("value")) <= 0:
            raise ValueError(f"{name} has no positive mass")
        if link.find("collision") is None:
            raise ValueError(f"{name} has no collision geometry")
    mesh_paths = set()
    for mesh in root.findall(".//mesh"):
        uri = mesh.get("filename", "")
        prefix = "package://m20pro_description/"
        if not uri.startswith(prefix):
            raise ValueError(f"Unexpected mesh URI: {uri}")
        relative = Path(uri[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts or not (package_dir / relative).is_file():
            raise ValueError(f"Missing or unsafe mesh: {uri}")
        mesh_paths.add(relative)
    if len(mesh_paths) != 17:
        raise ValueError(f"Expected 17 visual meshes; found {len(mesh_paths)}")
    return {
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "joint_names": sorted(joints),
        "wheel_joint_names": sorted(name for name in joints if name.endswith("wheel_joint")),
        "link_count": len(links),
        "mesh_count": len(mesh_paths),
        "mesh_paths": sorted(str(path) for path in mesh_paths),
    }


def stage(source: Path, destination: Path) -> None:
    report = inspect_urdf(source)
    package_dir = destination / "m20pro_description"
    (package_dir / "urdf").mkdir(parents=True, exist_ok=True)
    (package_dir / "meshes").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "urdf" / "M20.urdf", package_dir / "urdf" / "M20.urdf")
    for relative in report["mesh_paths"]:
        shutil.copy2(source / relative, package_dir / relative)
    shutil.copy2(Path(__file__), destination / "m20pro_asset.py")
    if inspect_urdf(package_dir) != report:
        raise RuntimeError("Staged M20 URDF differs from source")
    (destination / "urdf_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_dir": str(destination), **report}))


def import_usd(asset_dir: Path, usd_dir: Path) -> None:
    package_dir = asset_dir / "m20pro_description"
    report = inspect_urdf(package_dir)
    usd_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        for extension in ("omni.scene.optimizer.core", "isaacsim.robot.schema"):
            extension_manager.set_extension_enabled_immediate(extension, True)

        from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
        from pxr import Usd, UsdPhysics

        config = URDFImporterConfig()
        config.urdf_path = str(package_dir / "urdf" / "M20.urdf")
        config.usd_path = str(usd_dir)
        config.ros_package_paths = [{"name": "m20pro_description", "path": str(package_dir)}]
        config.fix_base = False
        config.merge_fixed_joints = False
        config.collision_from_visuals = False
        config.allow_self_collision = False
        config.joint_target_type = "none"
        config.robot_type = "Quadruped"
        output = URDFImporter(config).import_urdf()
        if not output or not Path(output).is_file():
            raise RuntimeError(f"Isaac Sim did not produce a USD asset: {output}")
        stage = Usd.Stage.Open(str(output))
        if stage is None:
            raise RuntimeError(f"Cannot open imported USD: {output}")
        imported_joints = {
            prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        }
        missing = JOINTS - imported_joints
        if missing:
            raise RuntimeError(f"Imported USD is missing joints: {sorted(missing)}")
        articulation_roots = [
            str(prim.GetPath()) for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if not articulation_roots:
            raise RuntimeError("Imported USD has no articulation root")
        result = {
            "schema": "m20pro_isaac_asset_v1",
            "source_urdf_sha256": report["urdf_sha256"],
            "usd_path": str(output),
            "joint_names": sorted(imported_joints & JOINTS),
            "articulation_roots": articulation_roots,
            "simulated": False,
        }
        (asset_dir / "usd_import_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--source", type=Path, required=True)
    stage_parser.add_argument("--destination", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--asset-dir", type=Path, required=True)
    import_parser.add_argument("--usd-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "stage":
        stage(args.source.resolve(), args.destination.resolve())
    else:
        import_usd(args.asset_dir.resolve(), args.usd_dir.resolve())


if __name__ == "__main__":
    main()

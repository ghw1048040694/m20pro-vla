"""Build a reproducible MuJoCo scene from the official M20 robot MJCF asset."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.environ.get("M20PRO_VLA_DATA_ROOT", WORKSPACE / ".runtime"))
ASSET_ROOT = RUNTIME_ROOT / "assets" / "m20_official"
OFFICIAL_MJCF = ASSET_ROOT / "mjcf" / "M20.xml"
MESH_ROOT = Path(os.environ.get("M20PRO_VLA_MESH_ROOT", WORKSPACE / "assets" / "m20_meshes"))

OFFICIAL_M20_COMMIT = "ec30acfa65131cd87f94260b8b2c552fba3b798e"
OFFICIAL_M20_URL = (
    "https://raw.githubusercontent.com/AI-DA-STC/M20-autonomy-sim/"
    f"{OFFICIAL_M20_COMMIT}/src/M20_sdk_deploy/model/M20/mjcf/M20.xml"
)
OFFICIAL_M20_SHA256 = "323048fbe5861522e3015bc12b8cd03e3d5a452f0b8a7b959df910bca82bdbf4"
OFFICIAL_POLICY = ASSET_ROOT / "policy" / "policy.onnx"
OFFICIAL_POLICY_URL = (
    "https://raw.githubusercontent.com/AI-DA-STC/M20-autonomy-sim/"
    f"{OFFICIAL_M20_COMMIT}/src/M20_sdk_deploy/policy/policy.onnx"
)
OFFICIAL_POLICY_SHA256 = "338cc3083af356119b76113ab0e6ad720ae98d4e2059aba5133303e806c88ad1"


@dataclass(frozen=True)
class TargetSpec:
    """A visible, non-colliding task object for the initial ObjectNav stage."""

    name: str = "green_cube"
    position: tuple[float, float, float] = (2.2, 0.0, 0.16)
    rgba: tuple[float, float, float, float] = (0.05, 0.8, 0.2, 1.0)
    half_extent: float = 0.16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_official_mjcf() -> Path:
    """Fetch the pinned model XML only when the checked asset is absent."""
    if OFFICIAL_MJCF.is_file():
        if _sha256(OFFICIAL_MJCF) != OFFICIAL_M20_SHA256:
            raise RuntimeError(f"Official M20 MJCF checksum mismatch: {OFFICIAL_MJCF}")
        return OFFICIAL_MJCF

    OFFICIAL_MJCF.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OFFICIAL_M20_URL, timeout=60) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(dir=OFFICIAL_MJCF.parent, delete=False) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    try:
        if _sha256(temporary_path) != OFFICIAL_M20_SHA256:
            raise RuntimeError("Downloaded official M20 MJCF has an unexpected checksum")
        temporary_path.replace(OFFICIAL_MJCF)
    finally:
        temporary_path.unlink(missing_ok=True)
    return OFFICIAL_MJCF


def ensure_official_policy() -> Path:
    """Fetch the pinned low-level locomotion policy only when absent."""
    if OFFICIAL_POLICY.is_file():
        if _sha256(OFFICIAL_POLICY) != OFFICIAL_POLICY_SHA256:
            raise RuntimeError(f"Official M20 policy checksum mismatch: {OFFICIAL_POLICY}")
        return OFFICIAL_POLICY

    OFFICIAL_POLICY.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OFFICIAL_POLICY_URL, timeout=60) as response:
        payload = response.read()
    with tempfile.NamedTemporaryFile(dir=OFFICIAL_POLICY.parent, delete=False) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    try:
        if _sha256(temporary_path) != OFFICIAL_POLICY_SHA256:
            raise RuntimeError("Downloaded official M20 policy has an unexpected checksum")
        temporary_path.replace(OFFICIAL_POLICY)
    finally:
        temporary_path.unlink(missing_ok=True)
    return OFFICIAL_POLICY


def _required_meshes(root: ET.Element) -> set[str]:
    return {mesh.attrib["file"] for mesh in root.findall("./asset/mesh")}


def _assert_mesh_contract(root: ET.Element) -> None:
    missing = sorted(name for name in _required_meshes(root) if not (MESH_ROOT / name).is_file())
    if missing:
        raise FileNotFoundError(f"M20 mesh assets are missing: {missing}")


def materialize_scene(output_path: Path, target: TargetSpec = TargetSpec()) -> Path:
    """Write one M20 scene with a forward camera and a language-addressable target."""
    source_path = ensure_official_mjcf()
    tree = ET.parse(source_path)
    root = tree.getroot()
    _assert_mesh_contract(root)

    mesh_compiler = root.find("./compiler[@meshdir]")
    if mesh_compiler is None:
        raise RuntimeError("Official M20 MJCF does not declare a mesh compiler path")
    mesh_compiler.set("meshdir", str(MESH_ROOT))

    option = root.find("./option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")

    worldbody = root.find("./worldbody")
    base = root.find(".//body[@name='base_link']")
    if worldbody is None or base is None:
        raise RuntimeError("Official M20 MJCF is missing the world body or base link")

    ET.SubElement(
        base,
        "camera",
        {
            "name": "front_rgb",
            "pos": "0.38 0 0.08",
            "euler": "0 -1.57079632679 0",
            "fovy": "72",
        },
    )
    ET.SubElement(
        worldbody,
        "body",
        {
            "name": target.name,
            "mocap": "true",
            "pos": " ".join(str(value) for value in target.position),
        },
    )
    target_body = worldbody[-1]
    ET.SubElement(
        target_body,
        "geom",
        {
            "name": f"{target.name}_geom",
            "type": "box",
            "size": f"{target.half_extent} {target.half_extent} {target.half_extent}",
            "rgba": " ".join(str(value) for value in target.rgba),
            "contype": "0",
            "conaffinity": "0",
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path

"""Create a real-runtime policy export manifest for M20Pro VLA checkpoints.

The export is intentionally contract-first.  The real ROS workspace should be
able to inspect observation/action semantics, safety limits, and checkpoint
identity before it attempts to load any model weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from m20pro_vla import __version__
from m20pro_vla.data.history import HISTORY_FEATURE_DIM, HISTORY_FEATURE_LABELS
from m20pro_vla.runtime.run_context import workspace_root

# Keep deployment manifest generation usable on lightweight ROS hosts that do
# not install torch. These constants must match m20pro_vla.policies.compact.
TEXT_ENCODING = "utf8_byte_v1"
TEXT_TOKEN_LENGTH = 64
PHASE_LABELS = ("search", "approach", "stop")
VISUAL_GEOMETRY_LABELS = ("visible", "x_offset", "area", "rear_source")


POLICY_EXPORT_SCHEMA = "m20pro_vla_policy_export_v1"
DEFAULT_REAL_CONTRACT = Path("configs/m20pro_real_vla_deploy_contract_v1.yaml")


@dataclass(frozen=True)
class PolicyExportRequest:
    checkpoint: Path
    output_dir: Path
    policy_id: str
    real_contract: Path
    copy_checkpoint: bool = False
    image_width: int = 160
    image_height: int = 96
    target_runtime: str = "shadow"
    notes: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _default_policy_id(checkpoint: Path) -> str:
    parent = checkpoint.parent.name.strip()
    return parent or checkpoint.stem


def policy_export_plan(
    *,
    checkpoint: Path | None,
    output_dir: Path | None = None,
    policy_id: str | None = None,
    real_contract: Path | None = None,
    copy_checkpoint: bool = False,
    image_width: int = 160,
    image_height: int = 96,
    target_runtime: str = "shadow",
) -> dict[str, Any]:
    root = workspace_root()
    checkpoint_path = checkpoint.expanduser() if checkpoint is not None else None
    resolved_contract = (real_contract or root / DEFAULT_REAL_CONTRACT).expanduser()
    if not resolved_contract.is_absolute():
        resolved_contract = root / resolved_contract
    resolved_output = output_dir.expanduser() if output_dir is not None else None
    if resolved_output is not None and not resolved_output.is_absolute():
        resolved_output = root / resolved_output
    return {
        "schema": "m20pro_vla_policy_export_plan_v1",
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_present": bool(checkpoint_path is not None and checkpoint_path.is_file()),
        "output_dir": None if resolved_output is None else str(resolved_output),
        "policy_id": policy_id or (_default_policy_id(checkpoint_path) if checkpoint_path else None),
        "real_contract": str(resolved_contract),
        "real_contract_present": resolved_contract.is_file(),
        "copy_checkpoint": bool(copy_checkpoint),
        "image": {"width": int(image_width), "height": int(image_height)},
        "target_runtime": str(target_runtime),
        "first_consumer": "M20Pro-3D-Nav m20pro_vla_bridge shadow mode",
    }


def _manifest(request: PolicyExportRequest, checkpoint_sha256: str, checkpoint_output: Path | None) -> dict[str, Any]:
    checkpoint = request.checkpoint
    training_config = _read_json_if_present(checkpoint.parent / "config.json")
    distribution_audit = _read_json_if_present(checkpoint.parent / "distribution_audit.json")
    history = _read_json_if_present(checkpoint.parent / "history.json")
    return {
        "schema": POLICY_EXPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "policy_id": request.policy_id,
        "target_runtime": request.target_runtime,
        "real_runtime": {
            "project": "M20Pro-3D-Nav",
            "first_mode": "shadow",
            "preferred_motion_handoff": "vla_short_subgoal_to_dddmr_3d",
            "velocity_handoff": "body_command_to_twist_only_after_supervisor_gate",
            "per_map_lora_required": False,
        },
        "checkpoint": {
            "source_path": str(checkpoint),
            "source_name": checkpoint.name,
            "sha256": checkpoint_sha256,
            "bytes": checkpoint.stat().st_size,
            "copied": checkpoint_output is not None,
            "export_path": None if checkpoint_output is None else str(checkpoint_output.name),
        },
        "model": {
            "class": "m20pro_vla.policies.compact.M20MuJoCoVLA",
            "text_encoding": TEXT_ENCODING,
            "text_token_length": TEXT_TOKEN_LENGTH,
            "phase_labels": list(PHASE_LABELS),
            "visual_geometry_labels": list(VISUAL_GEOMETRY_LABELS),
            "history_feature_dim": HISTORY_FEATURE_DIM,
            "history_feature_labels": list(HISTORY_FEATURE_LABELS),
            "training_config_present": training_config is not None,
        },
        "observation_contract": {
            "front_rgb": {"dtype": "uint8", "layout": "HWC_RGB", "width": request.image_width, "height": request.image_height},
            "rear_rgb": {"dtype": "uint8", "layout": "HWC_RGB", "width": request.image_width, "height": request.image_height},
            "lidar_72": {"dtype": "float32", "shape": [72], "source": "/scan downsample or edge-cloud projection"},
            "proprio": {"dtype": "float32", "shape": [45], "source": "pose/twist/history adapter"},
            "language_instruction": {"encoding": TEXT_ENCODING, "token_length": TEXT_TOKEN_LENGTH},
        },
        "action_contract": {
            "body_command_fields": ["forward_mps", "lateral_mps", "yaw_radps", "stop"],
            "sim_limits": {"forward_mps": [-0.40, 0.40], "lateral_mps": [-0.20, 0.20], "yaw_radps": [-0.15, 0.15]},
            "first_real_limits": {"forward_mps": [-0.05, 0.08], "lateral_mps": [0.0, 0.0], "yaw_radps": [-0.15, 0.15]},
        },
        "safety": {
            "shadow_required_before_motion": True,
            "dddmr_field_test_required_for_subgoal": True,
            "operator_estop_required_for_motion": True,
            "direct_tcp_control_prohibited": True,
            "direct_joint_control_prohibited": True,
        },
        "source_artifacts": {
            "training_config": training_config,
            "distribution_audit_summary": distribution_audit,
            "history_summary": history,
        },
        "notes": request.notes,
    }


def create_policy_export(
    *,
    checkpoint: Path,
    output_dir: Path,
    policy_id: str | None = None,
    real_contract: Path | None = None,
    copy_checkpoint: bool = False,
    image_width: int = 160,
    image_height: int = 96,
    target_runtime: str = "shadow",
    notes: str = "",
) -> dict[str, Any]:
    root = workspace_root()
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    contract = (real_contract or root / DEFAULT_REAL_CONTRACT).expanduser()
    if not contract.is_absolute():
        contract = root / contract
    if not contract.is_file():
        raise FileNotFoundError(f"real deploy contract not found: {contract}")
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    request = PolicyExportRequest(
        checkpoint=checkpoint,
        output_dir=output_dir,
        policy_id=policy_id or _default_policy_id(checkpoint),
        real_contract=contract,
        copy_checkpoint=copy_checkpoint,
        image_width=int(image_width),
        image_height=int(image_height),
        target_runtime=str(target_runtime),
        notes=str(notes or ""),
    )
    checkpoint_output = None
    if copy_checkpoint:
        checkpoint_output = output_dir / checkpoint.name
        if checkpoint_output.resolve() != checkpoint.resolve():
            shutil.copy2(checkpoint, checkpoint_output)
    contract_output = output_dir / contract.name
    if contract_output.resolve() != contract.resolve():
        shutil.copy2(contract, contract_output)
    manifest = _manifest(request, _sha256(checkpoint), checkpoint_output)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "schema": "m20pro_vla_policy_export_result_v1",
        "policy_id": request.policy_id,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "contract": str(contract_output),
        "checkpoint_copied": bool(checkpoint_output is not None),
        "checkpoint_export_path": None if checkpoint_output is None else str(checkpoint_output),
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "ready_for_m20pro_3d_nav_shadow": True,
    }


__all__ = [
    "POLICY_EXPORT_SCHEMA",
    "PolicyExportRequest",
    "create_policy_export",
    "policy_export_plan",
]

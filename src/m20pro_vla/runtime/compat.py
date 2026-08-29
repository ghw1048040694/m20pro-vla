"""Temporary adapters for workflows whose business logic is still scripted.

This module deliberately keeps the compatibility boundary in one place.  It
lets users operate the project through one command while the collection and
training implementations are migrated into package APIs in later stages.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .run_context import RunContext, workspace_root


COMPATIBILITY_SCRIPTS = {
    "collect": Path("scripts/mujoco/collect_m20_mujoco_vla.py"),
    "audit": Path("scripts/mujoco/audit_m20_mujoco_vla_distribution.py"),
    "convert": Path("scripts/convert_m20_objectnav_to_lerobot.py"),
    "train": Path("scripts/mujoco/train_m20_mujoco_vla.py"),
    "eval": Path("scripts/mujoco/evaluate_m20_mujoco_hidden_search.py"),
    "play": Path("scripts/mujoco/play_m20_mujoco_vla.py"),
}


def run_compatibility_script(
    kind: str,
    forwarded_args: Sequence[str],
    context: RunContext,
) -> dict:
    """Run one existing canonical script while streaming output to the run log."""
    relative_script = COMPATIBILITY_SCRIPTS[kind]
    script = workspace_root() / relative_script
    if not script.is_file():
        raise FileNotFoundError(f"Compatibility entry point is missing: {script}")
    args = list(forwarded_args)
    if args[:1] == ["--"]:
        args = args[1:]
    command = [sys.executable, str(script), *args]
    stdout_path = context.path / "logs" / "stdout.log"
    stderr_path = context.path / "logs" / "stderr.log"
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=workspace_root(), env=environment, stdout=stdout, stderr=stderr, check=False)
    return {
        "schema": "m20pro_vla_compat_run_v1",
        "workflow": kind,
        "compatibility_script": str(relative_script),
        "forwarded_args": args,
        "exit_code": process.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "business_logic_migrated": False,
    }


__all__ = ["COMPATIBILITY_SCRIPTS", "run_compatibility_script"]

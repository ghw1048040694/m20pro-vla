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
    "train": Path("scripts/mujoco/train_m20_mujoco_vla.py"),
    "eval": Path("scripts/mujoco/evaluate_m20_mujoco_hidden_search.py"),
    "smolvla-eval": Path("scripts/mujoco/play_m20_smolvla.py"),
    "play": Path("scripts/mujoco/play_m20_mujoco_vla.py"),
}


def run_compatibility_script(
    kind: str,
    forwarded_args: Sequence[str],
    context: RunContext,
    *,
    conda_env: str | None = None,
    hf_offline: bool = False,
) -> dict:
    """Run one existing canonical script while streaming output to the run log.

    ``conda_env`` names the environment that owns the script's heavy imports.
    The SmolVLA closed loop needs LeRobot, which the MuJoCo environment does not
    ship, so it can only run if that loop is dispatched into the SmolVLA
    environment instead of the environment that launched the unified command.

    ``hf_offline`` pins the Hugging Face cache. The WSL2 host has no route to
    huggingface.co, so an online lookup does not fail fast - it stalls through
    the full retry ladder before falling back to local files.
    """
    relative_script = COMPATIBILITY_SCRIPTS[kind]
    script = workspace_root() / relative_script
    if not script.is_file():
        raise FileNotFoundError(f"Compatibility entry point is missing: {script}")
    args = list(forwarded_args)
    if args[:1] == ["--"]:
        args = args[1:]
    stdout_path = context.path / "logs" / "stdout.log"
    stderr_path = context.path / "logs" / "stderr.log"
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if hf_offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    executable = sys.executable
    if conda_env:
        from ..training.smolvla import conda_env_executable

        executable = str(conda_env_executable(conda_env, "python"))
        if not Path(executable).is_file():
            raise FileNotFoundError(f"SmolVLA environment interpreter not found: {executable}")
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (str(workspace_root() / "src"), environment.get("PYTHONPATH", ""))
            if item
        )
    command = [executable, str(script), *args]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=workspace_root(), env=environment, stdout=stdout, stderr=stderr, check=False)
    return {
        "schema": "m20pro_vla_compat_run_v1",
        "workflow": kind,
        "compatibility_script": str(relative_script),
        "forwarded_args": args,
        "interpreter": executable,
        "conda_env": conda_env,
        "hf_offline": hf_offline,
        "exit_code": process.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "business_logic_migrated": False,
    }


__all__ = ["COMPATIBILITY_SCRIPTS", "run_compatibility_script"]

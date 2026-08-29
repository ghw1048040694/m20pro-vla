"""Reproducible run directories for all M20 VLA commands.

The runtime tree is intentionally ignored by git.  Source code and reviewed
configuration stay in the repository, while this module records the exact
command, configuration snapshot, status, and machine-readable result for each
local run.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_runtime_root() -> Path:
    return Path(os.environ.get("M20PRO_VLA_DATA_ROOT", workspace_root() / ".runtime"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass
class RunContext:
    """Filesystem context shared by one command invocation."""

    run_id: str
    kind: str
    path: Path
    command: tuple[str, ...]
    created_at: str

    @classmethod
    def create(
        cls,
        kind: str,
        command: Sequence[str],
        *,
        runtime_root: Path | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> "RunContext":
        root = (runtime_root or default_runtime_root()).expanduser().resolve()
        runs_root = root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        base_id = run_id or f"M20-{now.astimezone().strftime('%Y%m%d-%H%M%S')}"
        candidate = base_id
        suffix = 2
        while (runs_root / candidate).exists():
            candidate = f"{base_id}-{suffix:02d}"
            suffix += 1
        run_path = runs_root / candidate
        run_path.mkdir(parents=True)
        for directory in ("videos", "checkpoints", "dataset", "logs"):
            (run_path / directory).mkdir()
        created_at = now.isoformat()
        context = cls(candidate, kind, run_path, tuple(str(x) for x in command), created_at)
        snapshot = {
            "schema": "m20pro_vla_run_v1",
            "run_id": candidate,
            "kind": kind,
            "created_at": created_at,
            "git_revision": _git_revision(workspace_root()),
            "command": list(context.command),
            "config": dict(config or {}),
        }
        context._write_json("config.json", snapshot)
        (run_path / "command.txt").write_text(" ".join(context.command) + "\n", encoding="utf-8")
        context._write_json("status.json", {"status": "running", "updated_at": created_at})
        return context

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.path / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
        return path

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        payload = {"schema": "m20pro_vla_summary_v1", "run_id": self.run_id, **dict(summary)}
        return self._write_json("summary.json", payload)

    def finish(self, success: bool, *, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "status": "success" if success else "failed",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["error"] = error
        self._write_json("status.json", payload)


__all__ = ["RunContext", "default_runtime_root", "workspace_root"]

"""Small public CLI for the M20 Pro MuJoCo VLA package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .runtime import RunContext, default_runtime_root


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", type=Path, default=default_runtime_root())
    parser.add_argument("--run-id", help="Explicit run directory name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m20pro-vla")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("doctor", "Check the public package and optional dependencies"), ("smoke", "Render the RGB/LiDAR/proprioception observation contract"), ("low-level-gate", "Run the reusable locomotion controller gate"), ("report", "Summarize local runtime runs")):
        command = sub.add_parser(name, help=help_text)
        _runtime_args(command)
        if name == "smoke":
            command.add_argument("--width", type=int, default=320)
            command.add_argument("--height", type=int, default=180)
        elif name == "low-level-gate":
            command.add_argument("--warmup-steps", type=int, default=100)
            command.add_argument("--forward-steps", type=int, default=220)
            command.add_argument("--stop-steps", type=int, default=100)
            command.add_argument("--turn-steps", type=int, default=180)
            command.add_argument("--turn-command", type=float, default=0.10)
        elif name == "report":
            command.add_argument("--limit", type=int, default=20)
    return parser


def _dependency(name: str) -> dict[str, Any]:
    found = importlib.util.find_spec(name) is not None
    result: dict[str, Any] = {"installed": found}
    if found:
        try:
            module = __import__(name)
            result["version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _print(payload: Any, compact: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2, default=str))


def _finish(context: RunContext | None, success: bool, report: dict[str, Any], args: argparse.Namespace, error: Exception | None = None) -> int:
    output = dict(report)
    if context is not None:
        output.update({"run_id": context.run_id, "run_dir": str(context.path)})
        context.write_summary(output)
        context.finish(success, error=None if error is None else f"{type(error).__name__}: {error}")
    _print(output, args.json)
    if error is not None:
        print(f"m20pro-vla: {type(error).__name__}: {error}", file=sys.stderr)
    return 0 if success else 1


def _context(args: argparse.Namespace, kind: str, config: dict[str, Any]) -> RunContext | None:
    if args.dry_run:
        return None
    return RunContext.create(kind, [sys.executable, *sys.argv], runtime_root=args.runtime_root, run_id=args.run_id, config=config)


def command_doctor(args: argparse.Namespace) -> int:
    try:
        from .low_level import LOW_LEVEL_CONTRACT
        from .sim.mujoco import ASSET
        report = {"schema": "m20pro_vla_doctor_v2", "package": {"version": __version__}, "python": {"version": platform.python_version(), "required": ">=3.11"}, "dependencies": {name: _dependency(name) for name in ("mujoco", "numpy", "PIL", "torch")}, "prepared_asset": {"present": ASSET.is_file()}, "contract": LOW_LEVEL_CONTRACT, "public_snapshot": True}
        ready = report["dependencies"]["mujoco"]["installed"] and report["dependencies"]["numpy"]["installed"]
        if args.dry_run:
            report["dry_run"] = True
        return _finish(_context(args, "doctor", {}), bool(ready), report, args)
    except Exception as exc:
        return _finish(None, False, {"schema": "m20pro_vla_doctor_v2"}, args, exc)


def command_smoke(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        if args.width <= 0 or args.height <= 0:
            raise ValueError("width and height must be positive")
        if args.dry_run:
            return _finish(None, True, {"schema": "m20pro_vla_smoke_plan_v2", "width": args.width, "height": args.height, "dry_run": True}, args)
        context = _context(args, "smoke", {"width": args.width, "height": args.height})
        assert context is not None
        from .sim.observation_smoke import run_observation_smoke
        report = run_observation_smoke(context.path / "observations", width=args.width, height=args.height)
        return _finish(context, True, report, args)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_observation_smoke_v2"}, args, exc)


def command_low_level_gate(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        config = {key: value for key, value in vars(args).items() if key in {"warmup_steps", "forward_steps", "stop_steps", "turn_steps", "turn_command"}}
        if args.dry_run:
            return _finish(None, True, {"schema": "m20pro_vla_low_level_gate_plan_v2", "config": config, "dry_run": True}, args)
        context = _context(args, "low-level-gate", config)
        from .low_level.gate import run_low_level_gate
        report = run_low_level_gate(**config)
        return _finish(context, bool(report.get("eligible_for_flat_vla_execution", False)), report, args)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_low_level_gate_v2"}, args, exc)


def command_report(args: argparse.Namespace) -> int:
    try:
        root = args.runtime_root.expanduser().resolve() / "runs"
        rows = []
        paths = sorted(root.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True) if root.is_dir() else ()
        for path in paths:
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            if len(rows) >= max(1, args.limit):
                break
        return _finish(_context(args, "report", {"limit": args.limit}), True, {"schema": "m20pro_vla_report_v2", "count": len(rows), "runs": rows}, args)
    except Exception as exc:
        return _finish(None, False, {"schema": "m20pro_vla_report_v2"}, args, exc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command == "low-level-gate":
        return command_low_level_gate(args)
    return command_report(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified command-line entry point for the M20 VLA system.

The first migration stage owns the low-risk lifecycle gates directly inside
the installable package.  Historical collection/training scripts remain
available as compatibility entry points until their logic has been moved into
package APIs and verified against the same contracts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

from . import __version__
from .runtime import RunContext, default_runtime_root
from .runtime.compat import COMPATIBILITY_SCRIPTS, run_compatibility_script


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", type=Path, default=default_runtime_root())
    parser.add_argument("--run-id", help="Explicit run directory name; defaults to a timestamped ID")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writing runtime artifacts")
    parser.add_argument("--json", action="store_true", help="Print compact JSON instead of formatted JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m20pro-vla",
        description="M20 Pro VLA lifecycle CLI; source APIs own simulation and control.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check Python, dependencies, assets, and contracts")
    _runtime_args(doctor)

    prepare = subparsers.add_parser("prepare", help="Validate the prepared MuJoCo asset and versioned configs")
    _runtime_args(prepare)

    smoke = subparsers.add_parser("smoke", help="Run the RGB/LiDAR/proprioception observation smoke test")
    _runtime_args(smoke)
    smoke.add_argument("--width", type=int, default=320)
    smoke.add_argument("--height", type=int, default=180)

    gate = subparsers.add_parser("low-level-gate", help="Run the reusable bottom-controller regression gate")
    _runtime_args(gate)
    gate.add_argument("--warmup-steps", type=int, default=100)
    gate.add_argument("--forward-steps", type=int, default=220)
    gate.add_argument("--stop-steps", type=int, default=100)
    gate.add_argument("--turn-steps", type=int, default=180)
    gate.add_argument("--turn-command", type=float, default=0.10)

    for name in COMPATIBILITY_SCRIPTS:
        workflow = subparsers.add_parser(
            name,
            help=f"Run the {name} workflow through its recorded compatibility entry point",
        )
        _runtime_args(workflow)
        workflow.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments forwarded to the canonical workflow entry point")

    report = subparsers.add_parser("report", help="Summarize completed unified runs")
    _runtime_args(report)
    report.add_argument("--limit", type=int, default=20)
    return parser


def _print(payload: Any, compact: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2, default=str))


def _context(args: argparse.Namespace, kind: str, config: dict[str, Any]) -> RunContext | None:
    if args.dry_run:
        return None
    return RunContext.create(
        kind,
        [sys.executable, *sys.argv],
        runtime_root=args.runtime_root,
        run_id=args.run_id,
        config=config,
    )


def _finish(
    context: RunContext | None,
    success: bool,
    summary: dict[str, Any],
    error: Exception | None = None,
    *,
    compact: bool = False,
) -> int:
    output = dict(summary)
    if context is not None:
        output.update({"run_id": context.run_id, "run_dir": str(context.path)})
        context.write_summary(output)
        context.finish(success, error=None if error is None else f"{type(error).__name__}: {error}")
    _print(output, compact)
    if error is not None and not isinstance(error, SystemExit):
        print(f"m20pro-vla: {type(error).__name__}: {error}", file=sys.stderr)
    return 0 if success else 1


def _dependency(name: str) -> dict[str, Any]:
    found = importlib.util.find_spec(name) is not None
    result: dict[str, Any] = {"installed": found}
    if found:
        try:
            module = __import__(name)
            result["version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _doctor_report() -> dict[str, Any]:
    from .low_level import LOW_LEVEL_CONTRACT
    from .sim.mujoco import ASSET

    root = Path(__file__).resolve().parents[2]
    configs = sorted((root / "configs").glob("*.yaml")) + sorted((root / "configs").glob("*.json"))
    checks = {
        "python": {"version": platform.python_version(), "required": ">=3.11"},
        "package": {"version": __version__, "source_root": str(root / "src/m20pro_vla")},
        "dependencies": {
            "mujoco": _dependency("mujoco"),
            "numpy": _dependency("numpy"),
            "PIL": _dependency("PIL"),
            "torch_optional": _dependency("torch"),
        },
        "asset": {"path": str(ASSET), "present": ASSET.is_file()},
        "configs": {"count": len(configs), "versioned": all("_v" in path.stem for path in configs)},
        "contracts": {"low_level": LOW_LEVEL_CONTRACT, "body_command": ["forward", "lateral", "yaw", "stop"]},
    }
    required_ok = (
        sys.version_info >= (3, 11)
        and checks["dependencies"]["mujoco"]["installed"]
        and checks["dependencies"]["numpy"]["installed"]
        and checks["asset"]["present"]
        and checks["configs"]["versioned"]
    )
    return {"schema": "m20pro_vla_doctor_v1", "ready_for_smoke": bool(required_ok), "checks": checks}


def command_doctor(args: argparse.Namespace) -> int:
    try:
        report = _doctor_report()
        if args.dry_run:
            report["dry_run"] = True
            return _finish(None, bool(report["ready_for_smoke"]), report, compact=args.json)
        context = _context(args, "doctor", {})
        return _finish(context, bool(report["ready_for_smoke"]), report, compact=args.json)
    except Exception as exc:
        return _finish(None, False, {"schema": "m20pro_vla_doctor_v1"}, exc, compact=getattr(args, "json", False))


def command_prepare(args: argparse.Namespace) -> int:
    try:
        report = _doctor_report()
        report = {
            "schema": "m20pro_vla_prepare_v1",
            "asset_ready": report["checks"]["asset"]["present"],
            "config_count": report["checks"]["configs"]["count"],
            "next": "m20pro-vla smoke" if report["ready_for_smoke"] else "build the MuJoCo asset first",
        }
        if args.dry_run:
            report["dry_run"] = True
            return _finish(None, bool(report["asset_ready"]), report, compact=args.json)
        context = _context(args, "prepare", {})
        return _finish(context, bool(report["asset_ready"]), report, compact=args.json)
    except Exception as exc:
        return _finish(None, False, {"schema": "m20pro_vla_prepare_v1"}, exc, compact=getattr(args, "json", False))


def command_smoke(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        config = {"width": args.width, "height": args.height}
        if args.dry_run:
            return _finish(None, True, {"schema": "m20pro_vla_smoke_plan_v1", "config": config, "dry_run": True}, compact=args.json)
        context = _context(args, "smoke", config)
        from .sim.observation_smoke import run_observation_smoke

        report = run_observation_smoke(context.path / "observations", width=args.width, height=args.height)
        return _finish(context, True, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_observation_smoke_v1"}, exc, compact=getattr(args, "json", False))


def command_low_level_gate(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        config = {
            "warmup_steps": args.warmup_steps,
            "forward_steps": args.forward_steps,
            "stop_steps": args.stop_steps,
            "turn_steps": args.turn_steps,
            "turn_command": args.turn_command,
        }
        if args.dry_run:
            return _finish(None, True, {"schema": "m20pro_vla_low_level_gate_plan_v1", "config": config, "dry_run": True}, compact=args.json)
        context = _context(args, "low-level-gate", config)
        from .low_level.gate import run_low_level_gate

        report = run_low_level_gate(**config)
        (context.path / "low_level_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return _finish(context, bool(report["eligible_for_flat_vla_execution"]), report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_low_level_gate_v1"}, exc, compact=getattr(args, "json", False))


def command_compatibility(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        config = {"forwarded_args": list(args.extra), "compatibility_script": str(COMPATIBILITY_SCRIPTS[args.command])}
        if args.dry_run:
            return _finish(
                None,
                True,
                {"schema": "m20pro_vla_compat_plan_v1", "workflow": args.command, "config": config, "dry_run": True},
                compact=args.json,
            )
        context = _context(args, args.command, config)
        report = run_compatibility_script(args.command, args.extra, context)
        return _finish(context, report["exit_code"] == 0, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_compat_run_v1", "workflow": args.command}, exc, compact=getattr(args, "json", False))


def command_report(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        runs_root = args.runtime_root.expanduser().resolve() / "runs"
        rows = []
        if runs_root.is_dir():
            for summary_path in sorted(runs_root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("schema") == "m20pro_vla_report_v1":
                    continue
                rows.append(payload)
                if len(rows) >= max(1, args.limit):
                    break
        report = {"schema": "m20pro_vla_report_v1", "runs_root": str(runs_root), "count": len(rows), "runs": rows}
        if args.dry_run:
            report["dry_run"] = True
            return _finish(None, True, report, compact=args.json)
        context = _context(args, "report", {"limit": args.limit})
        return _finish(context, True, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_report_v1"}, exc, compact=getattr(args, "json", False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args, unknown = parser.parse_known_args(raw_args)
    if unknown:
        if args.command in COMPATIBILITY_SCRIPTS:
            # Reconstruct the native argument tail from the original token
            # order. Argparse may split unknown option values into the
            # positional remainder in a different order, which is unsafe for
            # a transparent compatibility adapter.
            command_index = raw_args.index(args.command)
            tail = raw_args[command_index + 1 :]
            forwarded: list[str] = []
            runtime_value_options = {"--runtime-root", "--run-id"}
            index = 0
            while index < len(tail):
                token = tail[index]
                if token == "--dry-run":
                    args.dry_run = True
                    index += 1
                    continue
                if token == "--json":
                    args.json = True
                    index += 1
                    continue
                if token in runtime_value_options:
                    if index + 1 >= len(tail):
                        parser.error(f"{token} requires a value")
                    setattr(args, token[2:].replace("-", "_"), Path(tail[index + 1]) if token == "--runtime-root" else tail[index + 1])
                    index += 2
                    continue
                if any(token.startswith(option + "=") for option in runtime_value_options):
                    option, value = token.split("=", 1)
                    setattr(args, option[2:].replace("-", "_"), Path(value) if option == "--runtime-root" else value)
                    index += 1
                    continue
                forwarded.append(token)
                index += 1
            args.extra = forwarded
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command == "low-level-gate":
        return command_low_level_gate(args)
    if args.command in COMPATIBILITY_SCRIPTS:
        return command_compatibility(args)
    if args.command == "report":
        return command_report(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

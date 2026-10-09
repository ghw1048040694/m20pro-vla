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
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

from . import __version__
from .runtime import RunContext, default_runtime_root
from .runtime.compat import run_compatibility_script
from .runtime.run_context import workspace_root


SMOLVLA_STAGE_ENV_GUARD = "M20PRO_VLA_SMOLVLA_STAGE_ENV"


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

    export = subparsers.add_parser(
        "export-policy",
        help="Export a policy manifest for M20Pro-3D-Nav shadow or supervised deployment",
    )
    _runtime_args(export)
    export.add_argument("--checkpoint", type=Path, help="Checkpoint to describe or package")
    export.add_argument("--output-dir", type=Path, help="Export directory; defaults to the current run directory")
    export.add_argument("--policy-id", help="Stable deployment label; defaults to the checkpoint parent directory")
    export.add_argument("--real-contract", type=Path, default=Path("configs/m20pro_real_vla_deploy_contract_v1.yaml"))
    export.add_argument("--copy-checkpoint", action="store_true", help="Copy checkpoint bytes into the export directory")
    export.add_argument("--image-width", type=int, default=160)
    export.add_argument("--image-height", type=int, default=96)
    export.add_argument("--target-runtime", choices=("shadow", "dddmr-subgoal", "supervised-velocity"), default="shadow")
    export.add_argument("--notes", default="")

    experiment = subparsers.add_parser(
        "experiment",
        help="Run every experiment stage from one versioned configuration file",
    )
    _runtime_args(experiment)
    experiment.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.json"),
    )
    experiment.add_argument(
        "--stage",
        choices=(
            "plan",
            "collect",
            "curate",
            "audit",
            "convert",
            "train",
            "train-smolvla",
            "smolvla-eval",
            "collect-recovery",
            "eval",
            "gate",
        ),
        default="plan",
    )

    report = subparsers.add_parser("report", help="Summarize completed unified runs")
    _runtime_args(report)
    report.add_argument("--limit", type=int, default=20)
    return parser


def _print(payload: Any, compact: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2, default=str))


def _context(args: argparse.Namespace, kind: str, config: dict[str, Any]) -> RunContext | None:
    if args.dry_run:
        return None
    if getattr(args, "command", None) == "experiment":
        config = {**config, "experiment": json.loads(args.config.read_text(encoding="utf-8"))}
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


def command_export_policy(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        from .deploy import create_policy_export, policy_export_plan

        if args.dry_run:
            report = policy_export_plan(
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                policy_id=args.policy_id,
                real_contract=args.real_contract,
                copy_checkpoint=args.copy_checkpoint,
                image_width=args.image_width,
                image_height=args.image_height,
                target_runtime=args.target_runtime,
            )
            report["dry_run"] = True
            return _finish(None, True, report, compact=args.json)
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --dry-run is used")
        config = {
            "checkpoint": str(args.checkpoint),
            "output_dir": None if args.output_dir is None else str(args.output_dir),
            "policy_id": args.policy_id,
            "real_contract": str(args.real_contract),
            "copy_checkpoint": bool(args.copy_checkpoint),
            "image_width": args.image_width,
            "image_height": args.image_height,
            "target_runtime": args.target_runtime,
        }
        context = _context(args, "export-policy", config)
        assert context is not None
        output_dir = args.output_dir or (context.path / "policy_export")
        report = create_policy_export(
            checkpoint=args.checkpoint,
            output_dir=output_dir,
            policy_id=args.policy_id,
            real_contract=args.real_contract,
            copy_checkpoint=args.copy_checkpoint,
            image_width=args.image_width,
            image_height=args.image_height,
            target_runtime=args.target_runtime,
            notes=args.notes,
        )
        return _finish(context, True, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "m20pro_vla_policy_export_result_v1"}, exc, compact=getattr(args, "json", False))


def _dispatch_lerobot_stage_environment(args: argparse.Namespace) -> int | None:
    """Re-run a LeRobot-owned stage inside the configured environment.

    Dataset conversion and SmolVLA training both import LeRobot in process.
    LeRobot is intentionally installed in its own Conda environment instead of
    the MuJoCo collection environment, so the whole command must be dispatched
    before either stage imports its heavy dependencies.
    """
    if os.environ.get(SMOLVLA_STAGE_ENV_GUARD) == "1":
        return None
    from .runtime.experiment import load_experiment_config
    from .training.smolvla import conda_env_executable

    settings = load_experiment_config(args.config)["smolvla"]
    executable = conda_env_executable(settings.get("conda_env", "lerobot"), "python")
    if not executable.is_file():
        raise FileNotFoundError(f"SmolVLA environment interpreter not found: {executable}")
    if Path(sys.executable).resolve() == executable.resolve():
        return None
    environment = os.environ.copy()
    environment[SMOLVLA_STAGE_ENV_GUARD] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(workspace_root() / "src"), environment.get("PYTHONPATH", ""))
        if item
    )
    command = [str(executable), "-m", "m20pro_vla.cli", *sys.argv[1:]]
    print(json.dumps({"lerobot_stage_dispatch": command[0], "reason": "environment owns LeRobot and Hugging Face Hub"}))
    process = subprocess.run(command, cwd=workspace_root(), env=environment, check=False)
    return process.returncode


def command_experiment(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        from .runtime.experiment import build_experiment_plan, load_experiment_config, experiment_low_level_environment
        from .training import evaluate_training_candidate

        plan = build_experiment_plan(args.config)
        stage = args.stage
        if args.dry_run or stage == "plan":
            plan["dry_run"] = bool(args.dry_run)
            return _finish(None, True, plan, compact=args.json)
        if stage == "convert" and plan["stages"]["train-smolvla"]["dataset_backend"] == "raw":
            return _finish(None, True, {"stage": stage, "skipped": True, "dataset_backend": "raw",
                "reason": "Raw RGB trajectories are read directly; conversion is not part of this training plan."}, compact=args.json)
        os.environ.update(experiment_low_level_environment(load_experiment_config(args.config)))
        if stage in {"convert", "train-smolvla"}:
            dispatch_code = _dispatch_lerobot_stage_environment(args)
            if dispatch_code is not None:
                return dispatch_code
        if stage == "audit":
            report = plan["training"]["distribution_audit"]
            context = _context(
                args,
                "experiment-audit",
                {"experiment_config": str(args.config), "experiment_id": plan["experiment_id"], "stage": stage},
            )
            return _finish(context, bool(report["passed"]), report, compact=args.json)

        if stage == "curate":
            from .data import M20MuJoCoDistributionThresholds, curate_episode_view

            config = load_experiment_config(args.config)
            paths = config["paths"]
            context = _context(
                args,
                "experiment-curate",
                {"experiment_config": str(args.config), "experiment_id": plan["experiment_id"], "stage": stage},
            )
            report = curate_episode_view(
                source=Path(paths["raw_dataset"]),
                recovery=Path(paths["recovery_dataset"]),
                output=Path(paths["dataset"]),
                thresholds=M20MuJoCoDistributionThresholds(**config["dataset"]),
            )
            return _finish(context, True, report, compact=args.json)

        if stage == "convert":
            from .data.conversion_cache import convert_m20_incremental

            config = load_experiment_config(args.config)
            settings = config["smolvla"]
            context = _context(
                args,
                "experiment-convert",
                {"experiment_config": str(args.config), "experiment_id": plan["experiment_id"], "stage": stage},
            )
            report = convert_m20_incremental(
                source=Path(config["paths"]["dataset"]),
                output=Path(config["paths"]["lerobot_dataset"]),
                cache_root=Path(settings.get("conversion_cache_dir", ".runtime/cache/lerobot_conversion")),
                workers=int(settings.get("conversion_workers", 2)),
                repo_id=str(settings["repo_id"]),
                source_fps=int(settings["source_fps"]),
                frame_stride=int(settings["frame_stride"]),
                terminal_stop_repeat=int(settings.get("terminal_stop_repeat", 1)),
                recovery_terminal_stop_repeat=int(settings.get("recovery_terminal_stop_repeat", 1)),
                use_videos=bool(settings.get("use_videos", True)),
                vcodec=str(settings.get("vcodec", "h264")),
            )
            return _finish(context, True, report, compact=args.json)

        context = _context(
            args,
            f"experiment-{stage}",
            {"experiment_config": str(args.config), "experiment_id": plan["experiment_id"], "stage": stage},
        )
        assert context is not None
        if stage == "train-smolvla":
            from .training import run_smolvla_training

            stage_plan = plan["stages"][stage]
            if not plan["training"]["distribution_audit"]["passed"]:
                raise RuntimeError("Source dataset failed distribution/quality audit; collect or filter episodes before SmolVLA training")
            if not stage_plan["dataset_ready"]:
                raise FileNotFoundError(f"SmolVLA training dataset is missing: {stage_plan['dataset']}")
            if stage_plan["output_already_exists"] and not stage_plan.get("resume_checkpoint"):
                raise FileExistsError(f"SmolVLA output already exists: {stage_plan['output']}")
            config = load_experiment_config(args.config)
            with (context.path / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, (
                context.path / "logs" / "stderr.log"
            ).open("a", encoding="utf-8") as stderr:
                report = run_smolvla_training(config, stdout=stdout, stderr=stderr)
            report["experiment_id"] = plan["experiment_id"]
            report["experiment_config"] = str(args.config)
            return _finish(context, report["exit_code"] == 0, report, compact=args.json)
        if stage == "gate":
            gate = plan["stages"]["gate"]
            report = evaluate_training_candidate(
                checkpoint=Path(gate["checkpoint"]),
                eval_summary=Path(gate["eval_summary"]),
                standard_path=args.config,
            )
            return _finish(context, bool(report["eligible_for_shadow"]), report, compact=args.json)

        stage_plan = plan["stages"][stage]
        if stage == "train":
            if not stage_plan["eligible"]:
                return _finish(context, False, plan["training"], compact=args.json)
            if stage_plan["output_already_exists"]:
                raise FileExistsError(
                    f"Checkpoint output already exists: {plan['paths']['checkpoint_dir']}; use a new experiment_id/path"
                )
        if stage == "eval" and not stage_plan["checkpoint_exists"]:
            raise FileNotFoundError(f"Experiment checkpoint does not exist: {stage_plan['args'][1]}")
        if stage in {"smolvla-eval", "collect-recovery"} and not stage_plan["checkpoint_exists"]:
            raise FileNotFoundError(f"SmolVLA checkpoint does not exist: {stage_plan['checkpoint']}")
        conda_env = None
        hf_offline = False
        if stage in {"smolvla-eval", "collect-recovery"}:
            smolvla_settings = load_experiment_config(args.config)["smolvla"]
            conda_env = str(smolvla_settings.get("conda_env", "lerobot"))
            hf_offline = bool(smolvla_settings.get("offline", False))
        report = run_compatibility_script(
            stage_plan["workflow"],
            stage_plan["args"],
            context,
            conda_env=conda_env,
            hf_offline=hf_offline,
        )
        report["experiment_id"] = plan["experiment_id"]
        report["experiment_config"] = str(args.config)
        return _finish(context, report["exit_code"] == 0, report, compact=args.json)
    except Exception as exc:
        return _finish(
            context,
            False,
            {"schema": "m20pro_vla_experiment_result_v1", "stage": getattr(args, "stage", None)},
            exc,
            compact=getattr(args, "json", False),
        )


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
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command == "low-level-gate":
        return command_low_level_gate(args)
    if args.command == "export-policy":
        return command_export_policy(args)
    if args.command == "experiment":
        return command_experiment(args)
    if args.command == "report":
        return command_report(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

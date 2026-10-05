"""One-config experiment planning for collection, training, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from m20pro_vla.training import build_standard_training_plan


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.json")


def load_experiment_config(path: Path = DEFAULT_EXPERIMENT_CONFIG) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "m20pro_vla_experiment_v1":
        raise ValueError(f"Unsupported experiment schema: {payload.get('schema')!r}")
    required = {"experiment_id", "paths", "collection", "training", "evaluation", "smolvla"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Experiment config is missing fields: {missing}")
    return payload


def _append_options(args: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            for item in value:
                args.extend((flag, str(item)))
        else:
            args.extend((flag, str(value)))


def smolvla_eval_checkpoint(config: dict[str, Any]) -> Path:
    """Resolve the SmolVLA checkpoint that the closed-loop stage must evaluate.

    Defaulting to ``smolvla.steps`` keeps the evaluation pinned to the final
    checkpoint of the configured training run, so raising the step budget cannot
    silently leave the closed-loop stage pointing at an older checkpoint. The
    trainer writes the loadable policy into a ``pretrained_model`` subdirectory;
    pointing at the step directory itself resolves to an empty config and the
    policy fails to load.
    """
    settings = config.get("smolvla_evaluation") or {}
    step = int(settings.get("checkpoint_step", config["smolvla"]["steps"]))
    return (
        Path(config["paths"]["smolvla_checkpoint_dir"])
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )


def experiment_stage_args(config: dict[str, Any], stage: str) -> list[str]:
    paths = config["paths"]
    if stage == "collect":
        args: list[str] = []
        _append_options(args, config["collection"])
        args.extend(("--output-dir", str(paths.get("raw_dataset", paths["dataset"]))))
        if paths.get("collection_videos"):
            args.extend(("--video-dir", str(paths["collection_videos"])))
        return args
    if stage in {"smolvla-eval", "collect-recovery"}:
        settings = dict(config.get("smolvla_evaluation") or {})
        settings.pop("checkpoint_step", None)
        if stage == "collect-recovery":
            settings.update(config.get("recovery_collection") or {})
        output_dir = Path(paths["recovery_eval_dir"] if stage == "collect-recovery" else paths["smolvla_eval_dir"])
        args = [
            "--checkpoint", str(smolvla_eval_checkpoint(config)),
            "--episodes-dir", str(settings.pop("episodes_dir", paths["dataset"])),
            "--output-dir", str(output_dir),
            "--summary", str(output_dir / "fleet_summary.json"),
        ]
        if stage == "collect-recovery":
            args.extend(("--recovery-output-dir", str(paths["recovery_dataset"])))
        _append_options(args, settings)
        return args
    if stage == "eval":
        args = [
            "--checkpoint", str(Path(paths["checkpoint_dir"]) / "best.pt"),
            "--output-dir", str(paths["eval_dir"]),
        ]
        _append_options(args, config["evaluation"])
        return args
    raise ValueError(f"Stage {stage!r} does not use compatibility arguments")


def build_experiment_plan(config_path: Path = DEFAULT_EXPERIMENT_CONFIG) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    paths = config["paths"]
    init_value = paths.get("init_checkpoint")
    training_plan = build_standard_training_plan(
        dataset=Path(paths["dataset"]),
        output_dir=Path(paths["checkpoint_dir"]),
        standard_path=Path(config_path),
        init_checkpoint=Path(init_value) if init_value else None,
    )
    checkpoint = Path(paths["checkpoint_dir"]) / "best.pt"
    eval_summary = Path(paths["eval_dir"]) / "summary.json"
    lerobot_dataset = Path(paths["lerobot_dataset"])
    smolvla_output = Path(paths["smolvla_checkpoint_dir"])
    return {
        "schema": "m20pro_vla_experiment_plan_v1",
        "experiment_id": config["experiment_id"],
        "config": str(Path(config_path)),
        "paths": paths,
        "training": training_plan,
        "stages": {
            "convert": {
                "source": str(paths["dataset"]),
                "output": str(lerobot_dataset),
                "output_already_exists": lerobot_dataset.exists(),
            },
            "collect": {"workflow": "collect", "args": experiment_stage_args(config, "collect")},
            "curate": {
                "source": str(paths.get("raw_dataset", paths["dataset"])),
                "recovery": str(paths["recovery_dataset"]),
                "output": str(paths["dataset"]),
                "output_already_exists": Path(paths["dataset"]).exists(),
            },
            "audit": {"passed": bool(training_plan["distribution_audit"]["passed"])},
            "train": {
                "workflow": "train",
                "args": training_plan["forwarded_args"],
                "eligible": bool(training_plan["eligible_to_train"]),
                "output_already_exists": checkpoint.exists(),
            },
            "train-smolvla": {
                "dataset_ready": (lerobot_dataset / "meta" / "info.json").is_file(),
                "output": str(smolvla_output),
                "output_already_exists": smolvla_output.exists(),
            },
            "smolvla-eval": {
                "workflow": "smolvla-eval",
                "args": experiment_stage_args(config, "smolvla-eval"),
                "checkpoint": str(smolvla_eval_checkpoint(config)),
                "checkpoint_exists": (smolvla_eval_checkpoint(config) / "config.json").is_file(),
                "output": str(Path(paths["smolvla_eval_dir"]) / "fleet_summary.json"),
            },
            "collect-recovery": {
                "workflow": "smolvla-eval",
                "args": experiment_stage_args(config, "collect-recovery"),
                "checkpoint": str(smolvla_eval_checkpoint(config)),
                "checkpoint_exists": (smolvla_eval_checkpoint(config) / "config.json").is_file(),
                "output": str(Path(paths["recovery_eval_dir"]) / "fleet_summary.json"),
            },
            "eval": {
                "workflow": "eval",
                "args": experiment_stage_args(config, "eval"),
                "checkpoint_exists": checkpoint.is_file(),
            },
            "gate": {"checkpoint": str(checkpoint), "eval_summary": str(eval_summary)},
        },
    }


__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG",
    "build_experiment_plan",
    "experiment_stage_args",
    "load_experiment_config",
    "smolvla_eval_checkpoint",
]

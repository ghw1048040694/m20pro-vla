"""Official LeRobot SmolVLA training backend for the unified experiment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, TextIO


def prepare_smolvla_training_source(config: dict[str, Any]) -> Path:
    """Create a dataset-adapted local policy source from the cached SmolVLA base."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import dataset_to_policy_features
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    paths = config["paths"]
    settings = config["smolvla"]
    dataset_root = Path(paths["lerobot_dataset"])
    prepared = Path(paths["smolvla_prepared_base"])
    prepared.mkdir(parents=True, exist_ok=True)

    init_policy = settings.get("init_policy")
    if init_policy:
        snapshot = Path(init_policy)
        if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError(f"SmolVLA init policy is incomplete: {snapshot}")
    else:
        snapshot = Path(
            snapshot_download(
                str(settings["base_model"]),
                local_files_only=bool(settings.get("offline", False)),
            )
        )
    metadata = LeRobotDatasetMetadata(str(settings["repo_id"]), root=dataset_root)
    features = dataset_to_policy_features(metadata.features)
    policy_config = PreTrainedConfig.from_pretrained(snapshot)
    policy_config.input_features = {
        key: feature for key, feature in features.items() if feature.type is not FeatureType.ACTION
    }
    policy_config.output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    policy_config.pretrained_path = None
    policy_config.push_to_hub = False
    policy_config.save_pretrained(prepared)

    source_weights = (snapshot / "model.safetensors").resolve()
    target_weights = prepared / "model.safetensors"
    if target_weights.exists() or target_weights.is_symlink():
        target_weights.unlink()
    try:
        os.link(source_weights, target_weights)
    except OSError:
        shutil.copy2(source_weights, target_weights)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy_config,
        dataset_stats=metadata.stats,
    )
    preprocessor.save_pretrained(prepared)
    postprocessor.save_pretrained(prepared)
    return prepared


CONDA_ENVS_ROOT = Path("/home/ubuntu/miniconda3/envs")


def conda_env_executable(conda_env: str, name: str) -> Path:
    """Resolve one binary from a named conda environment.

    The SmolVLA stack (LeRobot + Transformers) lives in its own environment, so
    every stage that has to import it - training and the learner-only closed
    loop alike - must call into that environment explicitly instead of assuming
    the MuJoCo environment can import it.
    """
    return CONDA_ENVS_ROOT / str(conda_env) / "bin" / name


def smolvla_training_command(
    config: dict[str, Any],
    *,
    policy_path: str | Path | None = None,
) -> list[str]:
    paths = config["paths"]
    settings = config["smolvla"]
    executable = conda_env_executable(
        settings.get("conda_env", "lerobot"),
        "lerobot-train",
    )
    if settings.get("resume_checkpoint"):
        checkpoint, _ = validate_smolvla_resume(config)
        # LeRobot restores the policy and optimizer from this saved config.
        # Passing policy.path here would bypass its resume checkpoint handling.
        return [str(executable), f"--config_path={checkpoint / 'pretrained_model' / 'train_config.json'}", "--resume=true"]
    return [
        str(executable),
        f"--policy.path={policy_path or settings['base_model']}",
        f"--dataset.repo_id={settings['repo_id']}",
        f"--dataset.root={paths['lerobot_dataset']}",
        f"--dataset.video_backend={settings.get('video_backend', 'pyav')}",
        f"--output_dir={paths['smolvla_checkpoint_dir']}",
        f"--job_name={config['experiment_id']}-smolvla",
        f"--batch_size={settings['batch_size']}",
        f"--steps={settings['steps']}",
        f"--num_workers={settings['num_workers']}",
        f"--log_freq={settings['log_freq']}",
        f"--save_freq={settings['save_freq']}",
        f"--seed={settings['seed']}",
        "--policy.device=cuda",
        f"--policy.use_amp={str(bool(settings.get('use_amp', False))).lower()}",
        f"--policy.use_peft={str(bool(settings.get('use_peft', False))).lower()}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]


def validate_smolvla_resume(config: dict[str, Any]) -> tuple[Path, int]:
    """Only resume a complete checkpoint from this unchanged training run."""
    settings = config["smolvla"]
    output = Path(config["paths"]["smolvla_checkpoint_dir"]).resolve()
    checkpoint = Path(settings["resume_checkpoint"]).resolve()
    if checkpoint.parent != output / "checkpoints":
        raise ValueError("Resume checkpoint must belong to the configured training output")
    required = (
        "pretrained_model/config.json", "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json", "pretrained_model/policy_preprocessor.json",
        "pretrained_model/policy_postprocessor.json", "training_state/training_step.json",
        "training_state/optimizer_param_groups.json", "training_state/optimizer_state.safetensors",
        "training_state/scheduler_state.json", "training_state/rng_state.safetensors",
    )
    for relative in required:
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete resume checkpoint: {path}")
    saved = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    for key in ("steps", "batch_size", "seed", "num_workers", "log_freq", "save_freq"):
        if saved[key] != settings[key]:
            raise ValueError(f"Resume training configuration differs: {key}")
    if Path(saved["output_dir"]).resolve() != output:
        raise ValueError("Resume output differs from saved configuration")
    if Path(saved["dataset"]["root"]).resolve() != Path(config["paths"]["lerobot_dataset"]).resolve() or saved["dataset"]["repo_id"] != settings["repo_id"]:
        raise ValueError("Resume dataset differs from saved configuration")
    step = int(json.loads((checkpoint / "training_state/training_step.json").read_text())["step"])
    if not 0 < step < int(settings["steps"]):
        raise ValueError("Resume checkpoint must precede the final training step")
    return checkpoint, step


def run_smolvla_training(
    config: dict[str, Any],
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> dict[str, Any]:
    if config["smolvla"].get("offline", False):
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    resume = bool(config["smolvla"].get("resume_checkpoint"))
    resume_step = validate_smolvla_resume(config)[1] if resume else None
    policy_path = None if resume else prepare_smolvla_training_source(config)
    command = smolvla_training_command(config, policy_path=policy_path)
    executable = Path(command[0])
    if not executable.is_file():
        raise FileNotFoundError(f"SmolVLA training executable not found: {executable}")
    dataset = Path(config["paths"]["lerobot_dataset"])
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Converted LeRobotDataset is missing: {dataset}")
    output = Path(config["paths"]["smolvla_checkpoint_dir"])
    if output.exists() and not resume:
        raise FileExistsError(f"SmolVLA output already exists: {output}")
    environment = os.environ.copy()
    if config["smolvla"].get("offline", False):
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    result = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=False)
    report = {
        "schema": "m20pro_smolvla_training_run_v1",
        "command": command,
        "exit_code": result.returncode,
        "dataset": str(dataset),
        "output_dir": str(output),
        "resume_checkpoint": config["smolvla"].get("resume_checkpoint"),
        "resume_step": resume_step,
    }
    if output.exists():
        (output / "m20_training_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "CONDA_ENVS_ROOT",
    "conda_env_executable",
    "prepare_smolvla_training_source",
    "run_smolvla_training",
    "smolvla_training_command",
    "validate_smolvla_resume",
]

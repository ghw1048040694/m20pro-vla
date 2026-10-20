"""Convert M20 MuJoCo trajectories to a standard LeRobotDataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


M20_STATE_NAMES = (
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    *(f"joint_position_{index:02d}" for index in range(16)),
    "base_linear_velocity_x",
    "base_linear_velocity_y",
    "base_linear_velocity_z",
    "base_angular_velocity_x",
    "base_angular_velocity_y",
    "base_angular_velocity_z",
    *(f"lidar_sector_min_{index:02d}" for index in range(6)),
)
M20_ACTION_NAMES = ("forward", "lateral", "yaw", "stop")


def m20_smolvla_state(proprio: np.ndarray, lidar: np.ndarray) -> np.ndarray:
    """Build the pretrained SmolVLA 32D state without world-position leakage."""
    proprio = np.asarray(proprio, dtype=np.float32)
    lidar = np.asarray(lidar, dtype=np.float32)
    if proprio.shape[-1] != 45:
        raise ValueError(f"Expected 45D proprioception, got {proprio.shape}")
    if lidar.shape[-1] != 72:
        raise ValueError(f"Expected 72D lidar, got {lidar.shape}")
    pose_and_joints = proprio[..., 3:23]
    base_velocity = proprio[..., 23:29]
    lidar_sectors = lidar.reshape(*lidar.shape[:-1], 6, 12).min(axis=-1)
    state = np.concatenate((pose_and_joints, base_velocity, lidar_sectors), axis=-1)
    if state.shape[-1] != len(M20_STATE_NAMES):
        raise AssertionError(f"Unexpected M20 SmolVLA state shape: {state.shape}")
    return state.astype(np.float32, copy=False)


def _episode_pairs(source: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for npz_path in sorted(source.glob("episode_*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing episode metadata: {json_path}")
        pairs.append((npz_path, json_path))
    if not pairs:
        raise ValueError(f"No episode NPZ files found in {source}")
    return pairs


def m20_lerobot_frame_indices(
    actions: np.ndarray,
    *,
    frame_stride: int,
    terminal_stop_repeat: int = 1,
    collection_mode: str = "search",
    recovery_terminal_stop_repeat: int = 1,
) -> list[int]:
    """Rebalance demonstration stops without inflating short correction tails."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != len(M20_ACTION_NAMES):
        raise ValueError(f"Expected Nx{len(M20_ACTION_NAMES)} actions, got {actions.shape}")
    if frame_stride <= 0 or terminal_stop_repeat <= 0 or recovery_terminal_stop_repeat <= 0:
        raise ValueError("frame_stride and terminal_stop_repeat must be positive")
    stop_repeat = recovery_terminal_stop_repeat if collection_mode == "failure_recovery" else terminal_stop_repeat
    indices: list[int] = []
    for index in range(0, len(actions), frame_stride):
        repeat = stop_repeat if float(actions[index, 3]) >= 0.5 else 1
        indices.extend([index] * repeat)
    return indices


def convert_m20_to_lerobot(
    *,
    source: Path,
    output: Path,
    repo_id: str,
    source_fps: int = 50,
    frame_stride: int = 2,
    terminal_stop_repeat: int = 1,
    recovery_terminal_stop_repeat: int = 1,
    overwrite: bool = False,
    use_videos: bool = True,
    vcodec: str = "h264",
) -> dict[str, Any]:
    """Convert paired M20 NPZ/JSON episodes using LeRobot's public writer API."""
    if frame_stride <= 0 or source_fps % frame_stride:
        raise ValueError("frame_stride must be positive and divide source_fps")
    if terminal_stop_repeat <= 0 or recovery_terminal_stop_repeat <= 0:
        raise ValueError("terminal_stop_repeat must be positive")
    pairs = _episode_pairs(Path(source))
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot dataset already exists: {output}")
        shutil.rmtree(output)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first = np.load(pairs[0][0])
    height, width = map(int, first["front_rgb"].shape[1:3])
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.images.front": {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.rear": {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(M20_STATE_NAMES),),
            "names": list(M20_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(M20_ACTION_NAMES),),
            "names": list(M20_ACTION_NAMES),
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source_fps // frame_stride,
        root=output,
        robot_type="m20pro",
        features=features,
        use_videos=use_videos,
        vcodec=vcodec,
        image_writer_threads=4,
    )

    total_frames = 0
    sampled_frames_before_repeat = 0
    stop_frames = 0
    episode_frames: list[int] = []
    frames_by_mode: dict[str, int] = {}
    stops_by_mode: dict[str, int] = {}
    for npz_path, json_path in pairs:
        arrays = np.load(npz_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        required = ("front_rgb", "rear_rgb", "lidar", "proprio", "action")
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {missing}")
        lengths = {key: len(arrays[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode arrays have different lengths in {npz_path}: {lengths}")
        states = m20_smolvla_state(arrays["proprio"], arrays["lidar"])
        task = str(metadata["task_text"])
        count = 0
        mode = str(metadata.get("collection_mode", "legacy"))
        episode_stops = 0
        frame_indices = m20_lerobot_frame_indices(
            arrays["action"],
            frame_stride=frame_stride,
            terminal_stop_repeat=terminal_stop_repeat,
            collection_mode=mode,
            recovery_terminal_stop_repeat=recovery_terminal_stop_repeat,
        )
        sampled_frames_before_repeat += len(range(0, lengths["action"], frame_stride))
        for index in frame_indices:
            dataset.add_frame(
                {
                    "observation.images.front": arrays["front_rgb"][index],
                    "observation.images.rear": arrays["rear_rgb"][index],
                    "observation.state": states[index],
                    "action": arrays["action"][index].astype(np.float32, copy=False),
                    "task": task,
                }
            )
            stop_frames += int(float(arrays["action"][index, 3]) >= 0.5)
            episode_stops += int(float(arrays["action"][index, 3]) >= 0.5)
            count += 1
        dataset.save_episode()
        total_frames += count
        episode_frames.append(count)
        frames_by_mode[mode] = frames_by_mode.get(mode, 0) + count
        stops_by_mode[mode] = stops_by_mode.get(mode, 0) + episode_stops
    dataset.finalize()
    report = {
        "schema": "m20pro_lerobot_conversion_v1",
        "source": str(source),
        "output": str(output),
        "repo_id": repo_id,
        "episodes": len(pairs),
        "frames": total_frames,
        "sampled_frames_before_terminal_repeat": sampled_frames_before_repeat,
        "terminal_stop_repeat": terminal_stop_repeat,
        "recovery_terminal_stop_repeat": recovery_terminal_stop_repeat,
        "frames_by_collection_mode": frames_by_mode,
        "stop_frames_by_collection_mode": stops_by_mode,
        "stop_frames": stop_frames,
        "stop_frame_fraction": stop_frames / total_frames,
        "episode_frames_min": min(episode_frames),
        "episode_frames_max": max(episode_frames),
        "source_fps": source_fps,
        "frame_stride": frame_stride,
        "fps": source_fps // frame_stride,
        "state_dim": len(M20_STATE_NAMES),
        "action_dim": len(M20_ACTION_NAMES),
        "camera_keys": ["observation.images.front", "observation.images.rear"],
        "use_videos": use_videos,
        "vcodec": vcodec,
    }
    (output / "m20_conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "M20_ACTION_NAMES",
    "M20_STATE_NAMES",
    "convert_m20_to_lerobot",
    "m20_lerobot_frame_indices",
    "m20_smolvla_state",
]

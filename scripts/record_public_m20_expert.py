"""Record trajectories from the released native M20 ONNX locomotion expert.

This is the first M20-specific positive dataset for the VLA project. It keeps
the publisher's 57->16 policy protocol while additionally recording the front
and rear RGB cameras, a 360-degree LiDAR scan, full state, command and task
language. The resulting HDF5 is intentionally LeRobot-like but lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from isaaclab.app import AppLauncher
from m20_indoor_scenarios import load_manifest, resolve_episode
from video_utils import finalize_h264_video


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_POLICY = (
    DATA_ROOT
    / "public_experts/m20_native/policy.onnx"
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT)))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--steps", type=int, default=500, help="50 Hz control steps per episode")
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=75,
    help="Startup frames that hold the symmetric nominal joint pose.",
)
parser.add_argument(
    "--startup-action-blend-steps",
    type=int,
    default=10,
    help="Initial recorded frames used to blend into commanded ONNX leg targets.",
)
parser.add_argument("--command-x", type=float, default=0.5)
parser.add_argument("--command-y", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--target-color", choices=["none", "red", "blue", "green"], default="none")
parser.add_argument("--target-x", type=float, default=3.0)
parser.add_argument("--target-y", type=float, default=0.0)
parser.add_argument("--initial-x", type=float, default=0.0)
parser.add_argument("--initial-y", type=float, default=0.0)
parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
parser.add_argument(
    "--initial-yaw-jitter-deg",
    type=float,
    default=0.0,
    help="Uniform per-episode yaw jitter around --initial-yaw-deg.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stop-after", type=int, default=None, help="Switch the expert command to zero after this control step.")
parser.add_argument("--stop-on-target", action="store_true", help="Stop with zero action when the simulated target enters the success radius.")
parser.add_argument("--navigate-to-target", action="store_true", help="Generate goal-directed steering demonstrations from target bearing.")
parser.add_argument(
    "--override-navigation-wheels",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use a geometric differential-wheel override (default: enabled for legacy indoor collection).",
)
parser.add_argument("--nav-forward-speed", type=float, default=0.5)
parser.add_argument("--nav-heading-gain", type=float, default=1.0)
parser.add_argument("--nav-max-yaw", type=float, default=0.5)
parser.add_argument("--nav-turn-threshold", type=float, default=1.57)
parser.add_argument("--nav-command-hold-steps", type=int, default=1)
parser.add_argument("--nav-fixed-turn-steps", type=int, default=0, help="Use a fixed turn skill before forward; zero keeps bearing control.")
parser.add_argument("--success-radius", type=float, default=0.8)
parser.add_argument(
    "--success-final-tolerance",
    type=float,
    default=0.02,
    help="Post-stop hold and final-position hysteresis after entering the success radius.",
)
parser.add_argument("--nav-slow-radius", type=float, default=1.8)
parser.add_argument("--nav-min-distance-scale", type=float, default=0.15, help="Minimum approach speed fraction before entering the success radius.")
parser.add_argument("--nav-wheel-acceleration", type=float, default=18.0, help="Maximum wheel-target slew rate in rad/s^2.")
parser.add_argument("--nav-wheel-yaw-gain", type=float, default=4.0, help="Empirical skid-steer gain applied to yaw differential.")
parser.add_argument(
    "--stop-wheel-damping",
    type=float,
    default=None,
    help="Wheel velocity gain after reaching a target (default: 0.6 indoor, 3.6 legacy).",
)
parser.add_argument("--stop-brake-gain", type=float, default=2.0, help="Proportional body-forward braking gain after reaching a target.")
parser.add_argument("--stop-yaw-brake-gain", type=float, default=1.5, help="Proportional yaw braking gain after reaching a target.")
parser.add_argument(
    "--stop-pretrigger-radius",
    type=float,
    default=0.0,
    help="Legacy compatibility field; canonical stop labels always use --success-radius.",
)
parser.add_argument("--stop-speed-threshold", type=float, default=0.08)
parser.add_argument("--stop-confirm-steps", type=int, default=10)
parser.add_argument("--target-hold-steps", type=int, default=100)
parser.add_argument(
    "--fall-height-threshold",
    type=float,
    default=None,
    help="Root height below which an episode is terminated (default: 0.40 indoor, 0.45 legacy).",
)
parser.add_argument(
    "--posture-min-root-height",
    type=float,
    default=None,
    help="Minimum acceptable root height (default: 0.46 m indoor, fall threshold otherwise).",
)
parser.add_argument("--posture-max-roll-deg", type=float, default=10.0)
parser.add_argument("--posture-max-pitch-deg", type=float, default=12.0)
parser.add_argument("--posture-max-root-height-std", type=float, default=0.025)
parser.add_argument(
    "--startup-posture-steps",
    type=int,
    default=25,
    help="Initial recorded frames covered by the stricter startup posture gate.",
)
parser.add_argument("--startup-max-roll-deg", type=float, default=6.0)
parser.add_argument("--startup-max-pitch-deg", type=float, default=8.0)
parser.add_argument("--startup-max-angular-speed", type=float, default=4.0)
parser.add_argument("--startup-max-joint-target-jump", type=float, default=0.20)
parser.add_argument("--startup-max-leg-symmetry-error", type=float, default=0.35)
parser.add_argument("--turn-wheel-damping", type=float, default=None, help="Wheel Kd used while a navigation yaw command is active.")
parser.add_argument("--wheel-radius", type=float, default=0.09)
parser.add_argument("--track-width", type=float, default=0.48)
parser.add_argument(
    "--implicit-wheel-actuator",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use a PhysX velocity drive for wheel targets (default: enabled for indoor scenes).",
)
parser.add_argument(
    "--implicit-wheel-damping",
    type=float,
    default=4.0,
    help="Velocity-drive damping for indoor implicit wheel actuators.",
)
parser.add_argument("--episode-offset", type=int, default=0, help="First output episode index for appending scenario runs.")
parser.add_argument("--wheel-damping", type=float, default=None, help="Isaac-only wheel Kd override; default adapts for yaw commands.")
parser.add_argument(
    "--mirror-negative-yaw",
    action="store_true",
    help="Mirror the public positive-yaw expert for negative yaw commands using M20 left-right symmetry.",
)
parser.add_argument("--task-text", default=None)
parser.add_argument(
    "--indoor-manifest",
    type=Path,
    default=None,
    help="Canonical indoor ObjectNav manifest. Requires --scenario-episode-id.",
)
parser.add_argument("--scenario-episode-id", default=None)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "datasets/public_m20_native_v1")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/public_m20_native_v1")
parser.add_argument("--image-width", type=int, default=160)
parser.add_argument("--image-height", type=int, default=96)
parser.add_argument(
    "--camera-pitch-deg",
    type=float,
    default=22.0,
    help="Pitch front/rear cameras down so a low object remains visible during approach.",
)
parser.add_argument(
    "--camera-focal-length",
    type=float,
    default=12.0,
    help="Pinhole focal length in mm; lower values widen the ObjectNav field of view.",
)
parser.add_argument("--video", action="store_true", help="Required: write one MP4 per episode.")
parser.add_argument("--dagger-checkpoint", type=Path, default=None, help="Optional VLA checkpoint used for mixed DAgger rollouts.")
parser.add_argument("--dagger-alpha", type=float, default=0.75, help="Fraction of the public expert action used during a DAgger rollout.")
parser.add_argument("--dagger-skill-checkpoint", type=Path, default=None, help="Optional high-level skill VLA used to visit states labeled by the public expert.")
parser.add_argument("--dagger-skill-expert-probability", type=float, default=0.25)
parser.add_argument("--dagger-skill-min-forward", type=float, default=0.08)
parser.add_argument("--dagger-skill-max-forward", type=float, default=0.35)
parser.add_argument("--dagger-skill-max-yaw", type=float, default=0.5)
parser.add_argument("--dagger-model-device", default="cpu", help="Device for the optional DAgger VLA model.")
parser.add_argument(
    "--smolvla-checkpoint",
    type=Path,
    default=None,
    help="Run the trained SmolVLA high-level policy in learner-only mode.",
)
parser.add_argument(
    "--smolvla-dagger-labels",
    action="store_true",
    help=(
        "Execute SmolVLA while writing the privileged simulation expert's "
        "6-D labels; learner actions are retained for DAgger diagnostics."
    ),
)
parser.add_argument(
    "--smolvla-dagger-expert-alpha",
    type=float,
    default=0.0,
    help="Blend this fraction of the expert command into a SmolVLA DAgger rollout.",
)
parser.add_argument(
    "--smolvla-dagger-visible-intervention-fraction",
    type=float,
    default=0.01,
    help=(
        "When DAgger labels are enabled, use the expert command if the target "
        "pixel fraction falls below this value; zero disables this intervention."
    ),
)
parser.add_argument(
    "--smolvla-dagger-visible-intervention-armed-fraction",
    type=float,
    default=0.05,
    help=(
        "Arm the visibility intervention only after the target has reached "
        "this pixel fraction, preventing an expert takeover at episode start."
    ),
)
parser.add_argument(
    "--smolvla-dagger-stability-intervention-height",
    type=float,
    default=0.48,
    help=(
        "During DAgger collection, use the expert command below this root "
        "height; zero disables the stability intervention."
    ),
)
parser.add_argument(
    "--smolvla-dataset-root",
    type=Path,
    default=DATA_ROOT / "datasets/m20_visible_objectnav_lerobot_v2",
)
parser.add_argument("--smolvla-model-device", default="cuda")
parser.add_argument(
    "--smolvla-action-hold-steps",
    type=int,
    default=10,
    help="Hold each SmolVLA command for this many 50 Hz control frames.",
)
parser.add_argument(
    "--smolvla-ensemble-size",
    type=int,
    default=4,
    help="Independent flow-matching samples per VLA query; one keeps legacy behavior.",
)
parser.add_argument(
    "--smolvla-inference-seed",
    type=int,
    default=20260723,
    help="Base seed for reproducible per-query SmolVLA ensemble samples.",
)
parser.add_argument(
    "--smolvla-stop-min-votes",
    type=int,
    default=1,
    help="Minimum ensemble members above the stop threshold for one stop vote.",
)
parser.add_argument("--smolvla-stop-threshold", type=float, default=0.4)
parser.add_argument(
    "--smolvla-stop-confirm-steps",
    type=int,
    default=2,
    help="Consecutive VLA queries required after ensemble voting before latching stop.",
)
parser.add_argument(
    "--smolvla-stop-approach-steps",
    type=int,
    default=60,
    help="Control frames to creep forward after confirmed stop intent before braking.",
)
parser.add_argument(
    "--smolvla-stop-approach-max-forward",
    type=float,
    default=0.18,
    help="Maximum forward command while the confirmed stop intent is approaching.",
)
parser.add_argument(
    "--smolvla-command-max-forward",
    type=float,
    default=0.45,
    help="Safety cap on learner forward command in m/s.",
)
parser.add_argument(
    "--smolvla-command-max-yaw",
    type=float,
    default=0.35,
    help="Safety cap on learner yaw command in rad/s.",
)
parser.add_argument(
    "--smolvla-command-smoothing",
    type=float,
    default=0.5,
    help="New-command weight for exponential smoothing between VLA queries.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
INDOOR_SELECTION = None
if args.indoor_manifest is not None:
    if args.scenario_episode_id is None:
        parser.error("--indoor-manifest requires --scenario-episode-id")
    if not args.indoor_manifest.is_file():
        parser.error(f"Indoor scenario manifest not found: {args.indoor_manifest}")
    try:
        INDOOR_SELECTION = resolve_episode(
            load_manifest(args.indoor_manifest), args.scenario_episode_id
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    if args.episodes != 1:
        parser.error("Indoor manifest collection records exactly one selected scenario per process")
    scenario_episode = INDOOR_SELECTION["episode"]
    target_position = INDOOR_SELECTION["target_slot"]["position"]
    start_position = INDOOR_SELECTION["start_slot"]["position"]
    scenario_text = INDOOR_SELECTION["task_text"]
    if args.task_text is not None and args.task_text != scenario_text:
        parser.error("--task-text must match the selected manifest instruction")
    args.task_text = scenario_text
    args.target_color = "none"
    args.target_x, args.target_y = target_position[:2]
    args.initial_x, args.initial_y = start_position[:2]
    args.initial_yaw_deg = INDOOR_SELECTION["start_slot"]["yaw_deg"]
    args.seed = int(scenario_episode["seed"])
    args.navigate_to_target = True
    args.stop_on_target = True
    if args.override_navigation_wheels is None:
        args.override_navigation_wheels = True
elif args.scenario_episode_id is not None:
    parser.error("--scenario-episode-id requires --indoor-manifest")
if args.override_navigation_wheels is None:
    args.override_navigation_wheels = False
if args.task_text is None:
    args.task_text = "向前走"
if args.stop_wheel_damping is None:
    args.stop_wheel_damping = 0.6 if INDOOR_SELECTION is not None else 3.6
if args.implicit_wheel_actuator is None:
    args.implicit_wheel_actuator = INDOOR_SELECTION is not None
if args.fall_height_threshold is None:
    args.fall_height_threshold = 0.40 if INDOOR_SELECTION is not None else 0.45
if args.posture_min_root_height is None:
    args.posture_min_root_height = (
        0.46 if INDOOR_SELECTION is not None else args.fall_height_threshold
    )
if not args.video:
    parser.error("--video is required so every recorded episode has an inspectable MP4")
if args.episodes <= 0 or args.steps <= 0:
    parser.error("--episodes and --steps must be positive")
if args.warmup_steps <= 0:
    parser.error("--warmup-steps must be positive")
if args.startup_action_blend_steps <= 0:
    parser.error("--startup-action-blend-steps must be positive")
if args.startup_posture_steps <= 0:
    parser.error("--startup-posture-steps must be positive")
if args.episode_offset < 0:
    parser.error("--episode-offset must be non-negative")
if args.initial_yaw_jitter_deg < 0.0:
    parser.error("--initial-yaw-jitter-deg must be non-negative")
if args.navigate_to_target and args.target_color == "none" and INDOOR_SELECTION is None:
    parser.error("--navigate-to-target requires a colored or manifest object target")
if args.nav_forward_speed <= 0.0 or args.nav_heading_gain <= 0.0 or args.nav_max_yaw <= 0.0:
    parser.error("navigation gains and speed must be positive")
if args.wheel_radius <= 0.0 or args.track_width <= 0.0:
    parser.error("wheel geometry must be positive")
if args.nav_command_hold_steps <= 0:
    parser.error("--nav-command-hold-steps must be positive")
if args.nav_turn_threshold <= 0.0:
    parser.error("--nav-turn-threshold must be positive")
if args.nav_fixed_turn_steps < 0:
    parser.error("--nav-fixed-turn-steps must be non-negative")
if args.success_radius <= 0.0 or args.nav_slow_radius <= args.success_radius:
    parser.error("--nav-slow-radius must be greater than the positive --success-radius")
if args.success_final_tolerance < 0.0:
    parser.error("--success-final-tolerance must be non-negative")
if not 0.0 < args.nav_min_distance_scale <= 1.0:
    parser.error("--nav-min-distance-scale must be in (0, 1]")
if args.nav_wheel_acceleration <= 0.0 or args.nav_wheel_yaw_gain <= 0.0 or args.stop_wheel_damping <= 0.0:
    parser.error("--nav-wheel-acceleration, --nav-wheel-yaw-gain and --stop-wheel-damping must be positive")
if args.implicit_wheel_damping <= 0.0:
    parser.error("--implicit-wheel-damping must be positive")
if args.stop_brake_gain < 0.0 or args.stop_yaw_brake_gain < 0.0:
    parser.error("--stop-brake-gain and --stop-yaw-brake-gain must be non-negative")
if args.stop_pretrigger_radius < 0.0:
    parser.error("--stop-pretrigger-radius must be non-negative")
if args.stop_pretrigger_radius > 0.0 and args.stop_pretrigger_radius < args.success_radius:
    parser.error("--stop-pretrigger-radius must be zero or at least --success-radius")
if args.stop_speed_threshold <= 0.0 or args.stop_confirm_steps <= 0 or args.target_hold_steps <= 0:
    parser.error("stop speed, confirmation and target hold settings must be positive")
if args.fall_height_threshold <= 0.0:
    parser.error("--fall-height-threshold must be positive")
if args.posture_min_root_height < args.fall_height_threshold:
    parser.error("--posture-min-root-height must not be below --fall-height-threshold")
if (
    args.posture_max_roll_deg <= 0.0
    or args.posture_max_pitch_deg <= 0.0
    or args.posture_max_root_height_std <= 0.0
    or args.startup_max_roll_deg <= 0.0
    or args.startup_max_pitch_deg <= 0.0
    or args.startup_max_angular_speed <= 0.0
    or args.startup_max_joint_target_jump <= 0.0
    or args.startup_max_leg_symmetry_error <= 0.0
):
    parser.error("posture thresholds must be positive")
if not -60.0 <= args.camera_pitch_deg <= 60.0:
    parser.error("--camera-pitch-deg must be between -60 and 60 degrees")
if args.camera_focal_length <= 0.0:
    parser.error("--camera-focal-length must be positive")
if not args.policy.is_file():
    parser.error(f"M20 policy not found: {args.policy}")
if args.dagger_checkpoint is not None and not args.dagger_checkpoint.is_file():
    parser.error(f"DAgger checkpoint not found: {args.dagger_checkpoint}")
if args.dagger_skill_checkpoint is not None and not args.dagger_skill_checkpoint.is_file():
    parser.error(f"Skill DAgger checkpoint not found: {args.dagger_skill_checkpoint}")
if args.dagger_checkpoint is not None and args.dagger_skill_checkpoint is not None:
    parser.error("Use either --dagger-checkpoint or --dagger-skill-checkpoint, not both")
if args.smolvla_checkpoint is not None and (
    args.dagger_checkpoint is not None or args.dagger_skill_checkpoint is not None
):
    parser.error("--smolvla-checkpoint cannot be combined with DAgger checkpoints")
if args.smolvla_checkpoint is not None and not args.smolvla_checkpoint.is_dir():
    parser.error(f"SmolVLA checkpoint directory not found: {args.smolvla_checkpoint}")
if args.smolvla_dagger_labels and args.smolvla_checkpoint is None:
    parser.error("--smolvla-dagger-labels requires --smolvla-checkpoint")
if not 0.0 <= args.smolvla_dagger_expert_alpha <= 1.0:
    parser.error("--smolvla-dagger-expert-alpha must be in [0, 1]")
if args.smolvla_dagger_expert_alpha > 0.0 and not args.smolvla_dagger_labels:
    parser.error("--smolvla-dagger-expert-alpha requires --smolvla-dagger-labels")
if not 0.0 <= args.smolvla_dagger_visible_intervention_fraction <= 1.0:
    parser.error("--smolvla-dagger-visible-intervention-fraction must be in [0, 1]")
if not 0.0 <= args.smolvla_dagger_visible_intervention_armed_fraction <= 1.0:
    parser.error("--smolvla-dagger-visible-intervention-armed-fraction must be in [0, 1]")
if (
    args.smolvla_dagger_visible_intervention_fraction
    > args.smolvla_dagger_visible_intervention_armed_fraction
):
    parser.error("visibility intervention fraction must not exceed its armed fraction")
if not 0.0 <= args.smolvla_dagger_stability_intervention_height <= 1.0:
    parser.error("--smolvla-dagger-stability-intervention-height must be in [0, 1]")
if args.smolvla_action_hold_steps <= 0 or args.smolvla_stop_confirm_steps <= 0:
    parser.error("SmolVLA hold and stop confirmation steps must be positive")
if args.smolvla_ensemble_size <= 0:
    parser.error("--smolvla-ensemble-size must be positive")
if args.smolvla_stop_min_votes <= 0 or args.smolvla_stop_min_votes > args.smolvla_ensemble_size:
    parser.error("--smolvla-stop-min-votes must be within the ensemble size")
if args.smolvla_command_max_forward <= 0.0 or args.smolvla_command_max_yaw <= 0.0:
    parser.error("SmolVLA command safety caps must be positive")
if args.smolvla_stop_approach_steps < 0:
    parser.error("--smolvla-stop-approach-steps must be non-negative")
if not 0.0 < args.smolvla_stop_approach_max_forward <= args.smolvla_command_max_forward:
    parser.error("SmolVLA stop approach speed must be positive and within the forward cap")
if not 0.0 < args.smolvla_command_smoothing <= 1.0:
    parser.error("--smolvla-command-smoothing must be in (0, 1]")
if args.smolvla_stop_threshold <= 0.0:
    parser.error("--smolvla-stop-threshold must be positive")
if args.dagger_skill_checkpoint is not None and not args.navigate_to_target:
    parser.error("--dagger-skill-checkpoint requires --navigate-to-target for expert command labels")
if not 0.0 <= args.dagger_alpha <= 1.0:
    parser.error("--dagger-alpha must be in [0, 1]")
if not 0.0 <= args.dagger_skill_expert_probability <= 1.0:
    parser.error("--dagger-skill-expert-probability must be in [0, 1]")
if not 0.0 < args.dagger_skill_min_forward <= args.dagger_skill_max_forward:
    parser.error("Skill DAgger forward limits are invalid")
if args.dagger_skill_max_yaw <= 0.0:
    parser.error("--dagger-skill-max-yaw must be positive")
args.enable_cameras = True
app = AppLauncher(args).app

TARGET_COLORS = {
    "red": (0.9, 0.05, 0.03),
    "blue": (0.03, 0.15, 0.95),
    "green": (0.04, 0.8, 0.08),
}
TARGET_RGB = TARGET_COLORS.get(args.target_color)
TARGET_PRESENT = TARGET_RGB is not None or INDOOR_SELECTION is not None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from m20_vla_model import M20VLAActionChunk  # noqa: E402
from m20_vla_skill_model import (  # noqa: E402
    COMMAND_SCALE,
    SKILL_NAMES,
    M20VLASkillPolicy,
)
if args.smolvla_checkpoint is not None:  # noqa: E402
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
    from lerobot.policies.smolvla.processor_smolvla import (  # noqa: E402
        make_smolvla_pre_post_processors,
    )

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

from assets.m20pro import M20PRO_CFG  # noqa: E402


POLICY_JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
MIRROR_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14], dtype=torch.long)
MIRROR_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
MIRROR_ANGULAR_VELOCITY = torch.tensor([-1.0, 1.0, -1.0])
MIRROR_POLAR_VECTOR = torch.tensor([1.0, -1.0, 1.0])
DEFAULT_POLICY_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
LEG_ACTION_SCALE = torch.tensor([0.125, 0.25, 0.25] * 4, dtype=torch.float32)
WHEEL_DAMPING = args.wheel_damping if args.wheel_damping is not None else (3.6 if abs(args.command_yaw) >= 0.05 else 0.6)
TURN_WHEEL_DAMPING = args.turn_wheel_damping if args.turn_wheel_damping is not None else WHEEL_DAMPING
SENSOR_LINK = "base_link/base_link" if "m20_mjcf" in str(os.environ.get("M20PRO_USD_PATH", "")) else "base_link"
WHEEL_ACTUATOR_CFG = (
    ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel_joint"],
        effort_limit_sim=21.6,
        velocity_limit_sim=79.3,
        stiffness=0.0,
        damping=args.implicit_wheel_damping,
        armature=0.02,
    )
    if args.implicit_wheel_actuator
    else DCMotorCfg(
        joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, saturation_effort=21.6,
        velocity_limit=79.3, stiffness=0.0, damping=WHEEL_DAMPING,
    )
)

PUBLIC_M20_CFG = M20PRO_CFG.replace(
    init_state=M20PRO_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.54),
        joint_pos={name: float(value) for name, value in zip(POLICY_JOINT_NAMES, DEFAULT_POLICY_POSE)},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hipx": DCMotorCfg(
            joint_names_expr=[".*_hipx_joint"], effort_limit=32.4, saturation_effort=32.4,
            velocity_limit=45.0, stiffness=80.0, damping=2.0,
        ),
        "hipy_knee": DCMotorCfg(
            joint_names_expr=[".*_(hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4,
            velocity_limit=22.4, stiffness=80.0, damping=2.0,
        ),
        "wheels": WHEEL_ACTUATOR_CFG,
    },
)


def target_asset_cfg() -> AssetBaseCfg | None:
    if INDOOR_SELECTION is not None:
        object_cfg = INDOOR_SELECTION["object"]
        scale = float(object_cfg["uniform_scale"])
        usd_path = f"{ISAAC_NUCLEUS_DIR}/{object_cfg['usd_path']}"
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(scale, scale, scale),
                semantic_tags=[("class", "target")],
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(args.target_x, args.target_y, 0.0)),
        )
    if TARGET_RGB is None:
        return None
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 0.42, 0.84),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_RGB),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(args.target_x, args.target_y, 0.42)),
    )


def lidar_cfg() -> RayCasterCfg:
    common = {
        "prim_path": f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}",
        "update_period": 0.02,
        "ray_alignment": "base",
        "offset": RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        "pattern_cfg": patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        "max_distance": 20.0,
    }
    if INDOOR_SELECTION is None:
        return RayCasterCfg(mesh_prim_paths=["/World/ground"], **common)
    return MultiMeshRayCasterCfg(
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground", track_mesh_transforms=False
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/IndoorScene", track_mesh_transforms=False
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Target", track_mesh_transforms=False
            ),
        ],
        **common,
    )


def camera_pitch_quaternion(pitch_deg: float, yaw_deg: float = 0.0) -> tuple[float, float, float, float]:
    """Return an Isaac (w, x, y, z) yaw-then-pitch quaternion."""
    yaw = np.deg2rad(yaw_deg) * 0.5
    pitch = np.deg2rad(pitch_deg) * 0.5
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return (
        float(cy * cp),
        float(-sy * sp),
        float(cy * sp),
        float(sy * cp),
    )


@configclass
class NativeM20SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = PUBLIC_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    target = target_asset_cfg()
    front_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/front_camera", update_period=0.02,
        height=args.image_height,
        width=args.image_width,
        data_types=(
            ["rgb", "semantic_segmentation"]
            if INDOOR_SELECTION is not None
            else ["rgb"]
        ),
        semantic_filter="class:target",
        semantic_segmentation_mapping={"class:target": (255, 0, 255, 255)},
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.camera_focal_length, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.38, 0.0, 0.12),
            rot=camera_pitch_quaternion(args.camera_pitch_deg),
            convention="world",
        ),
    )
    rear_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/rear_camera", update_period=0.02,
        height=args.image_height,
        width=args.image_width,
        data_types=(
            ["rgb", "semantic_segmentation"]
            if INDOOR_SELECTION is not None
            else ["rgb"]
        ),
        semantic_filter="class:target",
        semantic_segmentation_mapping={"class:target": (255, 0, 255, 255)},
        spawn=sim_utils.PinholeCameraCfg(focal_length=args.camera_focal_length, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.38, 0.0, 0.12),
            rot=camera_pitch_quaternion(args.camera_pitch_deg, 180.0),
            convention="world",
        ),
    )
    lidar = lidar_cfg()
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))


def native_observation(
    robot: Articulation,
    joint_ids: list[int],
    last_action: torch.Tensor,
    command: tuple[float, float, float],
    mirror: bool = False,
) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command = torch.tensor([list(command)], device=robot.device)
    joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POLICY_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    observation = torch.cat(
        [robot.data.root_ang_vel_b * 0.25, projected_gravity, command, joint_pos, joint_vel, last_action], dim=-1
    )
    if mirror:
        observation[:, 0:3] *= MIRROR_ANGULAR_VELOCITY.to(robot.device)
        observation[:, 3:6] *= MIRROR_POLAR_VECTOR.to(robot.device)
        observation[:, 9:25] = observation[:, 9:25][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
        observation[:, 25:41] = observation[:, 25:41][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
        observation[:, 41:57] = observation[:, 41:57][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
    return observation


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    return action[:, MIRROR_PERM.to(action.device)] * MIRROR_SIGN.to(action.device)


def command_for_step(step: int) -> tuple[float, float, float]:
    if args.stop_after is not None and step >= args.stop_after:
        return (0.0, 0.0, 0.0)
    return (args.command_x, args.command_y, args.command_yaw)


def yaw_from_quaternion(quat: torch.Tensor) -> float:
    values = quat.detach().cpu().numpy()
    return float(
        np.arctan2(
            2.0 * (values[0] * values[3] + values[1] * values[2]),
            1.0 - 2.0 * (values[2] ** 2 + values[3] ** 2),
        )
    )


def roll_pitch_from_quaternion(quat: torch.Tensor) -> tuple[float, float]:
    """Return body roll and pitch in degrees for posture validation."""
    w, x, y, z = quat.detach().cpu().numpy()
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(np.degrees(roll)), float(np.degrees(pitch))


def leg_symmetry_error(robot: Articulation, leg_ids: list[int]) -> float:
    """Measure left-right standing-pose asymmetry in joint radians."""
    leg_position = robot.data.joint_pos[0, leg_ids]
    symmetry_residual = torch.stack(
        (
            leg_position[0] + leg_position[3],
            leg_position[1] - leg_position[4],
            leg_position[2] - leg_position[5],
            leg_position[6] + leg_position[9],
            leg_position[7] - leg_position[10],
            leg_position[8] - leg_position[11],
        )
    )
    return float(torch.max(torch.abs(symmetry_residual)).item())


def navigation_command(robot: Articulation, target_reached: bool) -> tuple[float, float, float]:
    if target_reached:
        return (0.0, 0.0, 0.0)
    delta = torch.tensor([args.target_x, args.target_y], device=robot.device) - robot.data.root_pos_w[0, :2]
    distance = float(torch.linalg.vector_norm(delta).item())
    target_heading = float(torch.atan2(delta[1], delta[0]).item())
    current_heading = yaw_from_quaternion(robot.data.root_quat_w[0])
    heading_error = float(np.arctan2(np.sin(target_heading - current_heading), np.cos(target_heading - current_heading)))
    yaw_command = float(np.clip(args.nav_heading_gain * heading_error, -args.nav_max_yaw, args.nav_max_yaw))
    remaining = max(distance - args.success_radius, 0.0)
    slow_span = args.nav_slow_radius - args.success_radius
    distance_scale = float(np.clip(remaining / slow_span, args.nav_min_distance_scale, 1.0))
    heading_scale = max(float(np.cos(heading_error)), 0.0)
    if abs(heading_error) >= args.nav_turn_threshold:
        # The released M20 policy loses forward speed when a yaw command is
        # mixed with translation. Saturate the turn and wait for alignment,
        # then issue a pure forward command inside the heading deadband.
        return (0.0, 0.0, float(np.sign(heading_error) * args.nav_max_yaw))
    if not args.override_navigation_wheels:
        yaw_command = 0.0
    return (args.nav_forward_speed * distance_scale * heading_scale, 0.0, yaw_command)


def override_navigation_wheels(
    action: torch.Tensor,
    command: tuple[float, float, float],
    previous_action: torch.Tensor,
) -> torch.Tensor:
    forward, _, yaw_rate = command
    half_track = 0.5 * args.track_width
    yaw_differential = half_track * yaw_rate * args.nav_wheel_yaw_gain
    # M20's four wheel joints use the negative Y axis, so their signed
    # differential is opposite to the usual planar-drive convention.
    left_velocity = -(forward + yaw_differential) / args.wheel_radius
    right_velocity = -(forward - yaw_differential) / args.wheel_radius
    desired_wheel_action = torch.tensor(
        [[left_velocity, right_velocity, left_velocity, right_velocity]],
        dtype=action.dtype,
        device=action.device,
    ).div_(5.0).clamp_(-2.5, 2.5)
    max_action_delta = args.nav_wheel_acceleration / 50.0 / 5.0
    wheel_action = torch.clamp(
        desired_wheel_action,
        previous_action[:, 12:] - max_action_delta,
        previous_action[:, 12:] + max_action_delta,
    )
    action = action.clone()
    action[:, 12:] = wheel_action
    return action


def target_stop_command(robot: Articulation) -> tuple[float, float, float]:
    """Convert current body velocity into a bounded reverse wheel command."""
    body_velocity = robot.data.root_lin_vel_b[0]
    body_angular_velocity = robot.data.root_ang_vel_b[0]
    forward = float(torch.clamp(-args.stop_brake_gain * body_velocity[0], -0.35, 0.35).item())
    yaw_rate = float(torch.clamp(-args.stop_yaw_brake_gain * body_angular_velocity[2], -0.5, 0.5).item())
    return (forward, 0.0, yaw_rate)


def set_navigation_wheel_damping(
    robot: Articulation,
    command: tuple[float, float, float],
    target_reached: bool,
) -> None:
    if not args.navigate_to_target or args.implicit_wheel_actuator:
        return
    wheel_actuator = robot.actuators.get("wheels")
    if wheel_actuator is None:
        return
    if args.override_navigation_wheels:
        damping = args.stop_wheel_damping if target_reached else 0.6
    else:
        # Keep the released controller's wheel Kd. The earlier yaw-only 3.6
        # override created a large angular-velocity spike in Isaac PhysX.
        if target_reached:
            damping = args.stop_wheel_damping
        elif abs(command[2]) >= 0.05:
            damping = TURN_WHEEL_DAMPING
        else:
            damping = WHEEL_DAMPING
    wheel_actuator.damping.fill_(damping)


def rgb(camera) -> np.ndarray:
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def target_pixel_fraction(camera) -> float:
    segmentation = camera.data.output.get("semantic_segmentation")
    if segmentation is None:
        return 0.0
    target_color = torch.tensor(
        [255, 0, 255], dtype=segmentation.dtype, device=segmentation.device
    )
    target_pixels = torch.all(segmentation[0, ..., :3] == target_color, dim=-1)
    return float(target_pixels.float().mean().item())


def scan(lidar, max_distance: float = 20.0) -> np.ndarray:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    values = torch.linalg.vector_norm(hits - origin, dim=-1)
    return torch.nan_to_num(values, nan=max_distance, posinf=max_distance, neginf=0.0).clamp(0.0, max_distance).cpu().numpy().astype(np.float32)


def smolvla_state(proprio_observation: np.ndarray, lidar_observation: np.ndarray) -> np.ndarray:
    """Match the 8 invariant proprio + 24 LiDAR-sector state used in v2."""
    lidar = np.asarray(lidar_observation, dtype=np.float32)
    if lidar.size != 72:
        raise ValueError(f"SmolVLA expects 72 LiDAR beams, got {lidar.shape}")
    sectors = np.nan_to_num(lidar, nan=20.0, posinf=20.0, neginf=0.0).reshape(24, 3)
    proprio = np.asarray(proprio_observation, dtype=np.float32)
    if proprio.shape != (8,):
        raise ValueError(f"SmolVLA expects 8 invariant proprio values, got {proprio.shape}")
    result = np.concatenate((proprio, sectors.min(axis=1)))
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid SmolVLA state shape/value: {result.shape}")
    return result.astype(np.float32, copy=False)


def smolvla_body_proprio(robot: Articulation) -> np.ndarray:
    """Return body-frame velocity and gravity, excluding absolute world pose."""
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    body_linear_velocity = quat_apply_inverse(
        robot.data.root_quat_w, robot.data.root_lin_vel_w
    )[0]
    body_angular_velocity = quat_apply_inverse(
        robot.data.root_quat_w, robot.data.root_ang_vel_w
    )[0]
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)[0, :2]
    values = torch.cat(
        (
            body_linear_velocity,
            body_angular_velocity,
            projected_gravity,
        )
    )
    result = values.detach().cpu().numpy().astype(np.float32)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid invariant proprioception: {result.shape}")
    return result


def encode_text(text: str, max_length: int = 32) -> torch.Tensor:
    values = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[: len(values)] = values
    return torch.from_numpy(tokens)


def canonical_skill_command(
    command_norm: torch.Tensor,
    skill_logits: torch.Tensor,
) -> tuple[tuple[float, float, float], str]:
    command = command_norm[0].detach().cpu() * COMMAND_SCALE
    skill = SKILL_NAMES[int(skill_logits[0].argmax().item())]
    if skill == "left":
        yaw = min(max(abs(float(command[2])), 0.12), args.dagger_skill_max_yaw)
        return (0.0, 0.0, yaw), skill
    if skill == "right":
        yaw = min(max(abs(float(command[2])), 0.12), args.dagger_skill_max_yaw)
        return (0.0, 0.0, -yaw), skill
    if skill == "backward":
        forward = max(min(float(command[0]), 0.0), -args.dagger_skill_max_forward)
        return (forward, 0.0, 0.0), skill
    if skill == "search":
        yaw = args.dagger_skill_max_yaw if float(command[2]) >= 0.0 else -args.dagger_skill_max_yaw
        return (0.0, 0.0, yaw), skill
    if skill in {"stop", "jump"}:
        return (0.0, 0.0, 0.0), skill
    forward = min(
        max(float(command[0]), args.dagger_skill_min_forward),
        args.dagger_skill_max_forward,
    )
    return (forward, 0.0, 0.0), "forward"


def public_expert_action(
    session: ort.InferenceSession,
    robot: Articulation,
    joint_ids: list[int],
    previous_action: torch.Tensor,
    command: tuple[float, float, float],
    target_reached: bool,
) -> torch.Tensor:
    wheel_command = command
    policy_command = (0.0, 0.0, 0.0) if target_reached else command
    if args.navigate_to_target and args.override_navigation_wheels:
        # Steering is supplied by the differential-wheel override below. Do
        # not also feed yaw into the released leg policy: its native turning
        # behavior unloads one front leg while the overridden wheels continue
        # to push, producing the persistent asymmetric front-leg posture.
        policy_command = (policy_command[0], 0.0, 0.0)
    mirror = args.mirror_negative_yaw and policy_command[2] < -1e-6
    if mirror:
        policy_command = (policy_command[0], policy_command[1], abs(policy_command[2]))
    observation = native_observation(
        robot,
        joint_ids,
        previous_action,
        policy_command,
        mirror=mirror,
    )
    action_np = session.run(["actions"], {"obs": observation.cpu().numpy()})[0]
    action = torch.from_numpy(action_np).to(robot.device)
    if mirror:
        action = mirror_action(action)
    if args.navigate_to_target and args.override_navigation_wheels:
        override_command = target_stop_command(robot) if target_reached else wheel_command
        action = override_navigation_wheels(action, override_command, previous_action)
    return action


def reset_scene(scene: InteractiveScene, initial_yaw_rad: float) -> None:
    robot = scene["robot"]
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 0] += args.initial_x
    root_state[:, 1] += args.initial_y
    half_yaw = 0.5 * initial_yaw_rad
    root_state[:, 3:7] = torch.tensor(
        [[np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]],
        dtype=root_state.dtype,
        device=root_state.device,
    )
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()


def spawn_indoor_geometry() -> None:
    if INDOOR_SELECTION is None:
        return
    for item in INDOOR_SELECTION["scene"]["geometry"]:
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(value) for value in item["size"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(value) for value in item["color"])
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        )
        cfg.func(
            f"/World/IndoorScene/Geometry/{item['id']}",
            cfg,
            translation=tuple(float(value) for value in item["position"]),
        )


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 57] or session.get_outputs()[0].shape != [1, 16]:
        raise RuntimeError("Native M20 policy is not the expected 57->16 model")
    dagger_model = None
    dagger_skill_model = None
    smolvla_model = None
    smolvla_preprocessor = None
    smolvla_postprocessor = None
    dagger_device = torch.device(
        args.dagger_model_device
        if torch.cuda.is_available() or not args.dagger_model_device.startswith("cuda")
        else "cpu"
    )
    rng = np.random.default_rng(args.seed)
    if args.episode_offset:
        rng.uniform(
            -args.initial_yaw_jitter_deg,
            args.initial_yaw_jitter_deg,
            size=args.episode_offset,
        )
    if args.dagger_checkpoint is not None:
        payload = torch.load(args.dagger_checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("config", {})
        dagger_model = M20VLAActionChunk(
            int(payload["horizon"]), architecture=config.get("architecture", "global_v1")
        )
        dagger_model.load_state_dict(payload["model_state_dict"])
        dagger_model.to(dagger_device).eval()
    if args.dagger_skill_checkpoint is not None:
        payload = torch.load(args.dagger_skill_checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("config", {})
        dagger_skill_model = M20VLASkillPolicy(
            config.get("architecture", "spatial_v2"),
            search_head=bool(config.get("search_head", False)),
            target_head=bool(config.get("target_head", False)),
            target_head_mode=config.get("target_head_mode", "shared_v1"),
        )
        dagger_skill_model.load_state_dict(payload["model_state_dict"])
        dagger_skill_model.to(dagger_device).eval()
    if args.smolvla_checkpoint is not None:
        smolvla_device = (
            args.smolvla_model_device
            if torch.cuda.is_available() or not args.smolvla_model_device.startswith("cuda")
            else "cpu"
        )
        smolvla_config = SmolVLAPolicy.from_pretrained(
            args.smolvla_checkpoint, device=smolvla_device
        )
        smolvla_model = smolvla_config
        smol_dataset = LeRobotDataset(
            "m20pro_visible_objectnav_v2",
            root=args.smolvla_dataset_root,
            download_videos=False,
        )
        smolvla_preprocessor, smolvla_postprocessor = make_smolvla_pre_post_processors(
            smolvla_model.config, smol_dataset.meta.stats
        )
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=args.device or "cuda:0"))
    spawn_indoor_geometry()
    scene = InteractiveScene(NativeM20SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, joint_names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if joint_names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 policy order mismatch: {joint_names}")
    leg_ids, wheel_ids = joint_ids[:12], joint_ids[12:]
    front, rear, lidar, third = scene["front_camera"], scene["rear_camera"], scene["lidar"], scene["third_person"]
    ray_count = int(lidar.data.ray_hits_w.shape[1])
    physics_dt = sim.get_physics_dt()
    scenario_metadata = None
    if INDOOR_SELECTION is not None:
        task_type = (
            "hidden_object_search"
            if str(INDOOR_SELECTION["episode"]["split"]).startswith("hidden_")
            or bool(INDOOR_SELECTION["episode"].get("search_required", False))
            else "visible_object_navigation"
        )
        scenario_metadata = {
            "manifest": str(args.indoor_manifest.resolve()),
            "manifest_sha256": INDOOR_SELECTION["manifest_sha256"],
            "scenario_episode_id": INDOOR_SELECTION["episode"]["id"],
            "task_type": task_type,
            "scene_id": INDOOR_SELECTION["scene"]["id"],
            "split": INDOOR_SELECTION["episode"]["split"],
            "object_category": INDOOR_SELECTION["object"]["id"],
            "object_label_zh": INDOOR_SELECTION["object"]["label_zh"],
            "object_label_en": INDOOR_SELECTION["object"]["label_en"],
            "object_usd_path": (
                f"{ISAAC_NUCLEUS_DIR}/{INDOOR_SELECTION['object']['usd_path']}"
            ),
            "object_source": INDOOR_SELECTION["object"]["source"],
            "instruction_template_id": INDOOR_SELECTION["instruction_template"]["id"],
            "instruction_language": INDOOR_SELECTION["instruction_template"]["language"],
            "target_slot_id": INDOOR_SELECTION["target_slot"]["id"],
            "start_slot_id": INDOOR_SELECTION["start_slot"]["id"],
            "target_expected_visible_at_start": bool(
                INDOOR_SELECTION["target_visible_at_start"]
            ),
            "search_required": bool(INDOOR_SELECTION["search_required"]),
            "expert_uses_privileged_target_pose": True,
            "inference_uses_privileged_target_pose": False,
        }
        if INDOOR_SELECTION.get("search_family_id") is not None:
            scenario_metadata["search_family_id"] = INDOOR_SELECTION["search_family_id"]
            scenario_metadata["search_family_name"] = INDOOR_SELECTION["search_family_name"]
    metadata = {
        "format": "m20pro_native_expert_hdf5_v1", "expert": "AI-DA-STC/M20-autonomy-sim policy.onnx",
        "policy_protocol": "57 observation -> 16 action; official M20PolicyRunner; no PPO reward",
        "task_text": args.task_text, "command": [args.command_x, args.command_y, args.command_yaw],
        "stop_after": args.stop_after, "stop_on_target": args.stop_on_target,
        "navigate_to_target": args.navigate_to_target, "override_navigation_wheels": args.override_navigation_wheels,
        "target_color": args.target_color,
        "target_xy": [args.target_x, args.target_y], "wheel_damping": WHEEL_DAMPING,
        "implicit_wheel_damping": args.implicit_wheel_damping,
        "wheel_actuator": (
            "physx_implicit_velocity_drive"
            if args.implicit_wheel_actuator
            else "explicit_dc_motor"
        ),
        "scenario": scenario_metadata,
        "initial_yaw": {
            "center_deg": args.initial_yaw_deg,
            "jitter_deg": args.initial_yaw_jitter_deg,
            "seed": args.seed,
        },
        "asset_usd_path": os.environ.get("M20PRO_USD_PATH", str(M20PRO_CFG.spawn.usd_path)),
        "dagger": args.dagger_checkpoint is not None or args.dagger_skill_checkpoint is not None,
        "dagger_checkpoint": None if args.dagger_checkpoint is None else str(args.dagger_checkpoint),
        "dagger_alpha": args.dagger_alpha,
        "dagger_skill": args.dagger_skill_checkpoint is not None,
        "dagger_skill_checkpoint": (
            None if args.dagger_skill_checkpoint is None else str(args.dagger_skill_checkpoint)
        ),
        "dagger_skill_expert_probability": args.dagger_skill_expert_probability,
        "smolvla_checkpoint": (
            None if args.smolvla_checkpoint is None else str(args.smolvla_checkpoint)
        ),
        "smolvla_learner_only": args.smolvla_checkpoint is not None,
        "smolvla_dagger_labels": args.smolvla_dagger_labels,
        "smolvla_dagger_expert_alpha": args.smolvla_dagger_expert_alpha,
        "smolvla_dagger_visible_intervention_fraction": (
            args.smolvla_dagger_visible_intervention_fraction
        ),
        "smolvla_dagger_visible_intervention_armed_fraction": (
            args.smolvla_dagger_visible_intervention_armed_fraction
        ),
        "smolvla_dagger_stability_intervention_height": (
            args.smolvla_dagger_stability_intervention_height
        ),
        "smolvla_action_hold_steps": args.smolvla_action_hold_steps,
        "smolvla_ensemble_size": args.smolvla_ensemble_size,
        "smolvla_inference_seed": args.smolvla_inference_seed,
        "smolvla_stop_min_votes": args.smolvla_stop_min_votes,
        "smolvla_stop_threshold": args.smolvla_stop_threshold,
        "smolvla_stop_confirm_steps": args.smolvla_stop_confirm_steps,
        "smolvla_stop_approach_steps": args.smolvla_stop_approach_steps,
        "smolvla_stop_approach_max_forward": (
            args.smolvla_stop_approach_max_forward
        ),
        "smolvla_command_max_forward": args.smolvla_command_max_forward,
        "smolvla_command_max_yaw": args.smolvla_command_max_yaw,
        "smolvla_command_smoothing": args.smolvla_command_smoothing,
        "navigation": {
            "forward_speed": args.nav_forward_speed, "heading_gain": args.nav_heading_gain,
            "max_yaw": args.nav_max_yaw, "turn_threshold": args.nav_turn_threshold,
            "command_hold_steps": args.nav_command_hold_steps, "fixed_turn_steps": args.nav_fixed_turn_steps,
            "success_radius": args.success_radius, "slow_radius": args.nav_slow_radius,
            "success_final_tolerance": args.success_final_tolerance,
            "min_distance_scale": args.nav_min_distance_scale,
            "wheel_acceleration": args.nav_wheel_acceleration, "wheel_yaw_gain": args.nav_wheel_yaw_gain,
            "stop_wheel_damping": args.stop_wheel_damping, "turn_wheel_damping": TURN_WHEEL_DAMPING,
            "implicit_wheel_damping": args.implicit_wheel_damping,
            "wheel_radius": args.wheel_radius,
            "track_width": args.track_width,
            "stop_brake_gain": args.stop_brake_gain,
            "stop_yaw_brake_gain": args.stop_yaw_brake_gain,
            "stop_pretrigger_radius": args.stop_pretrigger_radius,
            "stop_speed_threshold": args.stop_speed_threshold,
            "stop_confirm_steps": args.stop_confirm_steps,
            "target_hold_steps": args.target_hold_steps,
            "source": "public M20 ONNX leg policy plus target-bearing differential wheel expert",
            "wheel_actuator": (
                "physx_implicit_velocity_drive"
                if args.implicit_wheel_actuator
                else "explicit_dc_motor"
            ),
        },
        "control_hz": 50.0, "joint_names": POLICY_JOINT_NAMES,
        "fall_height_threshold": args.fall_height_threshold,
        "startup_control": {
            "warmup_steps": args.warmup_steps,
            "action_blend_steps": args.startup_action_blend_steps,
        },
        "posture_gate": {
            "min_root_height_m": args.posture_min_root_height,
            "max_roll_deg": args.posture_max_roll_deg,
            "max_pitch_deg": args.posture_max_pitch_deg,
            "max_root_height_std_m": args.posture_max_root_height_std,
            "startup_steps": args.startup_posture_steps,
            "startup_max_roll_deg": args.startup_max_roll_deg,
            "startup_max_pitch_deg": args.startup_max_pitch_deg,
            "startup_max_angular_speed_rps": args.startup_max_angular_speed,
            "startup_max_joint_target_jump_rad": args.startup_max_joint_target_jump,
            "startup_max_leg_symmetry_error_rad": args.startup_max_leg_symmetry_error,
        },
        "sensor_alignment": "pre_action",
        "lidar_mesh_scope": (
            "ground_indoor_geometry_and_target"
            if INDOOR_SELECTION is not None
            else "ground_only"
        ),
        "smolvla_candidate": INDOOR_SELECTION is not None,
        "observation": {"front_rgb": [args.image_height, args.image_width, 3], "rear_rgb": [args.image_height, args.image_width, 3],
                         "lidar": [ray_count], "proprio": [57], "state": [45], "expert_command": [3]},
        "dagger_observation": {
            "learner_command": [3],
            "expert_intervention": [],
        } if args.dagger_skill_checkpoint is not None else None,
        "action": {"shape": [16], "leg": "default_pose + output[:12] * [0.125,0.25,0.25]", "wheel": "output[12:] * 5.0"},
        "high_level_action": {
            "shape": [6],
            "fields": ["forward_mps", "lateral_mps", "yaw_rps", "stop", "search", "parkour"],
            "source": "privileged demonstration expert only; never exposed at VLA inference",
        },
        "success_rule": (
            "posture gate plus command-direction check; target episodes also "
            "require reaching and holding target_xy"
        ),
    }
    if INDOOR_SELECTION is not None:
        metadata_name = f"metadata_{INDOOR_SELECTION['episode']['id']}.json"
    else:
        metadata_name = (
            "metadata.json"
            if args.episode_offset == 0
            else f"metadata_run_{args.episode_offset:04d}.json"
        )
    (args.output_dir / metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    default_pose = torch.tensor([0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0], device=robot.device).unsqueeze(0)
    leg_scale = LEG_ACTION_SCALE.to(robot.device).unsqueeze(0)
    zero_wheels = torch.zeros((1, 4), device=robot.device)

    for episode in range(args.episodes):
        episode_id = args.episode_offset + episode
        intervention_rng = np.random.default_rng(args.seed + 1_000_003 * (episode_id + 1))
        episode_initial_yaw_deg = args.initial_yaw_deg + float(
            rng.uniform(-args.initial_yaw_jitter_deg, args.initial_yaw_jitter_deg)
        )
        episode_initial_yaw_rad = float(np.deg2rad(episode_initial_yaw_deg))
        reset_scene(scene, episode_initial_yaw_rad)
        startup_action = torch.zeros((1, 16), device=robot.device)
        warmup_max_joint_target_jump = 0.0
        for _ in range(args.warmup_steps):
            warmup_action = torch.zeros_like(startup_action)
            robot.set_joint_position_target(
                default_pose,
                joint_ids=leg_ids,
            )
            robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_ids)
            robot.set_joint_effort_target(zero_wheels, joint_ids=wheel_ids)
            for physics_substep in range(4):
                # Explicit actuators must recompute and write PD effort at the
                # 200 Hz physics rate. Writing once per 50 Hz control frame
                # leaves three substeps with stale effort and excites the
                # startup oscillation before the policy sees its first frame.
                scene.write_data_to_sim()
                sim.step(render=physics_substep == 3)
                scene.update(physics_dt)
            startup_action = warmup_action
        warmup_final_roll_deg, warmup_final_pitch_deg = roll_pitch_from_quaternion(
            robot.data.root_quat_w[0]
        )
        warmup_final_angular_speed = float(
            torch.linalg.vector_norm(robot.data.root_ang_vel_b[0]).item()
        )
        warmup_final_leg_symmetry_error = leg_symmetry_error(robot, leg_ids)
        episode_stem = (
            f"episode_{INDOOR_SELECTION['episode']['id']}"
            if INDOOR_SELECTION is not None
            else f"episode_{episode_id:04d}"
        )
        final_path = args.output_dir / f"{episode_stem}.h5"
        final_video_path = args.video_dir / f"{episode_stem}.mp4"
        # Keep incomplete recorder output visibly separate from finalized
        # episodes. A killed Kit process can then be retried without treating
        # a truncated file as a valid demonstration.
        path = args.output_dir / f".{episode_stem}.part.h5"
        video_path = args.video_dir / f".{episode_stem}.part.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video writer: {video_path}")
        last_action = startup_action.clone()
        dagger_last_action = startup_action.clone()
        previous_execution_action = startup_action.clone()
        min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
        initial_roll_deg, initial_pitch_deg = roll_pitch_from_quaternion(
            robot.data.root_quat_w[0]
        )
        initial_angular_speed = float(
            torch.linalg.vector_norm(robot.data.root_ang_vel_b[0]).item()
        )
        initial_leg_symmetry_error = leg_symmetry_error(robot, leg_ids)
        root_height_sum = 0.0
        root_height_square_sum = 0.0
        max_abs_roll_deg = abs(initial_roll_deg)
        max_abs_pitch_deg = abs(initial_pitch_deg)
        max_body_angular_speed = initial_angular_speed
        max_leg_symmetry_error = initial_leg_symmetry_error
        max_joint_target_jump = warmup_max_joint_target_jump
        startup_max_abs_roll_deg = abs(initial_roll_deg)
        startup_max_abs_pitch_deg = abs(initial_pitch_deg)
        startup_max_body_angular_speed = initial_angular_speed
        startup_max_leg_symmetry_error = initial_leg_symmetry_error
        startup_max_joint_target_jump = warmup_max_joint_target_jump
        terminated_steps = 0
        start_x = float(robot.data.root_pos_w[0, 0].item())
        start_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        start_yaw = float(np.arctan2(2.0 * (start_quat[0] * start_quat[3] + start_quat[1] * start_quat[2]), 1.0 - 2.0 * (start_quat[2] ** 2 + start_quat[3] ** 2)))
        yaw_delta = 0.0
        target_reached = not TARGET_PRESENT
        target_reached_step = None
        stop_latched = False
        stop_latched_step = None
        stop_triggered_step = None
        smolvla_stop_latched = False
        smolvla_stop_latched_step = None
        smolvla_stop_armed_step = None
        smolvla_stop_streak = 0
        smolvla_cached_command = (0.0, 0.0, 0.0)
        smolvla_cached_action = np.zeros(6, dtype=np.float32)
        smolvla_stop_score = 0.0
        smolvla_stop_votes = 0
        smolvla_max_stop_score = 0.0
        smolvla_max_stop_votes = 0
        smolvla_inference_count = 0
        low_speed_streak = 0
        post_stop_target_hold_streak = 0
        max_post_stop_target_hold_steps = 0
        path_length = 0.0
        previous_xy = robot.data.root_pos_w[0, :2].clone()
        min_target_distance = float("inf")
        cached_navigation_command: tuple[float, float, float] | None = None
        navigation_command_until = -1
        initial_target_heading = float(
            np.arctan2(args.target_y - args.initial_y, args.target_x - args.initial_x)
        )
        initial_heading_error = float(
            np.arctan2(
                np.sin(initial_target_heading - episode_initial_yaw_rad),
                np.cos(initial_target_heading - episode_initial_yaw_rad),
            )
        )
        target_turn_sign = 1.0 if initial_heading_error >= 0.0 else -1.0
        expert_intervention_steps = 0
        stochastic_intervention_steps = 0
        forced_stop_intervention_steps = 0
        smolvla_visible_intervention_steps = 0
        smolvla_stability_intervention_steps = 0
        smolvla_target_intervention_steps = 0
        learner_skill_counts = {name: 0 for name in SKILL_NAMES}
        target_visible_at_start: bool | None = None
        max_target_pixel_fraction = 0.0
        with h5py.File(path, "w") as h5:
            obs = h5.create_group("observation")
            front_ds = obs.create_dataset("front_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            rear_ds = obs.create_dataset("rear_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            front_target_fraction_ds = obs.create_dataset(
                "front_target_pixel_fraction", (args.steps,), dtype="f4"
            )
            rear_target_fraction_ds = obs.create_dataset(
                "rear_target_pixel_fraction", (args.steps,), dtype="f4"
            )
            lidar_ds = obs.create_dataset("lidar", (args.steps, ray_count), dtype="f4", compression="lzf")
            proprio_ds = obs.create_dataset("proprio", (args.steps, 57), dtype="f4", compression="lzf")
            smolvla_proprio_ds = obs.create_dataset(
                "smolvla_proprio", (args.steps, 8), dtype="f4", compression="lzf"
            )
            state_ds = obs.create_dataset("state", (args.steps, 45), dtype="f4", compression="lzf")
            action_ds = h5.create_dataset("action", (args.steps, 16), dtype="f4", compression="lzf")
            command_ds = h5.create_dataset("expert_command", (args.steps, 3), dtype="f4", compression="lzf")
            high_level_action_ds = h5.create_dataset(
                "high_level_action", (args.steps, 6), dtype="f4", compression="lzf"
            )
            smolvla_learner_action_ds = (
                h5.create_dataset(
                    "smolvla_learner_action", (args.steps, 6), dtype="f4", compression="lzf"
                )
                if args.smolvla_dagger_labels
                else None
            )
            smolvla_stop_score_ds = (
                h5.create_dataset("smolvla_stop_score", (args.steps,), dtype="f4")
                if smolvla_model is not None
                else None
            )
            smolvla_stop_votes_ds = (
                h5.create_dataset("smolvla_stop_votes", (args.steps,), dtype="u1")
                if smolvla_model is not None
                else None
            )
            smolvla_execution_command_ds = (
                h5.create_dataset(
                    "smolvla_execution_command",
                    (args.steps, 3),
                    dtype="f4",
                    compression="lzf",
                )
                if smolvla_model is not None
                else None
            )
            learner_command_ds = (
                h5.create_dataset("learner_command", (args.steps, 3), dtype="f4", compression="lzf")
                if dagger_skill_model is not None
                else None
            )
            expert_intervention_ds = (
                h5.create_dataset("expert_intervention", (args.steps,), dtype="u1")
                if dagger_skill_model is not None
                else None
            )
            done_ds = h5.create_dataset("terminated", (args.steps,), dtype="u1")
            timestamp_ds = h5.create_dataset("timestamp", (args.steps,), dtype="f8")
            frame_index_ds = h5.create_dataset("frame_index", (args.steps,), dtype="i8")
            h5.attrs["task_text"] = args.task_text
            h5.attrs["command"] = np.asarray([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
            h5.attrs["stop_after"] = -1 if args.stop_after is None else args.stop_after
            h5.attrs["stop_on_target"] = args.stop_on_target
            h5.attrs["navigate_to_target"] = args.navigate_to_target
            h5.attrs["success_radius"] = args.success_radius
            h5.attrs["success_final_tolerance"] = args.success_final_tolerance
            h5.attrs["target_color"] = args.target_color
            h5.attrs["target_xy"] = np.asarray([args.target_x, args.target_y], dtype=np.float32)
            h5.attrs["initial_xy"] = np.asarray([args.initial_x, args.initial_y], dtype=np.float32)
            h5.attrs["initial_yaw_deg"] = episode_initial_yaw_deg
            h5.attrs["random_seed"] = args.seed
            h5.attrs["wheel_damping"] = WHEEL_DAMPING
            h5.attrs["implicit_wheel_damping"] = args.implicit_wheel_damping
            h5.attrs["wheel_actuator"] = metadata["wheel_actuator"]
            h5.attrs["fall_height_threshold"] = args.fall_height_threshold
            h5.attrs["warmup_steps"] = args.warmup_steps
            h5.attrs["startup_action_blend_steps"] = args.startup_action_blend_steps
            h5.attrs["posture_min_root_height"] = args.posture_min_root_height
            h5.attrs["posture_max_roll_deg"] = args.posture_max_roll_deg
            h5.attrs["posture_max_pitch_deg"] = args.posture_max_pitch_deg
            h5.attrs["posture_max_root_height_std"] = args.posture_max_root_height_std
            h5.attrs["startup_posture_steps"] = args.startup_posture_steps
            h5.attrs["startup_max_roll_deg"] = args.startup_max_roll_deg
            h5.attrs["startup_max_pitch_deg"] = args.startup_max_pitch_deg
            h5.attrs["startup_max_angular_speed"] = args.startup_max_angular_speed
            h5.attrs["startup_max_joint_target_jump_threshold"] = (
                args.startup_max_joint_target_jump
            )
            h5.attrs["startup_max_leg_symmetry_error_threshold"] = (
                args.startup_max_leg_symmetry_error
            )
            h5.attrs["expert"] = metadata["expert"]
            h5.attrs["asset_usd_path"] = metadata["asset_usd_path"]
            h5.attrs["control_hz"] = metadata["control_hz"]
            h5.attrs["sensor_alignment"] = metadata["sensor_alignment"]
            h5.attrs["smolvla_state_schema"] = (
                "body_linear_velocity_3,body_angular_velocity_3,projected_gravity_xy_2,lidar_sector_min_24"
            )
            h5.attrs["smolvla_dagger_labels"] = args.smolvla_dagger_labels
            h5.attrs["smolvla_ensemble_size"] = args.smolvla_ensemble_size
            h5.attrs["smolvla_inference_seed"] = args.smolvla_inference_seed
            h5.attrs["smolvla_stop_min_votes"] = args.smolvla_stop_min_votes
            h5.attrs["smolvla_stop_threshold"] = args.smolvla_stop_threshold
            h5.attrs["smolvla_stop_confirm_steps"] = args.smolvla_stop_confirm_steps
            h5.attrs["smolvla_stop_approach_steps"] = args.smolvla_stop_approach_steps
            h5.attrs["smolvla_stop_approach_max_forward"] = (
                args.smolvla_stop_approach_max_forward
            )
            h5.attrs["smolvla_command_max_forward"] = args.smolvla_command_max_forward
            h5.attrs["smolvla_command_max_yaw"] = args.smolvla_command_max_yaw
            h5.attrs["smolvla_command_smoothing"] = args.smolvla_command_smoothing
            h5.attrs["smolvla_dagger_expert_alpha"] = args.smolvla_dagger_expert_alpha
            h5.attrs["smolvla_dagger_visible_intervention_fraction"] = (
                args.smolvla_dagger_visible_intervention_fraction
            )
            h5.attrs["smolvla_dagger_visible_intervention_armed_fraction"] = (
                args.smolvla_dagger_visible_intervention_armed_fraction
            )
            h5.attrs["camera_pitch_deg"] = args.camera_pitch_deg
            h5.attrs["camera_focal_length"] = args.camera_focal_length
            h5.attrs["lidar_mesh_scope"] = metadata["lidar_mesh_scope"]
            h5.attrs["smolvla_candidate"] = metadata["smolvla_candidate"]
            h5.attrs["high_level_action_schema"] = json.dumps(
                metadata["high_level_action"], sort_keys=True
            )
            if scenario_metadata is not None:
                for key, value in scenario_metadata.items():
                    h5.attrs[key] = value
            h5.attrs["dagger"] = metadata["dagger"]
            h5.attrs["dagger_alpha"] = args.dagger_alpha
            h5.attrs["dagger_skill"] = metadata["dagger_skill"]
            h5.attrs["dagger_skill_checkpoint"] = metadata["dagger_skill_checkpoint"] or ""
            h5.attrs["dagger_skill_expert_probability"] = args.dagger_skill_expert_probability
            for step in range(args.steps):
                target_in_range = False
                if TARGET_PRESENT:
                    target_delta = robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                    target_distance_now = float(torch.linalg.vector_norm(target_delta).item())
                    target_in_range = target_distance_now <= args.success_radius
                    if target_in_range and not target_reached:
                        target_reached = True
                        target_reached_step = step
                if args.navigate_to_target:
                    if args.nav_fixed_turn_steps > 0 and step < args.nav_fixed_turn_steps and not target_reached:
                        cached_navigation_command = (0.0, 0.0, target_turn_sign * args.nav_max_yaw)
                        navigation_command_until = step + 1
                    elif args.nav_fixed_turn_steps > 0 and step >= args.nav_fixed_turn_steps and not target_reached:
                        cached_navigation_command = (args.nav_forward_speed, 0.0, 0.0)
                        navigation_command_until = args.steps + 1
                    elif cached_navigation_command is None or step >= navigation_command_until or target_reached:
                        cached_navigation_command = navigation_command(robot, target_reached)
                        navigation_command_until = step + args.nav_command_hold_steps
                    expert_command = cached_navigation_command
                else:
                    expert_command = (
                        (0.0, 0.0, 0.0)
                        if args.stop_on_target and target_reached
                        else command_for_step(step)
                    )
                # Keep every modality on the same pre-action timestamp as the
                # proprioception and expert action. The post-step frame is
                # reserved for the inspectable video only.
                front_observation = rgb(front)
                rear_observation = rgb(rear)
                front_target_fraction = target_pixel_fraction(front)
                rear_target_fraction = target_pixel_fraction(rear)
                if step == 0 and INDOOR_SELECTION is not None:
                    target_visible_at_start = bool(
                        front_target_fraction > 0.0 or rear_target_fraction > 0.0
                    )
                max_target_pixel_fraction = max(
                    max_target_pixel_fraction,
                    front_target_fraction,
                    rear_target_fraction,
                )
                lidar_observation = scan(lidar)
                smolvla_proprio = smolvla_body_proprio(robot)
                state_observation = torch.cat(
                    (
                        robot.data.root_pos_w[0],
                        robot.data.root_quat_w[0],
                        robot.data.root_lin_vel_w[0],
                        robot.data.root_ang_vel_w[0],
                        robot.data.joint_pos[0],
                        robot.data.joint_vel[0],
                    )
                ).cpu().numpy().astype(np.float32)
                # VLA always receives the robot's physical (unmirrored)
                # proprio frame. Mirroring is an implementation detail of
                # the public ONNX expert for negative-yaw commands only.
                policy_last_action = (
                    dagger_last_action
                    if dagger_model is not None or dagger_skill_model is not None
                    else last_action
                )
                vla_observation = native_observation(
                    robot,
                    joint_ids,
                    policy_last_action,
                    expert_command,
                    mirror=False,
                )
                vla_action = None
                learner_command = expert_command
                learner_skill = "stop" if target_reached else "forward"
                expert_intervention = False
                if smolvla_model is not None:
                    if step % args.smolvla_action_hold_steps == 0:
                        smol_state = smolvla_state(smolvla_proprio, lidar_observation)
                        smol_raw = {
                            "observation.state": torch.from_numpy(smol_state),
                            "observation.images.camera1": torch.from_numpy(
                                front_observation
                            ).permute(2, 0, 1).float().div_(255.0),
                            "observation.images.camera2": torch.from_numpy(
                                rear_observation
                            ).permute(2, 0, 1).float().div_(255.0),
                            "task": args.task_text,
                        }
                        smol_batch = smolvla_preprocessor(smol_raw)
                        ensemble_actions = []
                        query_seed = (
                            args.smolvla_inference_seed
                            + 1_000_003 * (episode_id + 1)
                            + 1_009 * smolvla_inference_count
                        )
                        for ensemble_index in range(args.smolvla_ensemble_size):
                            sample_seed = query_seed + ensemble_index
                            torch.manual_seed(sample_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(sample_seed)
                            smolvla_model.reset()
                            with torch.inference_mode():
                                smol_prediction = smolvla_postprocessor(
                                    smolvla_model.select_action(smol_batch)
                                )
                            sample_action = (
                                smol_prediction.detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                                .astype(np.float32)
                            )
                            ensemble_actions.append(
                                np.nan_to_num(
                                    sample_action, nan=0.0, posinf=0.0, neginf=0.0
                                )
                            )
                        ensemble_actions = np.asarray(ensemble_actions, dtype=np.float32)
                        smolvla_cached_action = ensemble_actions.mean(axis=0)
                        smolvla_stop_score = float(ensemble_actions[:, 3].max())
                        smolvla_stop_votes = int(
                            np.count_nonzero(
                                ensemble_actions[:, 3] >= args.smolvla_stop_threshold
                            )
                        )
                        smolvla_max_stop_score = max(
                            smolvla_max_stop_score, smolvla_stop_score
                        )
                        smolvla_max_stop_votes = max(
                            smolvla_max_stop_votes, smolvla_stop_votes
                        )
                        raw_command = (
                            float(
                                np.clip(
                                    smolvla_cached_action[0],
                                    -args.smolvla_command_max_forward,
                                    args.smolvla_command_max_forward,
                                )
                            ),
                            float(
                                np.clip(
                                    smolvla_cached_action[1],
                                    -args.smolvla_command_max_forward,
                                    args.smolvla_command_max_forward,
                                )
                            ),
                            float(
                                np.clip(
                                    smolvla_cached_action[2],
                                    -args.smolvla_command_max_yaw,
                                    args.smolvla_command_max_yaw,
                                )
                            ),
                        )
                        command_alpha = args.smolvla_command_smoothing
                        smolvla_cached_command = tuple(
                            (1.0 - command_alpha) * previous_command
                            + command_alpha * new_command
                            for previous_command, new_command in zip(
                                smolvla_cached_command, raw_command
                            )
                        )
                        smolvla_inference_count += 1
                        if smolvla_stop_votes >= args.smolvla_stop_min_votes:
                            smolvla_stop_streak += 1
                        else:
                            smolvla_stop_streak = 0
                        if (
                            smolvla_stop_streak >= args.smolvla_stop_confirm_steps
                            and smolvla_stop_armed_step is None
                        ):
                            smolvla_stop_armed_step = step
                        if (
                            smolvla_stop_armed_step is not None
                            and step - smolvla_stop_armed_step
                            >= args.smolvla_stop_approach_steps
                            and not smolvla_stop_latched
                        ):
                            smolvla_stop_latched = True
                            smolvla_stop_latched_step = step
                            stop_triggered_step = step
                    learner_command = smolvla_cached_command
                    if (
                        smolvla_stop_armed_step is not None
                        and not smolvla_stop_latched
                    ):
                        learner_command = (
                            float(
                                np.clip(
                                    learner_command[0],
                                    -args.smolvla_stop_approach_max_forward,
                                    args.smolvla_stop_approach_max_forward,
                                )
                            ),
                            learner_command[1],
                            learner_command[2],
                        )
                    learner_skill = "stop" if smolvla_stop_latched else "forward"
                elif dagger_model is not None:
                    vla_observation = native_observation(
                        robot, joint_ids, dagger_last_action, expert_command, mirror=False
                    )
                    vla_observation = vla_observation.clone()
                    vla_observation[:, 6:9] = 0.0
                    rgb_input = np.concatenate(
                        (
                            cv2.resize(front_observation, (80, 48), interpolation=cv2.INTER_AREA),
                            cv2.resize(rear_observation, (80, 48), interpolation=cv2.INTER_AREA),
                        ),
                        axis=-1,
                    ).transpose(2, 0, 1)
                    rgb_input = torch.from_numpy(rgb_input.copy()).float().div_(255.0).unsqueeze(0).to(dagger_device)
                    lidar_input = torch.from_numpy(lidar_observation.copy()).float().div_(20.0).unsqueeze(0).to(dagger_device)
                    language_input = encode_text(args.task_text).to(dagger_device).unsqueeze(0)
                    with torch.inference_mode():
                        vla_action = dagger_model(
                            rgb_input,
                            lidar_input,
                            vla_observation.to(dagger_device),
                            language_input,
                        )[:, 0].to(robot.device)
                elif dagger_skill_model is not None:
                    vla_observation = vla_observation.clone()
                    vla_observation[:, 6:9] = 0.0
                    vla_observation[:, 41:57] = 0.0
                    rgb_input = np.concatenate(
                        (
                            cv2.resize(front_observation, (80, 48), interpolation=cv2.INTER_AREA),
                            cv2.resize(rear_observation, (80, 48), interpolation=cv2.INTER_AREA),
                        ),
                        axis=-1,
                    ).transpose(2, 0, 1)
                    rgb_input = (
                        torch.from_numpy(rgb_input.copy())
                        .float()
                        .div_(255.0)
                        .unsqueeze(0)
                        .to(dagger_device)
                    )
                    lidar_input = (
                        torch.from_numpy(lidar_observation.copy())
                        .float()
                        .div_(20.0)
                        .unsqueeze(0)
                        .to(dagger_device)
                    )
                    language_input = encode_text(args.task_text).to(dagger_device).unsqueeze(0)
                    with torch.inference_mode():
                        command_norm, skill_logits = dagger_skill_model(
                            rgb_input,
                            lidar_input,
                            vla_observation.to(dagger_device),
                            language_input,
                        )[:2]
                    learner_command, learner_skill = canonical_skill_command(
                        command_norm,
                        skill_logits,
                    )
                    learner_skill_counts[learner_skill] += 1
                    forced_stop_intervention = target_reached
                    stochastic_intervention = (
                        not forced_stop_intervention
                        and bool(
                            intervention_rng.random()
                            < args.dagger_skill_expert_probability
                        )
                    )
                    expert_intervention = forced_stop_intervention or stochastic_intervention
                    expert_intervention_steps += int(expert_intervention)
                    stochastic_intervention_steps += int(stochastic_intervention)
                    forced_stop_intervention_steps += int(forced_stop_intervention)
                execution_command = (
                    (
                        tuple(
                            (1.0 - args.smolvla_dagger_expert_alpha) * learner_value
                            + args.smolvla_dagger_expert_alpha * expert_value
                            for learner_value, expert_value in zip(learner_command, expert_command)
                        )
                        if args.smolvla_dagger_labels
                    else learner_command
                    )
                    if smolvla_model is not None
                    else (
                        expert_command
                        if dagger_skill_model is None or expert_intervention
                        else learner_command
                    )
                )
                visible_intervention = bool(
                    smolvla_model is not None
                    and args.smolvla_dagger_labels
                    and args.smolvla_dagger_visible_intervention_fraction > 0.0
                    and not target_reached
                    and max_target_pixel_fraction
                    >= args.smolvla_dagger_visible_intervention_armed_fraction
                    and max(front_target_fraction, rear_target_fraction)
                    < args.smolvla_dagger_visible_intervention_fraction
                )
                if visible_intervention:
                    execution_command = expert_command
                    smolvla_visible_intervention_steps += 1
                stability_intervention = bool(
                    smolvla_model is not None
                    and args.smolvla_dagger_labels
                    and args.smolvla_dagger_stability_intervention_height > 0.0
                    and float(robot.data.root_pos_w[0, 2].item())
                    < args.smolvla_dagger_stability_intervention_height
                )
                if stability_intervention:
                    execution_command = expert_command
                    smolvla_stability_intervention_steps += 1
                target_intervention = bool(
                    smolvla_model is not None
                    and args.smolvla_dagger_labels
                    and target_reached
                )
                if target_intervention:
                    execution_command = expert_command
                    smolvla_target_intervention_steps += 1
                executed_command = execution_command
                # Target pose is evaluation/label truth only. A learner-only
                # SmolVLA replay may brake only after its own stop latch.
                wheel_stop_mode = (
                    smolvla_stop_latched if smolvla_model is not None else target_reached
                )
                planar_speed = float(
                    torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item()
                )
                if (
                    smolvla_model is None
                    and args.navigate_to_target
                    and target_reached
                    and not stop_latched
                ):
                    if planar_speed <= args.stop_speed_threshold:
                        low_speed_streak += 1
                    else:
                        low_speed_streak = 0
                    if low_speed_streak >= args.stop_confirm_steps:
                        stop_latched = True
                        stop_latched_step = step
                        stop_triggered_step = step
                # During pre-latch braking, preserve the last moving leg
                # targets. After latching, switch to the symmetric nominal
                # pose because the released policy is unstable at zero speed.
                stop_now = (
                    (args.stop_after is not None and step >= args.stop_after)
                    or (
                        args.stop_on_target
                        and stop_latched
                        and smolvla_model is None
                    )
                    or smolvla_stop_latched
                )
                if stop_now:
                    executed_command = (0.0, 0.0, 0.0)
                set_navigation_wheel_damping(robot, executed_command, wheel_stop_mode)
                if stop_now:
                    # The released ONNX policy is unstable for a zero command.
                    # Hold its symmetric nominal pose and lock only the wheels.
                    expert_action = torch.zeros((1, 16), device=robot.device)
                elif args.navigate_to_target and wheel_stop_mode and args.override_navigation_wheels:
                    # Preserve the last stable leg targets while wheel braking
                    # removes residual motion before the final standing pose.
                    expert_action = policy_last_action.clone()
                    expert_action = override_navigation_wheels(
                        expert_action,
                        target_stop_command(robot),
                        policy_last_action,
                    )
                else:
                    expert_action = public_expert_action(
                        session,
                        robot,
                        joint_ids,
                        policy_last_action,
                        executed_command,
                        False,
                    )
                execution_action = expert_action
                if vla_action is not None:
                    execution_action = (
                        args.dagger_alpha * expert_action + (1.0 - args.dagger_alpha) * vla_action
                    ).clamp(-4.0, 4.0)
                elif dagger_skill_model is not None and not expert_intervention and not stop_now:
                    execution_action = public_expert_action(
                        session,
                        robot,
                        joint_ids,
                        policy_last_action,
                        executed_command,
                        wheel_stop_mode,
                    )
                if stop_now:
                    # Do not let a learner reintroduce motion after stop.
                    execution_action = expert_action
                elif step < args.startup_action_blend_steps:
                    startup_action_alpha = (
                        float(step + 1) / args.startup_action_blend_steps
                    )
                    execution_action = execution_action.clone()
                    execution_action[:, :12] *= startup_action_alpha
                    if vla_action is None and dagger_skill_model is None:
                        expert_action = execution_action
                joint_target_jump = float(
                    torch.max(
                        torch.abs(
                            (execution_action[:, :12] - previous_execution_action[:, :12])
                            * leg_scale
                        )
                    ).item()
                )
                max_joint_target_jump = max(max_joint_target_jump, joint_target_jump)
                if step < args.startup_posture_steps:
                    startup_max_joint_target_jump = max(
                        startup_max_joint_target_jump,
                        joint_target_jump,
                    )
                robot.set_joint_position_target(default_pose + execution_action[:, :12] * leg_scale, joint_ids=leg_ids)
                robot.set_joint_velocity_target(execution_action[:, 12:] * 5.0, joint_ids=wheel_ids)
                robot.set_joint_effort_target(zero_wheels, joint_ids=wheel_ids)
                last_action = expert_action
                dagger_last_action = execution_action
                previous_execution_action = execution_action.clone()
                camera_target = robot.data.root_pos_w + quat_apply(
                    robot.data.root_quat_w,
                    torch.tensor([[1.2, 0.0, 0.15]], device=robot.device),
                )
                camera_eye = robot.data.root_pos_w + quat_apply(
                    robot.data.root_quat_w,
                    torch.tensor([[-1.7, 1.25, 1.15]], device=robot.device),
                )
                third.set_world_poses_from_view(camera_eye, camera_target)
                for _ in range(4):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(physics_dt)
                frame = rgb(third)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                front_ds[step], rear_ds[step], lidar_ds[step] = (
                    front_observation,
                    rear_observation,
                    lidar_observation,
                )
                proprio_ds[step], state_ds[step], action_ds[step] = (
                    vla_observation[0].cpu().numpy(),
                    state_observation,
                    expert_action[0].cpu().numpy(),
                )
                smolvla_proprio_ds[step] = smolvla_proprio
                if smolvla_stop_score_ds is not None:
                    smolvla_stop_score_ds[step] = smolvla_stop_score
                if smolvla_stop_votes_ds is not None:
                    smolvla_stop_votes_ds[step] = smolvla_stop_votes
                if smolvla_execution_command_ds is not None:
                    smolvla_execution_command_ds[step] = np.asarray(
                        executed_command, dtype=np.float32
                    )
                command_ds[step] = np.asarray(expert_command, dtype=np.float32)
                front_target_fraction_ds[step] = front_target_fraction
                rear_target_fraction_ds[step] = rear_target_fraction
                if smolvla_model is not None:
                    if smolvla_learner_action_ds is not None:
                        smolvla_learner_action_ds[step] = smolvla_cached_action
                    if args.smolvla_dagger_labels:
                        high_level_action_ds[step] = np.asarray(
                            [
                                expert_command[0],
                                expert_command[1],
                                expert_command[2],
                                float(target_reached),
                                0.0,
                                0.0,
                            ],
                            dtype=np.float32,
                        )
                    else:
                        high_level_action_ds[step] = smolvla_cached_action
                else:
                    high_level_action_ds[step] = np.asarray(
                        [
                            expert_command[0],
                            expert_command[1],
                            expert_command[2],
                            float(stop_now or target_reached),
                            0.0,
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                timestamp_ds[step] = step / metadata["control_hz"]
                frame_index_ds[step] = step
                if learner_command_ds is not None and expert_intervention_ds is not None:
                    learner_command_ds[step] = np.asarray(learner_command, dtype=np.float32)
                    expert_intervention_ds[step] = int(expert_intervention)
                current_xy = robot.data.root_pos_w[0, :2]
                path_length += float(torch.linalg.vector_norm(current_xy - previous_xy).item())
                previous_xy = current_xy.clone()
                height = float(robot.data.root_pos_w[0, 2].item())
                min_height, max_height = min(min_height, height), max(max_height, height)
                root_height_sum += height
                root_height_square_sum += height * height
                roll_deg, pitch_deg = roll_pitch_from_quaternion(
                    robot.data.root_quat_w[0]
                )
                body_angular_speed = float(
                    torch.linalg.vector_norm(robot.data.root_ang_vel_b[0]).item()
                )
                symmetry_error = leg_symmetry_error(robot, leg_ids)
                max_abs_roll_deg = max(max_abs_roll_deg, abs(roll_deg))
                max_abs_pitch_deg = max(max_abs_pitch_deg, abs(pitch_deg))
                max_body_angular_speed = max(
                    max_body_angular_speed,
                    body_angular_speed,
                )
                max_leg_symmetry_error = max(
                    max_leg_symmetry_error,
                    symmetry_error,
                )
                if step < args.startup_posture_steps:
                    startup_max_abs_roll_deg = max(
                        startup_max_abs_roll_deg,
                        abs(roll_deg),
                    )
                    startup_max_abs_pitch_deg = max(
                        startup_max_abs_pitch_deg,
                        abs(pitch_deg),
                    )
                    startup_max_body_angular_speed = max(
                        startup_max_body_angular_speed,
                        body_angular_speed,
                    )
                    startup_max_leg_symmetry_error = max(
                        startup_max_leg_symmetry_error,
                        symmetry_error,
                    )
                if TARGET_PRESENT:
                    target_delta = robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                    target_distance = float(torch.linalg.vector_norm(target_delta).item())
                    min_target_distance = min(min_target_distance, target_distance)
                    if target_distance <= args.success_radius and not target_reached:
                        target_reached = True
                        target_reached_step = step
                    effective_stop_latched = (
                        smolvla_stop_latched if smolvla_model is not None else stop_latched
                    )
                    if (
                        effective_stop_latched
                        and target_distance
                        <= args.success_radius + args.success_final_tolerance
                    ):
                        post_stop_target_hold_streak += 1
                        max_post_stop_target_hold_steps = max(
                            max_post_stop_target_hold_steps,
                            post_stop_target_hold_streak,
                        )
                    elif effective_stop_latched:
                        post_stop_target_hold_streak = 0
                gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
                terminated = int(
                    height < args.fall_height_threshold or gravity_z > -0.5
                )
                done_ds[step] = terminated
                terminated_steps += terminated
                quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                current_yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
                yaw_delta = float(np.arctan2(np.sin(current_yaw - start_yaw), np.cos(current_yaw - start_yaw)))
            displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
            root_height_mean = root_height_sum / args.steps
            root_height_std = float(
                np.sqrt(
                    max(
                        root_height_square_sum / args.steps
                        - root_height_mean * root_height_mean,
                        0.0,
                    )
                )
            )
            final_roll_deg, final_pitch_deg = roll_pitch_from_quaternion(
                robot.data.root_quat_w[0]
            )
            final_body_angular_speed = float(
                torch.linalg.vector_norm(robot.data.root_ang_vel_b[0]).item()
            )
            final_leg_symmetry_error = leg_symmetry_error(robot, leg_ids)
            startup_posture_ok = bool(
                startup_max_abs_roll_deg <= args.startup_max_roll_deg
                and startup_max_abs_pitch_deg <= args.startup_max_pitch_deg
                and startup_max_body_angular_speed
                <= args.startup_max_angular_speed
                and startup_max_joint_target_jump
                <= args.startup_max_joint_target_jump
                and startup_max_leg_symmetry_error
                <= args.startup_max_leg_symmetry_error
            )
            posture_ok = bool(
                min_height >= args.posture_min_root_height
                and max_abs_roll_deg <= args.posture_max_roll_deg
                and max_abs_pitch_deg <= args.posture_max_pitch_deg
                and root_height_std <= args.posture_max_root_height_std
                and startup_posture_ok
            )
            stable = terminated_steps == 0 and posture_ok
            final_target_distance = (
                float(torch.linalg.vector_norm(
                    robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                ).item())
                if TARGET_PRESENT else None
            )
            final_planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item())
            if args.navigate_to_target:
                effective_stop_latched = (
                    smolvla_stop_latched if smolvla_model is not None else stop_latched
                )
                command_ok = (
                    target_reached and final_target_distance is not None
                    and final_target_distance <= args.success_radius + args.success_final_tolerance
                    and final_planar_speed < 0.15
                    and effective_stop_latched
                    and max_post_stop_target_hold_steps >= args.target_hold_steps
                )
            elif abs(args.command_x) >= 0.05:
                command_ok = args.command_x * displacement > 0.5
            elif abs(args.command_yaw) >= 0.05:
                command_ok = args.command_yaw * yaw_delta > 0.25 and abs(displacement) < 2.0
            else:
                command_ok = True
            success = bool(stable and command_ok and target_reached)
            h5.attrs["x_displacement"] = displacement
            h5.attrs["yaw_delta"] = yaw_delta
            h5.attrs["min_root_height"] = min_height
            h5.attrs["max_root_height"] = max_height
            h5.attrs["root_height_mean"] = root_height_mean
            h5.attrs["root_height_std"] = root_height_std
            h5.attrs["warmup_final_roll_deg"] = warmup_final_roll_deg
            h5.attrs["warmup_final_pitch_deg"] = warmup_final_pitch_deg
            h5.attrs["warmup_final_angular_speed"] = warmup_final_angular_speed
            h5.attrs["warmup_final_leg_symmetry_error"] = (
                warmup_final_leg_symmetry_error
            )
            h5.attrs["max_abs_roll_deg"] = max_abs_roll_deg
            h5.attrs["max_abs_pitch_deg"] = max_abs_pitch_deg
            h5.attrs["max_body_angular_speed"] = max_body_angular_speed
            h5.attrs["max_leg_symmetry_error"] = max_leg_symmetry_error
            h5.attrs["max_joint_target_jump"] = max_joint_target_jump
            h5.attrs["startup_max_abs_roll_deg"] = startup_max_abs_roll_deg
            h5.attrs["startup_max_abs_pitch_deg"] = startup_max_abs_pitch_deg
            h5.attrs["startup_max_body_angular_speed"] = (
                startup_max_body_angular_speed
            )
            h5.attrs["startup_max_leg_symmetry_error"] = (
                startup_max_leg_symmetry_error
            )
            h5.attrs["startup_max_joint_target_jump"] = (
                startup_max_joint_target_jump
            )
            h5.attrs["final_roll_deg"] = final_roll_deg
            h5.attrs["final_pitch_deg"] = final_pitch_deg
            h5.attrs["final_body_angular_speed"] = final_body_angular_speed
            h5.attrs["final_leg_symmetry_error"] = final_leg_symmetry_error
            h5.attrs["startup_posture_ok"] = startup_posture_ok
            h5.attrs["posture_ok"] = posture_ok
            h5.attrs["terminated_steps"] = terminated_steps
            h5.attrs["command_ok"] = command_ok
            h5.attrs["target_reached"] = target_reached
            h5.attrs["target_reached_step"] = -1 if target_reached_step is None else target_reached_step
            h5.attrs["stop_latched"] = stop_latched
            h5.attrs["stop_latched_step"] = -1 if stop_latched_step is None else stop_latched_step
            h5.attrs["smolvla_stop_latched"] = smolvla_stop_latched
            h5.attrs["smolvla_stop_armed_step"] = (
                -1 if smolvla_stop_armed_step is None else smolvla_stop_armed_step
            )
            h5.attrs["smolvla_stop_latched_step"] = (
                -1 if smolvla_stop_latched_step is None else smolvla_stop_latched_step
            )
            h5.attrs["smolvla_inference_count"] = smolvla_inference_count
            h5.attrs["smolvla_stop_score_max"] = smolvla_max_stop_score
            h5.attrs["smolvla_stop_votes_max"] = smolvla_max_stop_votes
            h5.attrs["stop_pretrigger_radius"] = args.stop_pretrigger_radius
            h5.attrs["max_post_stop_target_hold_steps"] = max_post_stop_target_hold_steps
            h5.attrs["min_target_distance"] = min_target_distance if TARGET_PRESENT else -1.0
            h5.attrs["final_target_distance"] = -1.0 if final_target_distance is None else final_target_distance
            h5.attrs["final_planar_speed"] = final_planar_speed
            h5.attrs["path_length"] = path_length
            h5.attrs["success"] = success
            h5.attrs["target_visible_at_start"] = bool(target_visible_at_start)
            h5.attrs["max_target_pixel_fraction"] = max_target_pixel_fraction
            h5.attrs["dagger_skill_expert_interventions"] = expert_intervention_steps
            h5.attrs["dagger_skill_stochastic_interventions"] = stochastic_intervention_steps
            h5.attrs["dagger_skill_forced_stop_interventions"] = forced_stop_intervention_steps
            h5.attrs["smolvla_visible_intervention_steps"] = smolvla_visible_intervention_steps
            h5.attrs["smolvla_stability_intervention_steps"] = smolvla_stability_intervention_steps
            h5.attrs["smolvla_target_intervention_steps"] = smolvla_target_intervention_steps
            h5.attrs["dagger_skill_learner_skill_counts"] = json.dumps(
                learner_skill_counts,
                sort_keys=True,
            )
        finalize_h264_video(video, video_path)
        os.replace(video_path, final_video_path)
        os.replace(path, final_path)
        path = final_path
        video_path = final_video_path
        displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
        stable = terminated_steps == 0 and posture_ok
        final_target_distance = (
            float(torch.linalg.vector_norm(
                robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
            ).item())
            if TARGET_PRESENT else None
        )
        final_planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item())
        if args.navigate_to_target:
            effective_stop_latched = (
                smolvla_stop_latched if smolvla_model is not None else stop_latched
            )
            command_ok = (
                target_reached and final_target_distance is not None
                and final_target_distance <= args.success_radius + args.success_final_tolerance
                and final_planar_speed < 0.15
                and effective_stop_latched
                and max_post_stop_target_hold_steps >= args.target_hold_steps
            )
        elif abs(args.command_x) >= 0.05:
            command_ok = args.command_x * displacement > 0.5
        elif abs(args.command_yaw) >= 0.05:
            command_ok = args.command_yaw * yaw_delta > 0.25 and abs(displacement) < 2.0
        else:
            command_ok = True
        success = bool(stable and command_ok and target_reached)
        metrics_path = args.output_dir / f"{episode_stem}.json"
        episode_metrics = {
            "schema": "m20pro_objectnav_collection_episode_v1",
            "episode_id": episode_stem.removeprefix("episode_"),
            "scenario": scenario_metadata,
            "task_text": args.task_text,
            "data": str(path),
            "video": str(video_path),
            "frames": args.steps,
            "x_displacement_m": displacement,
            "yaw_delta_rad": yaw_delta,
            "path_length_m": path_length,
            "min_root_height_m": min_height,
            "max_root_height_m": max_height,
            "root_height_mean_m": root_height_mean,
            "root_height_std_m": root_height_std,
            "warmup_final_roll_deg": warmup_final_roll_deg,
            "warmup_final_pitch_deg": warmup_final_pitch_deg,
            "warmup_final_angular_speed_rps": warmup_final_angular_speed,
            "warmup_final_leg_symmetry_error_rad": warmup_final_leg_symmetry_error,
            "max_abs_roll_deg": max_abs_roll_deg,
            "max_abs_pitch_deg": max_abs_pitch_deg,
            "max_body_angular_speed_rps": max_body_angular_speed,
            "max_leg_symmetry_error_rad": max_leg_symmetry_error,
            "max_joint_target_jump_rad": max_joint_target_jump,
            "startup_max_abs_roll_deg": startup_max_abs_roll_deg,
            "startup_max_abs_pitch_deg": startup_max_abs_pitch_deg,
            "startup_max_body_angular_speed_rps": startup_max_body_angular_speed,
            "startup_max_leg_symmetry_error_rad": startup_max_leg_symmetry_error,
            "startup_max_joint_target_jump_rad": startup_max_joint_target_jump,
            "final_roll_deg": final_roll_deg,
            "final_pitch_deg": final_pitch_deg,
            "final_body_angular_speed_rps": final_body_angular_speed,
            "final_leg_symmetry_error_rad": final_leg_symmetry_error,
            "startup_posture_ok": startup_posture_ok,
            "posture_ok": posture_ok,
            "terminated_steps": terminated_steps,
            "target_reached": target_reached,
            "target_reached_step": target_reached_step,
            "stop_latched": stop_latched,
            "stop_latched_step": stop_latched_step,
            "smolvla_checkpoint": (
                None if args.smolvla_checkpoint is None else str(args.smolvla_checkpoint)
            ),
            "smolvla_stop_latched": smolvla_stop_latched,
            "smolvla_stop_armed_step": smolvla_stop_armed_step,
            "smolvla_stop_latched_step": smolvla_stop_latched_step,
            "smolvla_inference_count": smolvla_inference_count,
            "smolvla_ensemble_size": args.smolvla_ensemble_size,
            "smolvla_stop_min_votes": args.smolvla_stop_min_votes,
            "smolvla_stop_threshold": args.smolvla_stop_threshold,
            "smolvla_stop_confirm_queries": args.smolvla_stop_confirm_steps,
            "smolvla_stop_score_max": smolvla_max_stop_score,
            "smolvla_stop_votes_max": smolvla_max_stop_votes,
            "smolvla_command_max_forward": args.smolvla_command_max_forward,
            "smolvla_command_max_yaw": args.smolvla_command_max_yaw,
            "smolvla_command_smoothing": args.smolvla_command_smoothing,
            "smolvla_visible_intervention_steps": smolvla_visible_intervention_steps,
            "smolvla_stability_intervention_steps": smolvla_stability_intervention_steps,
            "smolvla_target_intervention_steps": smolvla_target_intervention_steps,
            "max_post_stop_target_hold_steps": max_post_stop_target_hold_steps,
            "target_hold_steps_required": args.target_hold_steps,
            "min_target_distance_m": (
                min_target_distance if TARGET_PRESENT else None
            ),
            "final_target_distance_m": final_target_distance,
            "final_planar_speed_mps": final_planar_speed,
            "target_visible_at_start": (
                None
                if INDOOR_SELECTION is None
                else target_visible_at_start
            ),
            "max_target_pixel_fraction": (
                None
                if INDOOR_SELECTION is None
                else max_target_pixel_fraction
            ),
            "stable": stable,
            "command_ok": command_ok,
            "success": success,
        }
        metrics_path.write_text(
            json.dumps(episode_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[M20PRO-NATIVE-EXPERT] episode={episode_id} initial_yaw_deg={episode_initial_yaw_deg:.3f} "
            f"x_displacement={displacement:.4f} m yaw_delta={yaw_delta:.4f} rad "
            f"min_root_height={min_height:.4f} m root_height_std={root_height_std:.4f} m "
            f"max_roll={max_abs_roll_deg:.2f} deg startup_ang_vel={startup_max_body_angular_speed:.3f} rad/s "
            f"posture_ok={posture_ok} terminated_steps={terminated_steps} command_ok={command_ok} "
            f"target_reached={target_reached} final_target_distance={final_target_distance} "
            f"reached_step={target_reached_step} path_length={path_length:.4f} m success={success} "
            f"expert_interventions={expert_intervention_steps} "
            f"stochastic_interventions={stochastic_intervention_steps} "
            f"forced_stop_interventions={forced_stop_intervention_steps} "
            f"data={path} video={video_path} metrics={metrics_path}",
            flush=True,
        )
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()

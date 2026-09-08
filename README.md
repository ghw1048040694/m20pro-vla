# m20pro-vla

**m20pro-vla** is the public MuJoCo-first VLA research runtime for the DEEP Robotics M20 Pro platform.

It provides a reusable boundary between multimodal high-level policies and a low-level locomotion controller. The public snapshot is intended to make the simulation architecture, observation contract, action contract, and evaluation philosophy easy to understand without exposing private training outputs or deployment material.

> This repository is a compact public research snapshot. It does not include real-robot ROS bring-up, field maps, logs, datasets, checkpoints, private assets, or machine-specific deployment configuration.

## Project focus

The current direction is language-conditioned object navigation and embodied control in randomized indoor MuJoCo scenes:

- observe front/rear RGB, planar LiDAR, proprioception, and a language instruction;
- select and approach a visual target without privileged target-world coordinates;
- navigate around obstacles and recover when a target is initially occluded;
- convert a high-level action chunk into safe body motion through one reusable low-level controller;
- evaluate complete closed-loop episodes rather than treating offline loss as success.

The package also keeps the interfaces needed for future VLA policies, world-model scoring, search MPC, dataset auditing, and sim-to-sim validation.

## Architecture

```text
RGB + 72-beam LiDAR + qpos/qvel + language
                    │
                    ▼
          m20pro_vla.policies
       visual/language policy latent
                    │
                    ▼
      planning / world-model action chunks
                    │
                    ▼
 [forward, lateral, yaw, stop] body command
                    │
                    ▼
       m20pro_vla.low_level
      M20LowLevelController
                    │
                    ▼
       12 leg joints + 4 wheel joints
                    │
                    ▼
                MuJoCo scene
```

The low-level controller is the single actuation boundary. Policies and planners must submit the four-field body command instead of directly writing joint trajectories. This keeps gait, posture, braking, terrain recovery, and safety limits in one place.

## Observation and action contract

### Policy observations

The versioned contract requires:

- front RGB image;
- rear RGB image;
- 72-beam planar LiDAR;
- proprioception (`qpos/qvel` features);
- UTF-8 language instruction;
- visual history/previous-action features where enabled by the policy.

The intended policy path prohibits simulator-only privileged inputs such as target world position, target geometry ID, simulator object pose, or semantic masks.

### High-level action

```text
[forward_mps, lateral_mps, yaw_radps, stop]
```

The command is clipped, rate-limited, converted into leg position targets and wheel velocity targets, and checked for finite values, height, tilt, contact, and braking behavior. The action contract is shared by compact policies, future SmolVLA/Pi0.5 adapters, and world-model MPC.

## Package layout

| Path | Role |
| --- | --- |
| `src/m20pro_vla/sim/` | MuJoCo scene creation, RGB/LiDAR/proprioception observations, video helpers |
| `src/m20pro_vla/low_level/` | Body-command data model, locomotion controller, diagnostics, regression gate |
| `src/m20pro_vla/policies/` | Compact RGB/LiDAR/language policy implementation |
| `src/m20pro_vla/planning/` | Search/MPC action-chunk recommendation and obstacle-aware routing |
| `src/m20pro_vla/world_model/` | Trajectory/action-chunk scoring interfaces |
| `src/m20pro_vla/data/` | Visibility, history, and dataset-distribution audits |
| `src/m20pro_vla/runtime/` | Unified run context and runtime artifact bookkeeping |
| `configs/` | Versioned observation, low-level, and evaluation contracts |
| `media/` | Small public preview image only |

## Install

The package requires Python 3.11 or newer and MuJoCo 3.3.x:

```bash
python3 -m pip install -e . --no-deps
```

For the optional VLA path:

```bash
python3 -m pip install -e '.[vla]'
```

The public package intentionally does not bundle a robot asset, a trained checkpoint, a dataset, or a simulator-specific private environment. Those resources must be prepared separately and kept outside Git.

## Unified CLI

After installation, use the single lifecycle entry point:

```bash
m20pro-vla doctor
m20pro-vla prepare
m20pro-vla smoke
m20pro-vla low-level-gate
m20pro-vla report
```

Use `--dry-run` to inspect a command plan without creating run artifacts:

```bash
m20pro-vla smoke --dry-run --json
m20pro-vla low-level-gate --dry-run --json
```

The compatibility workflows expose the public lifecycle without requiring each experiment to invent a new entry point:

```bash
m20pro-vla collect --help
m20pro-vla train --help
m20pro-vla eval --help
m20pro-vla play --help
```

Normal runs write summaries under the user runtime directory, not into the source tree. Run IDs, configuration, status, and summary files are designed for local reproducibility and are excluded from the public repository.

## Public demo

A small MuJoCo object-navigation scene preview is included for a quick visual overview of the robot, target objects, and obstacles. It is illustrative only and should not be read as a success-rate, benchmark, or real-robot claim.

![M20 Pro object-navigation scene](media/m20-objectnav-scene.jpg)

## Evaluation philosophy

Evaluation is staged and closed-loop:

1. **Runtime gate** — verify dependencies, assets, contracts, and observation generation.
2. **Low-level gate** — verify finite state, forward motion, turning, stopping drift, height, and tilt.
3. **Visible object navigation** — target is visible at the start and must be reached and held for the required duration.
4. **Hidden object search** — target is initially occluded; the agent must discover it before reaching it.
5. **Generalization and obstacle tests** — evaluate disjoint layouts, unseen objects/instructions, clearance, and landing stability where applicable.

Required episode artifacts for serious experiments are per-episode JSON, aggregate JSON, and a local rendered record. A single successful preview, an offline training loss, or a policy that uses privileged coordinates is not sufficient evidence of navigation capability.

## Versioned contracts

The public configuration files make the important boundaries explicit:

- `configs/m20pro_mujoco_vla_contract_v1.yaml` — observations, prohibited privileged inputs, action interface, data requirements, and acceptance rules;
- `configs/m20pro_low_level_v1.yaml` — low-level outputs, feedback, safety behavior, and regression gates;
- `configs/m20pro_vla_eval_v1.yaml` — task splits, success definitions, reporting metrics, and staged thresholds.

When changing an input, action field, safety limit, or success definition, update the corresponding contract and the implementation together.

## Scope and limitations

This repository is the MuJoCo/VLA research side of the M20 Pro work. It is not the real-robot ROS 2 supervision system and does not contain hardware bring-up, field networking, platform credentials, or live sensor drivers.

MuJoCo is the primary development and training environment in this public path. Isaac Sim or another simulator may be used for later sim-to-sim checks, but a sim-to-sim result should only be reported after the MuJoCo observation, action, and low-level gates pass.

Current source interfaces are reusable, but they do not by themselves claim a trained policy, benchmark score, or production readiness. Exact results depend on the asset revision, scene distribution, random seed, hardware/simulator versions, and the policy checkpoint used.

## Reproducibility checklist

For each private experiment, preserve:

- code and contract revision;
- simulator and dependency versions;
- scene/asset hash and random seed;
- sensor dimensions and preprocessing;
- instruction template and train/validation/test split;
- policy checkpoint and inference device;
- per-episode success, target discovery, clearance, false-stop, and stability metrics.

Keep datasets, checkpoints, generated videos, logs, and runtime outputs in private experiment storage rather than committing them here.

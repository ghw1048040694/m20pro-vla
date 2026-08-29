# M20 Pro VLA Architecture

The repository contains two intentionally separate systems:

- `src/m20pro_*`: existing ROS 2 simulation packages.
- `src/m20pro_vla/`: the MuJoCo-first embodied-learning stack.

## Directory Contract

```text
src/m20pro_vla/
  low_level/       # sole M20 joint-level execution implementation
  sim/             # MuJoCo scene construction, sensors, assets
  policies/        # VLA and future learned locomotion adapters; no joint I/O
  data/            # dataset schemas and conversion
  utils/           # shared utilities
configs/           # versioned input/output and safety contracts
scripts/mujoco/    # compatibility MuJoCo entry points during CLI migration
scripts/           # ROS, legacy Isaac, and compatibility CLI entry points
tests/             # import and contract gates
docs/              # architecture and operational decisions
.runtime/          # ignored local datasets, checkpoints, metrics, video
assets/            # Isaac Lab asset configuration retained for compatibility
tasks/             # Isaac Lab task definitions retained for compatibility
```

## Unified CLI and Run Lifecycle

The user-facing command is `m20pro-vla`, registered by `pyproject.toml`.
`m20pro_vla.cli` owns the lifecycle boundary and calls package APIs directly
for the migrated gates:

```text
doctor -> prepare -> smoke -> low-level-gate -> collect -> audit -> convert
                                                       -> train -> eval -> play -> report
```

Each invocation writes a timestamped run directory under `.runtime/runs/` with
the exact command, git revision, configuration snapshot, status, and summary.
`collect`, `audit`, `convert`, `train`, `eval`, and `play` are currently
recorded compatibility adapters in `m20pro_vla.runtime.compat`. They stream
stdout/stderr into the same run directory and preserve exit codes while their
business logic is moved to `src/m20pro_vla`; they must not create a second
controller or a second runtime artifact convention. `report` is already
package-native and summarizes completed runs.

The ROS 2 packages under `src/` remain self-contained `ament` packages. Their
launch files, maps, meshes, and package configuration stay inside the owning
package, which keeps `colcon build` and package installation deterministic.
The detailed ownership table and migration rules are in
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md).

## Low-Level Contract

`m20pro_vla.low_level.M20LowLevelController` is the only M20 VLA execution
layer. It accepts `M20BodyCommand(forward, lateral, yaw, stop)` and owns all
16 MuJoCo actuator targets, stance feedback, acceleration limiting, braking,
and diagnostics. It must never receive language, RGB, object identity, target
coordinates, semantic masks, or world-model labels.

The current M20 wheel bridge is nonholonomic: `lateral` remains in the public
contract for future hardware/locomotion replacements but is explicitly marked
unsupported in diagnostics rather than silently claimed as sideways motion.
Flat-ground yaw uses the calibrated positive/negative target slew values in
`configs/m20pro_low_level_v1.yaml`.
The controller also feeds measured body yaw rate back into the differential
wheel target, so a sustained command does not keep increasing angular speed.

The same controller now has a terrain-contact recovery path. Named MuJoCo
terrain contacts can trigger a bounded wheel-edge lift, traction boost, stronger
stance feedback, and a tilt safety stop; this is internal execution logic and
does not add terrain labels or target information to the body-command contract.
The versioned rough-terrain gate is `m20pro_low_level_rough_terrain_gate_v1` and
is run by `scripts/mujoco/evaluate_m20_rough_terrain_controller.py`.

Every high-level policy, including SmolVLA, Pi0.5, world-model MPC, offline
experts, and evaluation replayers, must emit this body command and invoke the
same controller. A future learned rough-terrain locomotion policy may replace
the internal stance law only if it preserves this public command and diagnostic
interface and passes the low-level gates in `configs/m20pro_low_level_v1.yaml`.

## Experiment Policy

Tracked code/configuration/documentation belongs in the repository. Large
runtime artifacts belong under `.runtime/` and are not committed. Every model
claim needs a versioned config, machine-readable metric report, and decoded
video when rendering is relevant. No VLA training is allowed to bypass the
low-level controller or use target poses as policy input.

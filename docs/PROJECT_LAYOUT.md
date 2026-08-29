# M20 Pro Project Layout

This repository follows the usual robot-project split: ROS 2 packages own
runtime integration, the installable Python package owns reusable learning and
simulation code, and command-line scripts only compose those APIs.

```text
m20pro-sim/
├── src/
│   ├── m20pro_description/       # URDF, meshes, robot model package
│   ├── m20pro_bringup/           # ROS launch, Nav2 config, maps, RViz
│   ├── m20pro_navigation/        # ROS navigation and simulator nodes
│   ├── m20pro_cloud_bridge/      # local dashboard/state bridge
│   └── m20pro_vla/               # installable MuJoCo/VLA Python package
│       ├── low_level/            # one reusable joint-level execution layer
│       ├── sim/                  # MuJoCo scene and sensor contracts
│       ├── policies/             # high-level/body-command policies
│       ├── data/                 # dataset schemas and conversion helpers
│       ├── runtime/               # run IDs, snapshots, status, and artifacts
│       └── utils/                 # package-level utilities
├── configs/                      # versioned experiment and safety contracts
├── scripts/
│   ├── mujoco/                   # compatibility MuJoCo CLIs during migration
│   └── *.py, *.sh                # ROS/Isaac/legacy commands and compatibility
├── tests/                        # fast contract tests; no runtime artifacts
├── docs/                         # architecture, decisions, operational notes
├── tools/                        # maintenance and workspace tooling
├── assets/                       # tracked Isaac Lab asset configuration only
├── tasks/                        # tracked Isaac Lab task definitions only
└── .runtime/                     # ignored local cache, data, videos, checkpoints
```

## Ownership Rules

| Area | May contain | Must not contain |
| --- | --- | --- |
| `src/m20pro_description` | URDF, meshes, package-local robot visuals/collisions | checkpoints or generated USD |
| `src/m20pro_bringup` | launch files, Nav2 parameters, maps, RViz layouts | learning code or episode data |
| `src/m20pro_navigation` | ROS nodes, message bridges, navigation algorithms | direct VLA model training |
| `src/m20pro_vla` | reusable Python APIs and contracts | command-line-only orchestration |
| `configs` | reviewed YAML/JSON contracts with `*_vN` names | mutable run outputs |
| `scripts/mujoco` | thin MuJoCo CLI entry points | a second controller implementation |
| `tests` | deterministic unit/contract tests | large videos or checkpoints |
| `.runtime` | datasets, logs, metrics, videos, generated scenes | tracked source code |

`assets/` and `tasks/` are retained because Isaac Lab scripts still import
their historical namespaces. They are compatibility areas, not the MuJoCo
mainline. New MuJoCo work belongs in `src/m20pro_vla` and `scripts/mujoco`.

## Control Boundary

Every high-level policy emits exactly one versioned body command:
`[forward, lateral, yaw, stop]`. SmolVLA, Pi0.5, world-model MPC, expert
replayers, and evaluation tools all call
`m20pro_vla.low_level.M20LowLevelController`; no policy may write MuJoCo
joint targets directly.

The current fixed-wheel M20 bridge does not support holonomic lateral motion.
The command field is retained for interface stability, and diagnostics expose
that limitation explicitly. It must not be interpreted as successful sideways
movement in a dataset or evaluation report.

## Runtime Layout

The ignored `.runtime/` tree is deliberately separated from source:

```text
.runtime/
├── assets/         # downloaded/compiled simulator assets
├── datasets/       # episode arrays and metadata
├── checkpoints/    # model checkpoints
├── diagnostics/    # smoke and gate reports
├── low_level/      # controller gate reports
├── mujoco_assets/  # generated MuJoCo XML and meshes
└── videos/         # H.264 playback/validation videos
```

Use a new versioned subdirectory for each meaningful experiment. Do not put
large outputs in the repository root, `src/`, `configs/`, or `tests/`.

## Adding a Feature

1. Put reusable behavior in the owning `src/m20pro_*` package.
2. Add or update a versioned contract under `configs/`.
3. Add reusable behavior under `src/m20pro_vla/` and register its lifecycle
   command in `m20pro_vla.cli`; keep a script only as a compatibility wrapper.
4. Add a fast test under `tests/` and write metrics under `.runtime/`.
5. Update `README.md`, `docs/ARCHITECTURE.md`, and the project log for a
   user-visible workflow or contract change.

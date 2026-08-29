# MuJoCo Mainline Commands

These are the compatibility command-line entry points for the WSL
MuJoCo-first M20 VLA workflow. The user-facing entry point is `m20pro-vla`,
which records a run context before invoking one of these adapters. They depend
on the installed `m20pro_vla` package and keep joint-level execution inside
`M20LowLevelController`.

Typical order:

```bash
m20pro-vla prepare
m20pro-vla smoke
m20pro-vla low-level-gate
m20pro-vla collect --mode randomized
m20pro-vla collect --mode search
m20pro-vla train
m20pro-vla eval --checkpoint .runtime/checkpoints/m20_mujoco_vla_v1/best.pt --output-dir .runtime/eval/m20_hidden_search_gate_v1
m20pro-vla audit --dataset .runtime/datasets/m20_mujoco_vla_v1 --strict
```

`collect`, `train`, and `play` write only to `.runtime/` by default. Use a
versioned output directory for any run that may be compared later.

For hidden-search data, keep the default `--search-start-mode balanced` unless
you are isolating a specific failure. It emits both `anchor` starts, which
preserve stable hidden-search success signals, and `varied` starts, which widen
initial position/yaw coverage for generalization.

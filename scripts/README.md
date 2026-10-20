# Script Boundary

Users operate the project through `m20pro-vla`. Experiment differences belong
in one versioned JSON file under `configs/`; never copy a Python or shell script
for a new run.

```bash
source scripts/activate_vla_env.sh
python -m pip install -e . --no-deps
m20pro-vla experiment \
  --config configs/m20_hidden_search_experiment_v2.json \
  --stage plan
```

Root scripts are limited to environment setup and ROS simulator start/status/
stop operations. `scripts/mujoco/` contains the generic executors called by the
CLI for collection, audit, training, evaluation, and playback. They are shared
implementations, not experiment-specific entry points.

Typical lifecycle:

```bash
m20pro-vla experiment --config configs/m20_hidden_search_experiment_v2.json --stage collect
m20pro-vla experiment --config configs/m20_hidden_search_experiment_v2.json --stage audit
m20pro-vla experiment --config configs/m20_hidden_search_experiment_v2.json --stage train
m20pro-vla experiment --config configs/m20_hidden_search_experiment_v2.json --stage eval
m20pro-vla experiment --config configs/m20_hidden_search_experiment_v2.json --stage gate
```

These scripts never connect to the real robot. Real deployment begins only
after the promotion gate and an explicit shadow export.

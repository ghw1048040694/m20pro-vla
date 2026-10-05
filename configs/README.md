# Configuration Contracts

Files in this directory are reviewed, versioned inputs rather than mutable
run outputs. Use a stable domain name plus a contract version, for example
`m20pro_low_level_v1.yaml` or `experiment.json`.

- `m20pro_low_level_vN.yaml`: safety and acceptance thresholds for the shared
  body-command execution layer.
- `m20pro_mujoco_vla_contract_vN.yaml`: observation/action and data schema.
- `m20pro_vla_eval_vN.yaml`: closed-loop evaluation and reporting rules.
- `m20_*_experiment_vN.json`: one complete experiment definition covering
  paths, collection, data gates, training, evaluation, and promotion.

Metrics, checkpoints, generated scenes, and videos belong under the ignored
`.runtime/` tree, never beside a config file.

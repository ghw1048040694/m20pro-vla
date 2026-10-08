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

The existing `experiment --stage convert` stage reuses content-verified finished
LeRobot artifacts under `.runtime/cache/lerobot_conversion`. Only new or changed
NPZ/JSON episodes are encoded. Cache keys include sampling/stop weights, codec,
adapter code and LeRobot version; artifact hashes reject stale or incomplete data.
`smolvla.conversion_workers` defaults to 2 bounded CPU workers (allowed 1–4).
`smolvla.conversion_cache_dir` can override the shared cache location. These are
execution settings; they do not change action labels or acceptance thresholds.
Existing completed artifacts can be certified with `register_conversion_cache`
after their normal manifest/frame/quality checks. Certification also verifies
actual action/state/task rows. Cached subset views reference unchanged video
files but expose only the selected training rows/timestamps; merging stream-copies
video without re-encoding. Reports include reused/new episode counts and actual
episode order. Partial work never replaces an existing training dataset.

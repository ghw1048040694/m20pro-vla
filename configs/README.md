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
# Raw training backend (opt-in)

`data.raw_dataset.RawM20Dataset` prepares strided RGB arrays once without video encoding,
uses content-verified immutable caches and bounded per-worker memory mappings,
and preserves stop sampling, state projection and episode-local action padding.
Its raw pixels differ from lossy H264 decoding.

The existing `experiment --stage train-smolvla` supports
`smolvla.dataset_backend: "raw"` as an opt-in. It reads the quality-audited
`paths.dataset` directly, prepares/reuses array caches during training setup,
and calls the installed official LeRobot training loop with a process-local
dataset factory. There is no separate video conversion step or second training
CLI. `lerobot` remains the default backend and its incremental conversion cache
remains available. `smolvla.raw_cache_dir` defaults to `.runtime/cache/raw_training`;
`smolvla.raw_max_open_episodes` defaults to 4 per worker. Sampling uses the existing
source_fps/frame_stride/stop-repeat settings. Resume validates source identities,
sampling, reader/adapter code and array artifact digests against the saved run.

CPU validation covered all 206 real training episodes/85,837 rows, feature stats,
episode padding, 12,000 random reads with two workers, and official pre/post
processors. The adapter also has CPU integration tests for the unified command,
policy source preparation without MP4, factory and resume contract. These checks
do not establish model forward/backward correctness or GPU training performance.
The active TRAIN23 configuration and runtime remain on the certified standard
backend; no live experiment is switched. Use this option for a new reviewed run,
then validate actual model training and the unchanged closed-loop acceptance
panel, preserving prior outputs.

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

`smolvla_evaluation.fresh_stop_confirmation` opts into independent model stop
confirmation (default false). A queued stop candidate schedules a new prediction
on the next observation and does not count as a vote. While confirming, only the
remaining action queue is invalidated; observation history is preserved. Three
fresh predictions must continue to request stopping with the existing visual
evidence rule. A fresh movement action cancels the candidate. LiDAR emergency
stops remain immediate and do not depend on these model votes. Traces record
prediction generation, freshness and forced replans for verification. This is
an execution experiment, not a change to model weights or acceptance thresholds.

For new S3 collection, `collection.s3_room_assignment: balanced-permutations`
cycles all six assignments of green cylinder, yellow box and red cube to the
north, south and end rooms using absolute layout ID modulo six. Each layout
still emits all three target instructions with identical geometry and start.
This option requires `scene: s3` and `scene_episode: sampled`; canonical sampling
and existing declared evaluation layouts retain their historical mapping.
Room ownership is saved as privileged provenance in JSON only, never as a
policy input or array training feature. Review accepted trajectories and room
balance after physical collection; geometric reachability alone is insufficient.

`collection.s3_search_teacher: observe-then-route` is an opt-in expert schedule
for S3 training collection. Until three consecutive onboard RGB observations
contain at least five task-colour pixels, it visits room centers in a fixed
geometry-only order and scans using the front/rear cameras. After discovery it
keeps that evidence through occlusion and uses the existing privileged expert
planner to approach the target. This prevents hidden object assignments from
choosing the pre-discovery route. It is not a learner execution gate or new
policy input. It requires actual RGB capture and rejects metadata-only collection.
The legacy `privileged-target` mode remains for historical physical baselines;
do not treat its hidden-target route labels as observation-grounded search data.
Room tours need a separately reviewed collection budget. CPU schedule/interface
tests do not establish real physical success, camera coverage, or learner memory.

The observation teacher also accepts `collection.s3_discovery_handoff:
interior-center`. If RGB discovery occurs while inside a room, it finishes
entering that room before changing to the target route; corridor discovery
keeps the direct handoff. Room geometry alone selects this temporary goal,
and the original task arrival/stop and quality rules still apply. This opt-in
maneuver addresses door-frame drift observed during a long in-place turn;
physical collection must verify it. The default remains `direct`.
Separate teacher traces and the three discovery RGB frames are diagnostic
sidecars, including for rejected episodes, and never become policy features.

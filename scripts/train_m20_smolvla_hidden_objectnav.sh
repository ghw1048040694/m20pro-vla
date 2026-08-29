#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_smolvla_env.sh"

DATA_ROOT="${M20PRO_VLA_DATA_ROOT}"
DATASET_ROOT="${M20PRO_SMOLVLA_DATASET_ROOT:-${DATA_ROOT}/datasets/m20_hidden_objectnav_lerobot_v1_camera12_stop08}"
DATASET_REPO_ID="${M20PRO_SMOLVLA_DATASET_REPO_ID:-m20pro_hidden_objectnav_v1_camera12_stop08}"
CHECKPOINT="${M20PRO_SMOLVLA_INIT_CHECKPOINT:-${M20PRO_SMOLVLA_CHECKPOINT}}"
OUTPUT_DIR="${DATA_ROOT}/checkpoints/smolvla_hidden_objectnav_v1_camera12_stop08_2000"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_DIR="$1"
  shift
fi
STEPS="${M20PRO_SMOLVLA_STEPS:-2000}"
BATCH_SIZE="${M20PRO_SMOLVLA_BATCH_SIZE:-2}"
SAVE_FREQ="${M20PRO_SMOLVLA_SAVE_FREQ:-500}"
WARMUP_STEPS="${M20PRO_SMOLVLA_WARMUP_STEPS:-100}"
DECAY_STEPS="${M20PRO_SMOLVLA_DECAY_STEPS:-${STEPS}}"
for value_name in STEPS BATCH_SIZE SAVE_FREQ WARMUP_STEPS DECAY_STEPS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer; got ${value}" >&2
    exit 2
  fi
done
if (( WARMUP_STEPS <= 0 || WARMUP_STEPS >= DECAY_STEPS )); then
  echo "Require 0 < warmup steps < decay steps; got ${WARMUP_STEPS}/${DECAY_STEPS}" >&2
  exit 2
fi
if (( DECAY_STEPS != STEPS )); then
  echo "Decay steps must equal training steps; got ${DECAY_STEPS}/${STEPS}" >&2
  exit 2
fi
if (( SAVE_FREQ > STEPS )); then
  echo "Save frequency must not exceed training steps; got ${SAVE_FREQ}/${STEPS}" >&2
  exit 2
fi

MANIFEST="${DATASET_ROOT}/m20pro_conversion_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Canonical dataset manifest not found: ${MANIFEST}" >&2
  exit 3
fi
"${M20PRO_SMOLVLA_PYTHON}" - "${MANIFEST}" "${DATASET_REPO_ID}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_repo_id = sys.argv[2]
policy = manifest.get("stop_label_policy", {})
if manifest.get("repo_id") != expected_repo_id:
    raise SystemExit(
        f"Dataset repo id mismatch: {manifest.get('repo_id')} != {expected_repo_id}"
    )
if manifest.get("schema") != "m20pro_hidden_objectnav_lerobot_v1":
    raise SystemExit(f"Unsupported hidden dataset schema: {manifest.get('schema')}")
if manifest.get("task_type_filter") != "hidden_object_search":
    raise SystemExit(f"Unexpected task type filter: {manifest.get('task_type_filter')}")
if manifest.get("candidate_split") != "hidden_train":
    raise SystemExit(f"Unexpected candidate split: {manifest.get('candidate_split')}")
if policy.get("version") != "target_radius_latched_v1":
    raise SystemExit(f"Unsupported stop label policy: {policy}")
if abs(float(policy.get("success_radius_m", -1.0)) - 0.8) > 1.0e-6:
    raise SystemExit(f"Stop label radius is not 0.8 m: {policy}")
audit = manifest.get("stop_label_audit", {})
if audit.get("stop_motion_conflict_frames") != 0:
    raise SystemExit(f"Stop/motion conflicts remain in dataset: {audit}")
if audit.get("trimmed_before_target_reach_episodes") != 0:
    raise SystemExit(f"Dataset trims episodes before target reach: {audit}")
if audit.get("retained_invisible_stop_frames") != 0:
    raise SystemExit(f"Dataset contains visually unobservable stop labels: {audit}")
if audit.get("canonical_stop_visibility_valid") is not True:
    raise SystemExit(f"Canonical stop visibility gate is missing or failed: {audit}")
sensor = manifest.get("sensor_config", {})
if abs(float(sensor.get("camera_focal_length_mm", -1.0)) - 12.0) > 1.0e-6:
    raise SystemExit(f"Dataset camera focal length is not 12 mm: {sensor}")
if int(manifest.get("episode_count", 0)) < 29:
    raise SystemExit(f"Dataset must contain at least 29 audited episodes: {manifest.get('episode_count')}")
coverage = manifest.get("coverage", {})
required_coverage = {
    "unique_scenes": 8,
    "unique_object_categories": 12,
    "unique_instruction_templates": 24,
}
for field, minimum in required_coverage.items():
    if int(coverage.get(field, 0)) < minimum:
        raise SystemExit(
            f"Dataset coverage {field}={coverage.get(field)} is below {minimum}: {coverage}"
        )
scenario_ids = coverage.get("scenario_episode_ids", [])
episode_count = int(manifest.get("episode_count", 0))
if len(scenario_ids) != episode_count or len(set(scenario_ids)) != episode_count:
    raise SystemExit(
        f"Dataset scenario ids are missing or duplicated: {len(scenario_ids)}/{episode_count}"
    )
if not all(str(scenario_id).startswith("hidden_train_") for scenario_id in scenario_ids):
    raise SystemExit(f"Hidden dataset includes non-hidden training ids: {scenario_ids[:8]}")
PY
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Training output already exists; choose a new directory: ${OUTPUT_DIR}" >&2
  exit 4
fi

printf '[M20PRO-TRAIN] dataset=%s repo_id=%s\n' "${DATASET_ROOT}" "${DATASET_REPO_ID}"
printf '[M20PRO-TRAIN] init=%s output=%s\n' "${CHECKPOINT}" "${OUTPUT_DIR}"
printf '[M20PRO-TRAIN] steps=%s batch=%s warmup=%s decay=%s save_freq=%s\n' \
  "${STEPS}" "${BATCH_SIZE}" "${WARMUP_STEPS}" "${DECAY_STEPS}" "${SAVE_FREQ}"

exec lerobot-train \
  "$@" \
  --policy.path="${CHECKPOINT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers=0 \
  --log_freq=10 \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${DECAY_STEPS}" \
  --policy.push_to_hub=false \
  --policy.repo_id=m20pro_hidden_objectnav \
  --wandb.enable=false

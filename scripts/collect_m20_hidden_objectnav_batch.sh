#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA}"
DATASET_VERSION="${M20PRO_OBJECTNAV_DATASET_VERSION:-v1_hidden_search_source}"
LOG_DIR="${M20PRO_BATCH_LOG_DIR:-${DATA_ROOT}/logs/m20_hidden_objectnav_${DATASET_VERSION}_batch}"
LOCK_DIR="${DATA_ROOT}/locks"
LOCK_PATH="${LOCK_DIR}/m20_hidden_objectnav_${DATASET_VERSION}.lock"
AUDIT_REPORT="${DATA_ROOT}/logs/m20_smolvla_data_audit_${DATASET_VERSION}.json"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 hidden_train_0001 hidden_train_0002 ..." >&2
  echo "Set M20PRO_OVERWRITE=1 only when replacing an existing episode." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[M20PRO-BATCH] another ${DATASET_VERSION} collection is already running: ${LOCK_PATH}" >&2
  exit 4
fi
failed=()

validate_episode_audit() {
  python3 - "${AUDIT_REPORT}" "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
episode_id = sys.argv[2]
report = json.loads(report_path.read_text(encoding="utf-8"))
matches = [
    episode
    for episode in report.get("episodes", [])
    if episode.get("scenario_episode_id") == episode_id
]
if len(matches) != 1:
    raise SystemExit(
        f"Expected one audited {episode_id} episode, found {len(matches)}"
    )
episode = matches[0]
problems = []
if episode.get("errors"):
    problems.append(f"errors={episode['errors']}")
if episode.get("success") is not True:
    problems.append("success=false")
if episode.get("smolvla_eligible") is not True:
    problems.append("smolvla_eligible=false")
if episode.get("task_type") != "hidden_object_search":
    problems.append(f"task_type={episode.get('task_type')}")
if episode.get("search_required") is not True:
    problems.append("search_required=false")
if episode.get("target_visible_at_start") is not False:
    problems.append(f"target_visible_at_start={episode.get('target_visible_at_start')}")
for field, expected in (
    ("success_radius_m", 0.8),
    ("success_final_tolerance_m", 0.02),
    ("camera_focal_length_mm", 12.0),
):
    try:
        value = float(episode.get(field))
    except (TypeError, ValueError):
        problems.append(f"{field}=missing")
        continue
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-6):
        problems.append(f"{field}={value} (expected {expected})")
if problems:
    raise SystemExit(f"{episode_id} audit failed: " + "; ".join(problems))
print(
    f"[M20PRO-BATCH] episode audit passed: {episode_id} "
    f"sha256={episode.get('sha256')}"
)
PY
}

validate_final_audit() {
  python3 - "${AUDIT_REPORT}" "$@" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

report_path = Path(sys.argv[1])
expected_ids = sys.argv[2:]
report = json.loads(report_path.read_text(encoding="utf-8"))
if len(set(expected_ids)) != len(expected_ids):
    duplicates = sorted(
        episode_id
        for episode_id, count in Counter(expected_ids).items()
        if count > 1
    )
    raise SystemExit(f"Duplicate requested episode ids: {duplicates}")
episodes = report.get("episodes", [])
audited_ids = [episode.get("scenario_episode_id") for episode in episodes]
missing = sorted(set(expected_ids) - set(audited_ids))
if missing:
    raise SystemExit(f"Requested episodes missing from final audit: {missing}")
eligible_ids = {
    episode.get("scenario_episode_id")
    for episode in episodes
    if episode.get("smolvla_candidate")
    and episode.get("split") == "hidden_train"
    and episode.get("smolvla_eligible") is True
}
if eligible_ids != set(expected_ids):
    raise SystemExit(
        "Final eligible episode set differs from the requested set: "
        f"missing={sorted(set(expected_ids) - eligible_ids)}, "
        f"unexpected={sorted(eligible_ids - set(expected_ids))}"
    )
bad = sorted(
    episode.get("scenario_episode_id")
    for episode in episodes
    if episode.get("scenario_episode_id") in expected_ids
    and (
        episode.get("errors")
        or episode.get("success") is not True
        or episode.get("smolvla_eligible") is not True
    )
)
if bad:
    raise SystemExit(f"Requested episodes failed final eligibility: {bad}")
required_gates = (
    "minimum_train_scenes_8",
    "minimum_object_categories_12",
    "minimum_instruction_templates_24",
    "minimum_hidden_search_families_4",
    "all_hidden_candidates_valid_and_timestamp_aligned",
    "search_required_and_occluded_at_start",
    "scene_geometry_visible_to_lidar",
    "six_dimensional_action_labels_present",
    "hidden_object_search_success_present",
)
gates = report.get("hidden_object_search_gates", {})
failed_gates = [name for name in required_gates if gates.get(name) is not True]
if report.get("ready_for_hidden_object_search_finetune") is not True or failed_gates:
    raise SystemExit(f"Final hidden ObjectNav gates failed: {failed_gates}")
inventory = report.get("inventory", {})
print(
    "[M20PRO-BATCH] final audit passed: "
    f"eligible={inventory.get('smolvla_eligible_episodes')} "
    f"scenes={inventory.get('annotated_scenes')} "
    f"objects={len(inventory.get('target_categories', []))} "
    f"templates={len(inventory.get('instruction_template_ids', []))}"
)
PY
}

for episode_id in "$@"; do
  log_path="${LOG_DIR}/${episode_id}.log"
  echo "[M20PRO-BATCH] collecting ${episode_id}; log=${log_path}"
  if "${SCRIPT_DIR}/collect_m20_hidden_objectnav.sh" "${episode_id}" 2>&1 | tee "${log_path}"; then
    echo "[M20PRO-BATCH] collection passed: ${episode_id}"
  else
    echo "[M20PRO-BATCH] collection failed: ${episode_id}" >&2
    failed+=("${episode_id}")
    continue
  fi

  audit_path="${LOG_DIR}/${episode_id}.audit.json"
  if "${SCRIPT_DIR}/audit_m20_smolvla_data.sh" \
      --input-root "${DATA_ROOT}/datasets/m20_hidden_objectnav_${DATASET_VERSION}" \
      --output "${AUDIT_REPORT}" \
      >"${audit_path}" 2>&1; then
    echo "[M20PRO-BATCH] audit completed after ${episode_id}; report=${audit_path}"
    rg \
      -e '"annotated_scenes"' \
      -e '"smolvla_candidate_episodes"' \
      -e '"smolvla_eligible_episodes"' \
      -e '"hidden_search_family_ids"' \
      -e '"ready_for_hidden_object_search_finetune"' \
      -e '"ready_for_smolvla_finetune"' \
      "${audit_path}" || true
    if ! validate_episode_audit "${episode_id}" 2>&1 | tee -a "${log_path}"; then
      echo "[M20PRO-BATCH] episode audit failed: ${episode_id}" >&2
      failed+=("${episode_id}:eligibility")
    fi
  else
    cat "${audit_path}" >>"${log_path}"
    echo "[M20PRO-BATCH] audit command failed after ${episode_id}; report=${audit_path}" >&2
    failed+=("${episode_id}:audit")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  printf '[M20PRO-BATCH] failed=%s\n' "${failed[*]}" >&2
  exit 1
fi

if [[ $# -ge 24 ]]; then
  if ! validate_final_audit "$@"; then
    echo "[M20PRO-BATCH] final dataset audit failed" >&2
    exit 1
  fi
fi

echo "[M20PRO-BATCH] completed=$#"

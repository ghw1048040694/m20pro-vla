#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA}"
INPUT_ROOT="${M20PRO_OBJECTNAV_INPUT_ROOT:-${DATA_ROOT}/datasets/m20_hidden_objectnav_v1_hidden_search_source/train}"
OUTPUT_ROOT="${M20PRO_OBJECTNAV_OUTPUT_ROOT:-${DATA_ROOT}/datasets/m20_hidden_objectnav_lerobot_v1_camera12_stop08}"
AUDIT="${M20PRO_OBJECTNAV_AUDIT:-${DATA_ROOT}/logs/m20_smolvla_data_audit_v1_hidden_search_source.json}"
REPO_ID="${M20PRO_OBJECTNAV_REPO_ID:-m20pro_hidden_objectnav_v1_camera12_stop08}"
LOCK_PATH="${M20PRO_OBJECTNAV_LOCK:-${DATA_ROOT}/locks/m20_hidden_objectnav_v1_hidden_search.lock}"

exec "${SCRIPT_DIR}/convert_m20_objectnav_to_lerobot.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --audit "${AUDIT}" \
  --repo-id "${REPO_ID}" \
  --task-type-filter hidden_object_search \
  --candidate-split hidden_train \
  --require-audit-ready \
  --no-require-exact-scenario-set \
  --collection-lock "${LOCK_PATH}" \
  "$@"

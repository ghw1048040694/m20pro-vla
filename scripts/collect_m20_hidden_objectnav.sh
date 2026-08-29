#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA}"
MANIFEST="${M20PRO_OBJECTNAV_MANIFEST:-${WORKSPACE}/configs/m20_hidden_objectnav_scenarios_v1.json}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SCENARIO_EPISODE_ID [recorder options...]" >&2
  exit 2
fi

EPISODE_ID="$1"
shift
case "${EPISODE_ID}" in
  hidden_train_*) SPLIT="hidden_train" ;;
  hidden_validation_*) SPLIT="hidden_validation" ;;
  hidden_test_*) SPLIT="hidden_test" ;;
  *)
    echo "Unsupported hidden ObjectNav episode id: ${EPISODE_ID}" >&2
    exit 2
    ;;
esac

DATASET_VERSION="${M20PRO_OBJECTNAV_DATASET_VERSION:-v1_hidden_search_source}"
OUTPUT_DIR="${DATA_ROOT}/datasets/m20_hidden_objectnav_${DATASET_VERSION}/${SPLIT}"
VIDEO_DIR="${DATA_ROOT}/videos/m20_hidden_objectnav_${DATASET_VERSION}/${SPLIT}"
if [[ -e "${OUTPUT_DIR}/episode_${EPISODE_ID}.h5" && "${M20PRO_OVERWRITE:-0}" != "1" ]]; then
  echo "Episode already exists; set M20PRO_OVERWRITE=1 to replace it: ${EPISODE_ID}" >&2
  exit 3
fi

exec "${SCRIPT_DIR}/record_public_m20_expert.sh" \
  --indoor-manifest "${MANIFEST}" \
  --scenario-episode-id "${EPISODE_ID}" \
  --episodes 1 \
  --steps 320 \
  --warmup-steps 75 \
  --nav-forward-speed 0.45 \
  --nav-wheel-acceleration 12.0 \
  --success-radius 0.8 \
  --success-final-tolerance 0.02 \
  --camera-focal-length 12.0 \
  --stop-yaw-brake-gain 0.0 \
  --stop-pretrigger-radius 0.0 \
  --stop-speed-threshold 0.15 \
  --stop-confirm-steps 5 \
  --target-hold-steps 100 \
  --output-dir "${OUTPUT_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  "$@"

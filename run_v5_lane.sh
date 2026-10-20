#!/usr/bin/env bash
# v5 (S2 room) 多档 checkpoint 同分布评测 —— 单泳道驱动
# 用法: bash run_v5_lane.sh <泳道名> <ckpt步数1> <ckpt步数2> ...
# 每档 13 个评测 episode，独立输出目录 .runtime/eval/m20_s2_room_v5_ft1_s<step>/
# 注意: play_m20_smolvla.py 依赖 LeRobot，必须用 lerobot 环境跑。
set -u
cd ~/VLA-Learning/projects/m20pro-ros2-sim || exit 1

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PY=/home/ubuntu/miniconda3/envs/lerobot/bin/python

LANE="$1"; shift

CKPT_ROOT=.runtime/checkpoints/smolvla_m20_s2_room_v5_ft1_steps13500/checkpoints
EPISODES_DIR=.runtime/datasets/m20_s2_room_v5
IDS=6000,6006,6012,6022,6026,6032,6037,6042,6048,6054,6057,6063,6066

echo "###### LANE ${LANE} START  ($(date +%H:%M:%S))  ckpts: $*"

for S in "$@"; do
  OUT=.runtime/eval/m20_s2_room_v5_ft1_s${S}
  CKPT=${CKPT_ROOT}/${S}/pretrained_model
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] LANE ${LANE} START checkpoint ${S} -> ${OUT}"
  echo "=================================================================="
  if [ ! -f "${CKPT}/config.json" ]; then
    echo "[$(date +%H:%M:%S)] LANE ${LANE} MISSING checkpoint ${CKPT}; skip"
    continue
  fi
  "${PY}" scripts/mujoco/play_m20_smolvla.py \
    --checkpoint "${CKPT}" \
    --episodes-dir "${EPISODES_DIR}" \
    --output-dir "${OUT}" \
    --summary "${OUT}/fleet_summary.json" \
    --episode-ids "${IDS}" \
    --policy-steps 2600 \
    --device cuda \
    --seed 20260918 \
    --no-write-videos \
    --motion-smoothing 0.2 \
    --max-forward-delta 0.02 \
    --max-yaw-delta 0.015 \
    --success-radius 0.95 \
    --safety-stop-distance 0.5 \
    --visual-stop-min-pixels 80 \
    --post-stop-hold-steps 25
  echo "[$(date +%H:%M:%S)] LANE ${LANE} DONE checkpoint ${S} exit=$?"
done

echo "###### LANE ${LANE} DONE  ($(date +%H:%M:%S))"

#!/usr/bin/env bash

_M20PRO_VLA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M20PRO_SIM_WS="$(cd "${_M20PRO_VLA_SCRIPT_DIR}/.." && pwd)"
export M20PRO_VLA_ENV="${M20PRO_VLA_ENV:-m20pro-vla}"
export M20PRO_VLA_DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-${M20PRO_SIM_WS}/.runtime}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${M20PRO_VLA_ENV}"

# MuJoCo is the primary WSL simulator. Windows Isaac Sim is reserved for
# later sim2sim validation and is never imported by this Linux environment.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# All local datasets, checkpoints, videos and logs share one ignored runtime root.
export M20PRO_OUTPUT_ROOT="${M20PRO_VLA_DATA_ROOT}"

unset _M20PRO_VLA_SCRIPT_DIR
echo "Activated ${M20PRO_VLA_ENV} (MuJoCo primary; Windows Isaac Sim for sim2sim only)"

#!/usr/bin/env bash

_M20PRO_SMOLVLA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_M20PRO_SMOLVLA_SCRIPT_DIR}/activate_vla_env.sh"
export M20PRO_SMOLVLA_ENV_PREFIX="${M20PRO_SMOLVLA_ENV_PREFIX:-${M20PRO_VLA_DATA_ROOT}/envs/m20pro-smolvla}"
export M20PRO_ISAAC_ENV_PREFIX="${M20PRO_ISAAC_ENV_PREFIX:-${CONDA_PREFIX}}"

_M20PRO_SMOLVLA_SITE="${M20PRO_SMOLVLA_ENV_PREFIX}/lib/python3.11/site-packages"
_M20PRO_ISAAC_SITE="${M20PRO_ISAAC_ENV_PREFIX}/lib/python3.11/site-packages"

if [[ ! -x "${M20PRO_SMOLVLA_ENV_PREFIX}/bin/python" ]]; then
  echo "SmolVLA environment not found: ${M20PRO_SMOLVLA_ENV_PREFIX}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${_M20PRO_ISAAC_SITE}/torch" ]]; then
  echo "Validated PyTorch base environment not found: ${M20PRO_ISAAC_ENV_PREFIX}" >&2
  return 1 2>/dev/null || exit 1
fi

# LeRobot lives in the small overlay environment. The already validated
# PyTorch/CUDA stack is reused read-only from the Isaac environment.
export PATH="${M20PRO_SMOLVLA_ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${_M20PRO_SMOLVLA_SITE}:${_M20PRO_ISAAC_SITE}"
export M20PRO_SMOLVLA_PYTHON="${M20PRO_SMOLVLA_ENV_PREFIX}/bin/python"

export HF_HOME="${M20PRO_VLA_DATA_ROOT}/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export M20PRO_SMOLVLA_CHECKPOINT="${M20PRO_SMOLVLA_CHECKPOINT:-${M20PRO_VLA_DATA_ROOT}/models/lerobot_smolvla_base_c83c316}"
export M20PRO_SMOLVLM_PROCESSOR="${M20PRO_SMOLVLM_PROCESSOR:-${M20PRO_VLA_DATA_ROOT}/models/smolvlm2_500m_processor_7b375e1}"

unset _M20PRO_SMOLVLA_SCRIPT_DIR _M20PRO_SMOLVLA_SITE _M20PRO_ISAAC_SITE
echo "Activated SmolVLA overlay (LeRobot 0.4.4, PyTorch base: ${M20PRO_ISAAC_ENV_PREFIX})"

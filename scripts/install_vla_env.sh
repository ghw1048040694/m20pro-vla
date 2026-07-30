#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${M20PRO_VLA_ENV:-m20pro-vla}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda was not found at ${CONDA_EXE}. Set CONDA_EXE explicitly." >&2
  exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_EXE}" create -y -n "${ENV_NAME}" python=3.11 --override-channels -c conda-forge
fi

ENV_PYTHON="${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  ENV_PYTHON="$("${CONDA_EXE}" run -n "${ENV_NAME}" command -v python)"
fi

"${ENV_PYTHON}" -m pip install --upgrade pip "setuptools==80.9.0" "wheel==0.45.1"
"${ENV_PYTHON}" -m pip install "mujoco==3.3.7" "PyYAML==6.0.2"

"${ENV_PYTHON}" - <<'PY'
import mujoco
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is required for the VLA stage but is unavailable.")

model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print(f"MuJoCo {mujoco.__version__} and PyTorch {torch.__version__} are ready.")
PY

echo "Environment ${ENV_NAME} is ready for the MuJoCo-first M20 VLA workflow."
echo "Windows Isaac Sim remains a separate sim2sim validation runtime."
echo "Activate it with: source ${WORKSPACE}/scripts/activate_vla_env.sh"

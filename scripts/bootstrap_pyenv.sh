#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${ALPHAFORGE_PYTHON_VERSION:-3.11.8}"
VENV_PATH="${ALPHAFORGE_VENV_PATH:-.venv311}"

command -v pyenv >/dev/null
pyenv shell "${PYTHON_VERSION}"
python -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -r requirements-server.txt

"${VENV_PATH}/bin/python" - <<'PY'
import platform
import qlib
import torch

print("python_arch:", platform.machine())
print("qlib:", qlib.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

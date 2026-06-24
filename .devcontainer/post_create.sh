#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/conda/envs/hovsg/bin:/opt/conda/bin:${PATH}"

python -m pip install -e .
mkdir -p data checkpoints

python - <<'PY'
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: sm_{capability[0]}{capability[1]}")
    print(f"PyTorch arch list: {arch_list}")
    print(f"Blackwell sm_120 supported: {'sm_120' in arch_list}")
PY

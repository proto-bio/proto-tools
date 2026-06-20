#!/bin/bash
set -euo pipefail

echo "Setting up DeepPBS specificity standalone environment..."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv package manager..."
  pip install uv
fi

echo "Installing PyTorch stack for DeepPBS..."
uv pip install "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0" --torch-backend=auto

echo "Installing PyG dependencies..."
# Derive the PyG wheel tag from THIS env's torch CUDA build (e.g. 12.1 -> cu121).
# PYTHONNOUSERSITE=1 prevents a stray user-site torch (e.g. a different CUDA build in
# ~/.local) from shadowing the venv torch and mis-selecting a CPU/mismatched wheel,
# which yields ABI-broken torch_cluster (_version_cuda.so: undefined symbol) at runtime.
PYG_TAG="$(PYTHONNOUSERSITE=1 "$PYTHON_EXE" - <<'PY'
import torch
cuda = torch.version.cuda
print("cu" + cuda.replace(".", "") if cuda else "cpu")
PY
)"
PYG_URL="https://data.pyg.org/whl/torch-2.3.0+${PYG_TAG}.html"
PYTHONNOUSERSITE=1 uv pip install torch_scatter torch_sparse torch_cluster torch_geometric -f "${PYG_URL}"

echo "Installing DeepPBS Python dependencies from requirements.txt..."
uv pip install -r requirements.txt
uv pip install pdb2pqr

DEEPPBS_REPO_PATH="${DEEPPBS_REPO_PATH:-/large_storage/hielab/userspace/adititm/DeepPBS}"
if [[ -d "${DEEPPBS_REPO_PATH}" ]]; then
  echo "Installing deeppbs package from ${DEEPPBS_REPO_PATH}..."
  uv pip install -e "${DEEPPBS_REPO_PATH}"
fi

echo "DeepPBS specificity setup complete."

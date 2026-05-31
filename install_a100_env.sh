#!/usr/bin/env bash
set -euo pipefail

# One-command installer for the SGCN A100 environment.
# Defaults follow the SGCN_new README: Python 3.10, PyTorch 2.4.0+cu118,
# PyG 2.7.0, and OGB for ogbn-proteins.

ENV_NAME="${ENV_NAME:-sgcn-a100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.0}"
CUDA_TAG="${CUDA_TAG:-cu118}"
PYG_VERSION="${PYG_VERSION:-2.7.0}"

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "==> SGCN A100 environment installer"
echo "    env:      ${ENV_NAME}"
echo "    python:   ${PYTHON_VERSION}"
echo "    torch:    ${TORCH_VERSION}+${CUDA_TAG}"
echo "    pyg:      ${PYG_VERSION}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Detected NVIDIA GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo
else
  echo "WARN: nvidia-smi was not found. The install can continue, but GPU validation may fail."
  echo
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if command -v mamba >/dev/null 2>&1; then
    CREATE_CMD="mamba"
  else
    CREATE_CMD="conda"
  fi
elif command -v mamba >/dev/null 2>&1; then
  echo "ERROR: mamba was found but conda was not. This installer needs conda activation support." >&2
  echo "Install Miniconda/Mambaforge first, then rerun this script." >&2
  exit 1
else
  echo "ERROR: conda is required. Install Miniconda/Mambaforge first, then rerun this script." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda environment '${ENV_NAME}' already exists"
else
  echo "==> Creating conda environment '${ENV_NAME}'"
  "${CREATE_CMD}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "==> Activating '${ENV_NAME}'"
conda activate "${ENV_NAME}"

echo "==> Upgrading Python packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch CUDA wheels"
python -m pip install \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}"

echo "==> Installing PyG compiled extensions"
python -m pip install \
  pyg-lib \
  torch-scatter \
  torch-sparse \
  torch-cluster \
  torch-spline-conv \
  -f "${PYG_WHEEL_URL}"

echo "==> Installing project Python dependencies"
python -m pip install \
  "torch-geometric==${PYG_VERSION}" \
  "ogb==1.3.6" \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "pandas==2.2.3" \
  "scikit-learn==1.5.2" \
  "networkx==3.3" \
  "tqdm==4.66.5" \
  "matplotlib==3.9.2" \
  "PyYAML==6.0.2" \
  "ipykernel==6.29.5" \
  "jupyterlab==4.2.5"

echo "==> Registering Jupyter kernel"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "==> Running import and CUDA smoke test"
python - <<'PY'
import torch
import torch_geometric
import ogb
from torch_geometric.nn import GCNConv, SAGEConv

print(f"torch={torch.__version__}")
print(f"torch_geometric={torch_geometric.__version__}")
print(f"ogb={ogb.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    x = torch.randn(8, 16, device=device)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
        device=device,
    )
    conv = GCNConv(16, 8, add_self_loops=False).to(device)
    y = conv(x, edge_index)
    torch.cuda.synchronize()
    print(f"pyg_cuda_smoke_shape={tuple(y.shape)}")
else:
    raise SystemExit("CUDA is not available inside this environment.")
PY

cat <<EOF

==> Done.
Activate the environment with:
  conda activate ${ENV_NAME}

Useful overrides:
  ENV_NAME=my-env bash install_a100_env.sh
  CUDA_TAG=cu121 TORCH_VERSION=2.4.0 bash install_a100_env.sh

EOF

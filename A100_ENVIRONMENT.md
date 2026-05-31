# A100 environment setup

This repository targets the SGCN/OGBN-Proteins code in `src/` and follows the
environment notes from `SGCN_new`:

- Python 3.10
- PyTorch 2.4.0 CUDA 11.8 wheels
- PyG 2.7.0 plus PyG compiled extensions
- OGB, NumPy, SciPy, pandas, scikit-learn, NetworkX, tqdm, matplotlib, PyYAML

CUDA 11.8 PyTorch wheels run well on A100 40GB nodes as long as the NVIDIA
driver is new enough for CUDA 11.8 runtime wheels.

## One-command install

From the repository root:

```bash
bash install_a100_env.sh
```

The script creates a conda environment named `sgcn-a100`, installs the CUDA
PyTorch/PyG stack, registers a Jupyter kernel, and runs a CUDA smoke test.

To use another environment name:

```bash
ENV_NAME=sgcn-a100-40g bash install_a100_env.sh
```

After installation:

```bash
conda activate sgcn-a100
```

## Dataset note

`src/utils.py` currently loads OGBN-Proteins from:

```python
PygNodePropPredDataset(name=dataset_name, root='/mnt/SGCN/dataset')
```

On a new A100 server, make sure `/mnt/SGCN/dataset` exists or adjust that path
before training. OGB will download missing dataset files on first use if the
path is writable.

## Version overrides

The installer is controlled by environment variables:

```bash
ENV_NAME=sgcn-a100 \
PYTHON_VERSION=3.10 \
TORCH_VERSION=2.4.0 \
CUDA_TAG=cu118 \
PYG_VERSION=2.7.0 \
bash install_a100_env.sh
```

The default combination is the recommended one for migrating from the previous
P100-16G setup while keeping the project code and PyG wheels aligned.

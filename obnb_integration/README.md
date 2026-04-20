# SGCN → obnbench Integration

This directory contains all the files needed to port the SGCN (Subgraph-based GCN)
training algorithm from [`Corayu0323/SGCN`](https://github.com/Corayu0323/SGCN)
into the [`krishnanlab/obnbench`](https://github.com/krishnanlab/obnbench) benchmarking
framework.

---

## What was changed

| File | Status | Description |
|------|--------|-------------|
| `obnbench/model_layers/mp_layers.py` | **Modified** | Added `SGCNConv` and `SGCNMPModule` |
| `obnbench/model.py` | **Modified** | Added `SGCNModelModule`, `_sample_subgraph_nodes`, patched `build_mp_module` |
| `conf/model/sgcn.yaml` | **New** | Hydra config exposing all SGCN hyperparameters |
| `main.py` | **Modified** | Routes to `SGCNModelModule` when `mp_type == 'SGCN'` |

The unchanged obnbench files (`data_module.py`, `preprocess.py`, `metrics.py`,
`schedulers.py`, etc.) require **no modifications**.

---

## How to apply

Copy the three files into the corresponding locations inside the obnbench repo:

```bash
# from the root of krishnanlab/obnbench
cp obnb_integration/obnbench/model_layers/mp_layers.py  obnbench/model_layers/mp_layers.py
cp obnb_integration/obnbench/model.py                   obnbench/model.py
cp obnb_integration/conf/model/sgcn.yaml                conf/model/sgcn.yaml
cp obnb_integration/main.py                             main.py
```

---

## Running SGCN

```bash
# inside krishnanlab/obnbench, after applying the files above
python main.py \
  +model=sgcn \
  dataset.network=BioGRID \
  dataset.label=DisGeNET \
  save_results=true
```

All SGCN-specific hyperparameters can be overridden on the command line:

```bash
python main.py +model=sgcn \
  dataset.network=BioGRID dataset.label=DisGeNET \
  model.sgcn.n_subgraphs=10 \
  model.sgcn.local_epochs=3 \
  model.sgcn.subsampling_method=random_walk \
  model.sgcn.subgraph_max_nodes=1024 \
  model.sgcn.truncation_ratio=0.3 \
  model.sgcn.aggregation_method=avg
```

---

## Architecture overview

```
obnbench (Lightning)              SGCN contribution
─────────────────────────────     ────────────────────────────────────
DataModule (full-batch)       →   same (no changes needed)
  │
  ▼
SGCNModelModule.training_step  ←  SGCN epoch loop (train.py)
  ├─ _sample_subgraph_nodes       4 strategies: random_node/edge/walk/snowball
  ├─ _make_sub_batch              slice rawfeat_* + edge_index + edge_weight
  ├─ _forward_subgraph            feature_encoder → SGCNMPModule → pred_head → sigmoid
  ├─ local L-step training        reset → grad steps → val score
  ├─ truncation                   drop worst (truncation_ratio) subgraphs
  └─ aggregation                  softmax / avg / weighted parameter averaging
  │
SGCNMPModule (mp_layers.py)    ←  GNN_PyG (models.py)
  ├─ SGCNConv × num_layers        GCNConv, no self-loops, edge_weight
  ├─ pre-BN residual skip         h = h_conv + h_last_pre_bn
  └─ optional JK sum              sum of all layer outputs
  │
ModelModule.validation_step    ←  unchanged (metrics, logging, EarlyStopping)
ModelModule.test_step
```

---

## Design decisions

### Lightning framework is preserved
SGCN's training algorithm is implemented inside `training_step` using
`automatic_optimization = False` (manual optimisation mode). All other Lightning
machinery — metrics, callbacks (EarlyStopping, ModelCheckpoint), LR scheduler,
wandb/CSV logging, test step — is inherited unchanged from `ModelModule`.

### Full-batch DataLoader is sufficient
obnbench uses a single full-graph batch per epoch (`batch_size=1`).
`SGCNModelModule` receives this full batch and samples subgraphs internally,
exactly as the original SGCN `train_epoch_sgcn` does.

### edge_attr → edge_weight mapping
The original SGCN uses multi-dimensional `edge_attr` (OGBN-Proteins has 8
edge features, aggregated to node features during preprocessing).  obnbench
uses scalar `edge_weight`.  `SGCNConv` inherits from `GCNConv` with
`_edge_usage = "edge_weight"`, so the edge weight is passed automatically when
`use_edge_feature = true`.

### Feature encoders on subgraphs
The feature encoder is called on each subgraph (its parameters are reset to
`epoch_init_state` and trained as part of the local steps).  This works for
all **positional** encoders (`OneHotLogDeg`, `SVD`, `LapEigMap`, `Node2vec`,
etc.) whose raw features are precomputed tensors that can be sliced by node
index.

> ⚠️ **Incompatible encoders**: `AdjEmbBag` and `Embedding` use fixed-size
> weight matrices indexed by global node IDs; they cannot be used with
> subgraph training.

### Gradient clipping
With `automatic_optimization = False`, Lightning does **not** apply
`gradient_clip_val` automatically.  `SGCNModelModule` calls
`self.clip_gradients(optimizer, ...)` inside each local training step.
`main.py` sets `gradient_clip_val=None` on the `Trainer` for SGCN to avoid
double-clipping.

---

## Hyperparameter reference

| Key (`model.sgcn.*`) | Type | Default | Description |
|----------------------|------|---------|-------------|
| `n_subgraphs` | int | 5 | Subgraphs per epoch (≤0 → auto from coverage) |
| `local_epochs` | int | 5 | Local gradient steps per subgraph |
| `subsampling_method` | str | `random_node` | Sampling strategy |
| `subgraph_max_nodes` | int | 2048 | Hard node budget per subgraph (0 → ratio) |
| `max_subgraph_edges` | int | 0 | Hard edge cap (0 → disabled) |
| `subgraph_ratio` | float | 0.5 | Fallback fraction when `subgraph_max_nodes≤0` |
| `truncation_ratio` | float | 0.2 | Fraction of worst subgraphs to discard |
| `aggregation_method` | str | `sgcn` | `sgcn` / `avg` / `weighted` |
| `model.jk` | bool | false | Jumping Knowledge sum aggregation |

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler
from torch_geometric.utils import subgraph as pyg_subgraph

from .utils import add_labels, gen_model


def _cuda_sync(device):
    """Synchronize CUDA if available to get accurate timing boundaries."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


# ── SGCN helpers ─────────────────────────────────────────────────────────────

# Fraction of n_sample used as BFS seed nodes (1/20 = 5%)
_SGCN_SEED_RATIO = 20
# Hard upper bound on BFS hops for random_walk sampling
_SGCN_RANDOM_WALK_MAX_HOPS = 10
# Minimum training nodes to inject when a sampled subgraph contains none
_SGCN_MIN_TRAIN_NODES = 32
# Number of validation nodes sampled for the per-subgraph quality score
_SGCN_VAL_SAMPLE_SIZE = 512

def _sample_subgraph_nodes(edge_index, n_nodes, train_idx_cpu, method, n_sample,
                           subgraph_max_nodes=None):
    """Return a 1-D sorted LongTensor of sampled node indices.

    Supported methods
    -----------------
    random_node  – uniformly sample *n_sample* nodes at random.
    random_edge  – sample random edges and collect their incident nodes.
    random_walk  – BFS expansion from random training-set seeds (no hop cap).
    snowball     – BFS expansion capped at 2 hops from random seeds.

    Parameters
    ----------
    subgraph_max_nodes : int or None
        When provided, overrides *n_sample* as the hard upper bound on the
        number of returned nodes.  Has priority over the ratio-based value.
    """
    # subgraph_max_nodes takes priority over the ratio-derived n_sample.
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    n_sample = min(n_sample, n_nodes)

    if method == 'random_node':
        perm = torch.randperm(n_nodes)[:n_sample]
        return perm.sort().values

    elif method == 'random_edge':
        n_edges   = edge_index.shape[1]
        edge_perm = torch.randperm(n_edges)[:min(n_sample * 2, n_edges)]
        nodes     = edge_index[:, edge_perm].flatten().unique()
        if len(nodes) < n_sample:
            extra = torch.randperm(n_nodes)[:n_sample - len(nodes)]
            nodes = torch.cat([nodes, extra]).unique()
        return nodes[:n_sample].sort().values

    elif method in ('random_walk', 'snowball'):
        n_seeds  = min(max(n_sample // _SGCN_SEED_RATIO, 1), len(train_idx_cpu))
        seed_perm = torch.randperm(len(train_idx_cpu))[:n_seeds]
        seeds     = train_idx_cpu[seed_perm]

        visited = torch.zeros(n_nodes, dtype=torch.bool)
        visited[seeds] = True

        row, col  = edge_index
        max_hops  = 2 if method == 'snowball' else _SGCN_RANDOM_WALK_MAX_HOPS

        for _ in range(max_hops):
            if int(visited.sum()) >= n_sample:
                break
            mask      = visited[row]
            new_nodes = col[mask]
            visited[new_nodes] = True

        visited_nodes = visited.nonzero(as_tuple=False).squeeze(1)

        if len(visited_nodes) > n_sample:
            perm = torch.randperm(len(visited_nodes))[:n_sample]
            return visited_nodes[perm].sort().values
        elif len(visited_nodes) < n_sample:
            unvisited   = (~visited).nonzero(as_tuple=False).squeeze(1)
            extra_perm  = torch.randperm(len(unvisited))[:n_sample - len(visited_nodes)]
            return torch.cat([visited_nodes, unvisited[extra_perm]]).sort().values

        return visited_nodes.sort().values

    else:
        raise ValueError(
            f"Unknown subsampling_method: {method!r}. "
            f"Choose from: 'random_node', 'random_edge', 'random_walk', 'snowball'."
        )


def train_epoch_sgcn(model, data, criterion, optimizer, device,
                     train_idx, val_idx,
                     n_subgraphs=5,
                     subsampling_method='random_node',
                     subgraph_max_nodes=256,
                     max_subgraph_edges=300000,
                     subgraph_ratio=0.5,
                     truncation_ratio=0.2,
                     aggregation_method='sgcn',
                     use_labels=False, n_classes=112,
                     debug_subgraph_stats=False):
    """SGCN training epoch with subgraph sampling and configurable aggregation.

    Algorithm
    ---------
    For each of *n_subgraphs* independent subgraphs:

    1. Sample a subgraph of at most *subgraph_max_nodes* nodes using the
       chosen *subsampling_method*.  *subgraph_ratio* is used as a fallback
       when *subgraph_max_nodes* is not set (i.e. <= 0).
    2. Enforce a hard edge-count limit of *max_subgraph_edges* by randomly
       dropping edges when the induced subgraph exceeds that threshold.
    3. Reset the model to the epoch-start parameters and run one gradient step
       on the subgraph's training nodes.
    4. Evaluate the resulting local model on a mini-batch of validation nodes
       (via a forward pass over the subgraph augmented with those val nodes).
    5. Record the local state dict and validation loss (as quality score).

    After all subgraphs are processed:

    * Discard the bottom *truncation_ratio* fraction by validation score
      (truncation mechanism – suppresses noise-dominated subgraphs).
    * Aggregate the remaining local states according to *aggregation_method*.
    * Load the aggregated state into the model and clear stale optimizer
      momentum.

    Parameters
    ----------
    n_subgraphs         : int   – number of independent subgraphs per epoch.
    subsampling_method  : str   – one of 'random_node', 'random_edge',
                                   'random_walk', 'snowball'.
    subgraph_max_nodes  : int   – hard upper bound on nodes per subgraph.
                                   Takes priority over *subgraph_ratio*.
                                   Set <= 0 to fall back to *subgraph_ratio*.
    max_subgraph_edges  : int   – hard upper bound on edges per subgraph.
                                   Excess edges are randomly dropped (with
                                   matching edge_attr rows when present).
                                   Set <= 0 to disable.
    subgraph_ratio      : float – fraction of graph nodes per subgraph; used
                                   only when *subgraph_max_nodes* <= 0.
    truncation_ratio    : float – fraction of worst-performing subgraphs
                                   to discard before aggregation.
    aggregation_method  : str   – aggregation strategy after truncation:
                                   'sgcn'     – softmax-weighted average over
                                                validation scores (default).
                                   'avg'      – uniform equal-weight average
                                                (SGCN-Avg).
                                   'weighted' – performance-based linear-
                                                normalized weighted average
                                                (SGCN-Weighted).
    debug_subgraph_stats : bool – when True, print per-subgraph shape and
                                   CUDA memory stats before each forward pass.
    """
    model.train()

    train_idx_cpu = train_idx.cpu()
    val_idx_cpu   = val_idx.cpu()
    n_nodes       = data.num_nodes
    # subgraph_max_nodes takes priority; fall back to ratio-based size.
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    else:
        n_sample = max(1, int(n_nodes * subgraph_ratio))
    edge_index_cpu = data.edge_index.cpu()
    train_idx_set  = set(train_idx_cpu.tolist())

    # Snapshot of model parameters at the start of this epoch.  Every local
    # subgraph model is initialised from this state so that aggregation is
    # well-defined.
    epoch_init_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    local_states  = []
    val_scores    = []
    loss_sum      = 0.0
    valid_batches = 0

    val_sample_size = min(_SGCN_VAL_SAMPLE_SIZE, len(val_idx_cpu))

    _cuda_sync(device)
    epoch_start   = time.time()
    sampling_time = 0.0

    for _sg in range(n_subgraphs):
        # ── 1. Sample subgraph node indices ─────────────────────────────────
        t_sample  = time.time()
        node_idx  = _sample_subgraph_nodes(
            edge_index_cpu, n_nodes, train_idx_cpu, subsampling_method, n_sample,
            subgraph_max_nodes=subgraph_max_nodes,
        )

        # Guarantee at least a few training nodes are included.
        if not torch.isin(node_idx, train_idx_cpu).any():
            extra    = train_idx_cpu[
                torch.randperm(len(train_idx_cpu))[:min(_SGCN_MIN_TRAIN_NODES, len(train_idx_cpu))]
            ]
            node_idx = torch.cat([node_idx, extra]).unique()
            node_idx = node_idx.sort().values

        # ── 2. Extract induced subgraph ──────────────────────────────────────
        edge_index_sub, edge_attr_sub = pyg_subgraph(
            node_idx,
            data.edge_index,
            data.edge_attr,
            relabel_nodes=True,
            num_nodes=n_nodes,
        )

        # ── 2a. Enforce hard edge-count cap ─────────────────────────────────
        if max_subgraph_edges is not None and max_subgraph_edges > 0:
            n_edges_sub = edge_index_sub.size(1)
            if n_edges_sub > max_subgraph_edges:
                perm          = torch.randperm(n_edges_sub)[:max_subgraph_edges]
                edge_index_sub = edge_index_sub[:, perm]
                if edge_attr_sub is not None:
                    edge_attr_sub = edge_attr_sub[perm]

        sampling_time += time.time() - t_sample

        x_sub  = data.x[node_idx].to(device)
        y_sub  = data.y[node_idx].to(device)
        ei_sub = edge_index_sub.to(device)
        ea_sub = edge_attr_sub.to(device) if edge_attr_sub is not None else None

        train_mask = torch.isin(node_idx, train_idx_cpu)
        if not train_mask.any():
            del x_sub, y_sub, ei_sub, ea_sub
            continue

        if use_labels:
            non_train_local = torch.where(~train_mask)[0]
            x_sub = add_labels(
                x_sub, data.train_labels_onehot, non_train_local, n_classes, device
            )

        # ── Debug: print subgraph stats before forward pass ─────────────────
        if debug_subgraph_stats:
            print(
                f'[SGCN debug] subgraph {_sg}: '
                f'num_sub_nodes={x_sub.shape[0]}, '
                f'num_sub_edges={ei_sub.shape[1]}, '
                f'x_sub.shape={tuple(x_sub.shape)}, '
                f'ei_sub.shape={tuple(ei_sub.shape)}'
                + (f', ea_sub.shape={tuple(ea_sub.shape)}' if ea_sub is not None else '')
                + (
                    f', cuda_allocated={torch.cuda.memory_allocated(device)}, '
                    f'cuda_reserved={torch.cuda.memory_reserved(device)}'
                    if device.type == 'cuda' else ''
                )
            )

        # ── 3. Reset to epoch-start state, do one gradient step ─────────────
        model.load_state_dict({k: v.to(device) for k, v in epoch_init_state.items()})
        model.train()

        train_mask_dev = train_mask.to(device)
        pred = model(x_sub, ei_sub, ea_sub)
        loss = criterion(pred[train_mask_dev], y_sub[train_mask_dev].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum      += loss.item()
        valid_batches += 1

        # Free training-step tensors before the validation forward pass.
        del pred, loss, train_mask_dev
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ── 4. Quick validation score ────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_sample  = val_idx_cpu[
                torch.randperm(len(val_idx_cpu))[:val_sample_size]
            ]
            # Augment subgraph with val nodes so GCN can aggregate their
            # neighborhood context without leaking their labels.
            eval_node_idx = torch.cat([node_idx, val_sample]).unique()
            eval_node_idx = eval_node_idx.sort().values

            ei_eval, ea_eval = pyg_subgraph(
                eval_node_idx,
                data.edge_index,
                data.edge_attr,
                relabel_nodes=True,
                num_nodes=n_nodes,
            )

            # Apply the same edge cap to the eval subgraph.
            if max_subgraph_edges is not None and max_subgraph_edges > 0:
                n_edges_eval = ei_eval.size(1)
                if n_edges_eval > max_subgraph_edges:
                    perm_eval = torch.randperm(n_edges_eval)[:max_subgraph_edges]
                    ei_eval   = ei_eval[:, perm_eval]
                    if ea_eval is not None:
                        ea_eval = ea_eval[perm_eval]

            x_eval  = data.x[eval_node_idx].to(device)
            y_eval  = data.y[eval_node_idx].to(device)
            ei_eval = ei_eval.to(device)
            ea_eval = ea_eval.to(device) if ea_eval is not None else None

            if use_labels:
                eval_train_mask = torch.isin(eval_node_idx, train_idx_cpu)
                non_train_eval  = torch.where(~eval_train_mask)[0]
                x_eval = add_labels(
                    x_eval, data.train_labels_onehot,
                    non_train_eval, n_classes, device
                )

            pred_eval     = model(x_eval, ei_eval, ea_eval)
            val_local_mask = torch.isin(eval_node_idx, val_sample).to(device)
            val_loss = criterion(
                pred_eval[val_local_mask], y_eval[val_local_mask].float()
            )
            val_score = -val_loss.item()   # higher is better

            del pred_eval, val_loss, val_local_mask
            del x_eval, y_eval, ei_eval, ea_eval

        # Save local state dict on CPU to avoid multi-copy GPU residency.
        local_states.append({k: v.clone().cpu() for k, v in model.state_dict().items()})
        val_scores.append(val_score)

        # Release per-subgraph GPU tensors.
        del x_sub, y_sub, ei_sub, ea_sub, train_mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Fallback if no valid subgraph was processed ──────────────────────────
    if not local_states:
        model.load_state_dict({k: v.to(device) for k, v in epoch_init_state.items()})
        _cuda_sync(device)
        return 0.0, time.time() - epoch_start, sampling_time

    # ── 5. Truncation: keep top (1 − truncation_ratio) subgraphs ────────────
    n_keep     = max(1, int(len(local_states) * (1.0 - truncation_ratio)))
    sorted_idx = sorted(range(len(val_scores)), key=lambda i: val_scores[i],
                        reverse=True)
    kept_idx   = sorted_idx[:n_keep]

    # ── 6. Aggregate local states according to aggregation_method ───────────
    kept_scores = torch.tensor([val_scores[i] for i in kept_idx], dtype=torch.float)

    if aggregation_method == 'avg':
        # SGCN-Avg: uniform equal-weight average
        weights = torch.ones(len(kept_idx), dtype=torch.float) / len(kept_idx)
    elif aggregation_method == 'weighted':
        # SGCN-Weighted: performance-based linear normalization.
        # Shift scores so the minimum becomes a small positive value, then
        # normalize so weights sum to 1.
        shifted = kept_scores - kept_scores.min() + 1e-8
        weights = shifted / shifted.sum()
    else:
        # Default 'sgcn': softmax-weighted average over validation scores
        if aggregation_method != 'sgcn':
            raise ValueError(
                f"Unknown aggregation_method: {aggregation_method!r}. "
                f"Choose from: 'sgcn', 'avg', 'weighted'."
            )
        weights = torch.softmax(kept_scores, dim=0)

    agg_state = {}
    for key in epoch_init_state:
        stacked = torch.stack(
            [local_states[i][key].float() for i in kept_idx], dim=0
        )
        w           = weights.view([-1] + [1] * (stacked.dim() - 1))
        agg_state[key] = (stacked * w).sum(dim=0).to(epoch_init_state[key].dtype)

    # ── 7. Load aggregated state; clear stale optimiser momentum ────────────
    model.load_state_dict({k: v.to(device) for k, v in agg_state.items()})
    optimizer.state.clear()

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start
    avg_loss   = loss_sum / valid_batches if valid_batches > 0 else 0.0
    return avg_loss, epoch_time, sampling_time


def train_epoch(model, dataloader, criterion, optimizer, device,
                use_labels=False, n_classes=112):
    model.train()
    loss_sum, total = 0, 0
    sampling_time   = 0.0

    _cuda_sync(device)
    epoch_start = time.time()

    # Manual iterator is used to time each batch fetch (sampling_time) separately
    # from the forward/backward pass without restructuring the training loop.
    loader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        t_sample = time.time()
        batch = next(loader_iter)
        sampling_time += time.time() - t_sample

        batch      = batch.to(device)
        batch_size = batch.batch_size

        if use_labels:
            non_seed_idx = torch.arange(batch_size, batch.x.shape[0], device=device)
            x = add_labels(batch.x, batch.train_labels_onehot, non_seed_idx, n_classes, device)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[:batch_size], batch.y[:batch_size].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_size
        total    += batch_size

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start

    return loss_sum / total, epoch_time, sampling_time


def train_epoch_saint(model, dataloader, criterion, optimizer, device,
                      train_idx, use_labels=False, n_classes=112):
    """Training epoch using GraphSAINT subgraph-sampling batches.

    GraphSAINT batches are induced subgraphs where every node may be a
    training node.  Training nodes are identified via ``batch.train_mask``.
    When the batch carries ``batch.node_norm`` (sampling normalisation
    weights produced by the GraphSAINT sampler), each per-node loss term is
    scaled by the corresponding weight before summing.
    """
    model.train()
    loss_sum, total = 0, 0
    sampling_time   = 0.0

    _cuda_sync(device)
    epoch_start = time.time()

    loader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        t_sample = time.time()
        batch = next(loader_iter)
        sampling_time += time.time() - t_sample

        batch = batch.to(device)

        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        if use_labels:
            # Reveal labels for non-training nodes only (same convention as
            # train_epoch, where seed-node labels are withheld).
            non_train_local_idx = torch.where(~train_mask)[0]
            x = add_labels(batch.x, batch.train_labels_onehot,
                           non_train_local_idx, n_classes, device)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)

        # Apply GraphSAINT sampling normalisation weights when available.
        if hasattr(batch, 'node_norm'):
            loss_per_node = F.binary_cross_entropy_with_logits(
                pred[train_mask], batch.y[train_mask].float(), reduction='none'
            )
            loss = (loss_per_node * batch.node_norm[train_mask]).sum()
        else:
            loss = criterion(pred[train_mask], batch.y[train_mask].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_train_in_batch = train_mask.sum().item()
        # node_norm path: loss is already a weighted sum; add directly.
        # fallback path: criterion returns a mean; scale back to a sum for
        # consistent per-node averaging across the epoch.
        if hasattr(batch, 'node_norm'):
            loss_sum += loss.item()
        else:
            loss_sum += loss.item() * n_train_in_batch
        total    += n_train_in_batch

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start

    return (loss_sum / total if total > 0 else 0.0), epoch_time, sampling_time


@torch.no_grad()
def evaluate(model, dataloader, labels, train_idx, val_idx, test_idx,
             criterion, evaluator, device, use_labels=False, n_classes=112):
    model.eval()
    preds      = torch.zeros(labels.shape, device=device)
    eval_times = 1

    _cuda_sync(device)
    eval_start = time.time()

    for _ in range(eval_times):
        for batch in dataloader:
            batch      = batch.to(device)
            batch_size = batch.batch_size

            if use_labels:
                all_idx = torch.arange(batch.x.shape[0], device=device)
                x = add_labels(batch.x, batch.train_labels_onehot, all_idx, n_classes, device)
            else:
                x = batch.x

            pred = model(x, batch.edge_index, batch.edge_attr)
            preds[batch.n_id[:batch_size]] += pred[:batch_size]

    preds /= eval_times

    _cuda_sync(device)
    eval_time = time.time() - eval_start

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss   = criterion(preds[val_idx],   labels[val_idx].float()).item()
    test_loss  = criterion(preds[test_idx],  labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx],   labels[val_idx]),
        evaluator(preds[test_idx],  labels[test_idx]),
        train_loss, val_loss, test_loss,
        preds,
        eval_time,
    )


def run(data, labels, train_idx, val_idx, test_idx, evaluator, n_running,
        gen_model_fn, device, n_layers, lr, weight_decay, n_epochs,
        eval_every, log_every, save_pred, use_labels=False, n_classes=112,
        mpnn='gcn',
        subsampling_method='random_node',
        truncation_ratio=0.2,
        aggregation_method='sgcn',
        n_subgraphs=5,
        subgraph_max_nodes=256,
        max_subgraph_edges=300000,
        debug_subgraph_stats=False):
    evaluator_wrapper = lambda pred, lbls: evaluator.eval(
        {'y_pred': pred, 'y_true': lbls}
    )['rocauc']

    train_batch_size = (len(train_idx) + 9) // 10

    if mpnn == 'graphsaint':
        # Attach boolean split masks to data so GraphSAINT batches inherit them.
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True

        # GraphSAINT: sample random-walk-induced subgraphs instead of
        # per-node neighborhoods.  num_steps mirrors the ~10 batches/epoch
        # produced by NeighborLoader, and walk_length provides a 2-hop reach.
        saint_num_steps = max(len(train_idx) // train_batch_size, 1)
        train_loader = GraphSAINTRandomWalkSampler(
            data,
            batch_size=train_batch_size,
            walk_length=2,
            num_steps=saint_num_steps,
            num_workers=4,
        )
    elif mpnn == 'sgcn':
        # SGCN handles its own subgraph sampling inside train_epoch_sgcn;
        # no external DataLoader is required.
        train_loader = None
    else:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[16] * n_layers,
            batch_size=train_batch_size,
            input_nodes=train_idx.cpu(),
            shuffle=True,
            num_workers=4,
        )

    eval_loader = NeighborLoader(
        data,
        num_neighbors=[32] * n_layers,
        batch_size=32768,
        input_nodes=torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
        shuffle=False,
        num_workers=4,
    )

    criterion    = nn.BCEWithLogitsLoss()
    model        = gen_model_fn().to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=50, verbose=True
    )

    best_val_score, final_test_score = 0, 0
    val_score, test_score = 0, 0
    final_pred  = None
    epoch_records = []

    _cuda_sync(device)
    run_start = time.time()

    for epoch in range(1, n_epochs + 1):
        if mpnn == 'graphsaint':
            loss, epoch_time, sampling_time = train_epoch_saint(
                model, train_loader, criterion, optimizer, device,
                train_idx, use_labels, n_classes
            )
        elif mpnn == 'sgcn':
            loss, epoch_time, sampling_time = train_epoch_sgcn(
                model, data, criterion, optimizer, device,
                train_idx, val_idx,
                n_subgraphs=n_subgraphs,
                subsampling_method=subsampling_method,
                subgraph_max_nodes=subgraph_max_nodes,
                max_subgraph_edges=max_subgraph_edges,
                truncation_ratio=truncation_ratio,
                aggregation_method=aggregation_method,
                use_labels=use_labels, n_classes=n_classes,
                debug_subgraph_stats=debug_subgraph_stats,
            )
        else:
            loss, epoch_time, sampling_time = train_epoch(
                model, train_loader, criterion, optimizer, device, use_labels, n_classes
            )

        record = {
            'epoch':               epoch,
            'train_loss':          loss,
            'val_auc':             float('nan'),
            'test_auc':            float('nan'),
            'train_sampling_time': sampling_time,
            'train_epoch_time':    epoch_time,
            'eval_time':           float('nan'),
        }

        if epoch == n_epochs or epoch % eval_every == 0 or epoch % log_every == 0:
            (train_score, val_score, test_score,
             train_loss, val_loss, test_loss,
             pred, eval_time) = evaluate(
                model, eval_loader, labels, train_idx, val_idx, test_idx,
                criterion, evaluator_wrapper, device, use_labels, n_classes
            )

            record['val_auc']   = val_score
            record['test_auc']  = test_score
            record['eval_time'] = eval_time

            if val_score > best_val_score:
                best_val_score   = val_score
                final_test_score = test_score
                final_pred       = pred

            if epoch % log_every == 0:
                print(
                    f'Epoch: {epoch:04d} | '
                    f'Loss: {loss:.4f} | '
                    f'Train: {100 * train_score:.2f}% | '
                    f'Valid: {100 * val_score:.2f}% | '
                    f'Test: {100 * test_score:.2f}% | '
                    f'Best Valid: {100 * best_val_score:.2f}% | '
                    f'Best Test: {100 * final_test_score:.2f}%'
                )

        epoch_records.append(record)
        lr_scheduler.step(val_score)

    _cuda_sync(device)
    total_run_time = time.time() - run_start

    if save_pred and final_pred is not None:
        os.makedirs('./output', exist_ok=True)
        torch.save(torch.sigmoid(final_pred), f'./output/{n_running}.pt')

    return {
        'best_val_auc':   best_val_score,
        'best_test_auc':  final_test_score,
        'final_val_auc':  val_score,
        'final_test_auc': test_score,
        'total_run_time': total_run_time,
        'epoch_records':  epoch_records,
    }

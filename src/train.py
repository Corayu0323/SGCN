import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler

from .utils import add_labels, gen_model


def _cuda_sync(device):
    """Synchronize CUDA if available to get accurate timing boundaries."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


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

    Unlike NeighborLoader batches (which separate seed nodes from context
    neighbors), GraphSAINT batches are induced subgraphs where every node
    may be a training node.  We identify the training nodes within each
    subgraph via ``batch.n_id`` and compute the loss only over them.
    """
    model.train()
    train_idx_device = train_idx.to(device)
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

        # Identify which nodes in this subgraph belong to the training set.
        train_mask = torch.isin(batch.n_id, train_idx_device)
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
        loss = criterion(pred[train_mask], batch.y[train_mask].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_train_in_batch = train_mask.sum().item()
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
        mpnn='gcn'):
    evaluator_wrapper = lambda pred, lbls: evaluator.eval(
        {'y_pred': pred, 'y_true': lbls}
    )['rocauc']

    train_batch_size = (len(train_idx) + 9) // 10

    if mpnn == 'graphsaint':
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

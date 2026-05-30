import random

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import coalesce, remove_self_loops, scatter, to_undirected

from .models import GNN_PyG


def apply_ood_perturbation(data, Pood, Pcr, seed):
    """Apply node-level OOD perturbation.

    Semantics (per selected node i):
      1) Select anomalous nodes with Bernoulli(Pood).
      2) Remove ALL original incident edges of i.
      3) Reconnect to *original-neighbor candidates* with probability Pcr.

    Important: "周围的邻居" here is interpreted as the *original* neighbors of the
    node (the neighborhood before disconnect). If you want to reconnect to
    non-neighbors (random nodes), you should instead sample from the complement
    set.

    For dense graphs (e.g., ogbn-proteins), sampling from non-neighbors can be
    extremely weak because there are very few non-neighbor candidates; in that
    case the perturbation barely changes the graph and ROC may not move.
    """
    if not (0.0 <= Pood <= 1.0):
        raise ValueError(f'Pood must be in [0, 1], got {Pood}')
    if not (0.0 <= Pcr <= 1.0):
        raise ValueError(f'Pcr must be in [0, 1], got {Pcr}')

    edge_index = data.edge_index
    device = edge_index.device
    num_nodes = int(data.num_nodes)
    if num_nodes == 0 or edge_index.numel() == 0:
        data_ood = data.clone()
        data_ood.edge_attr = None
        stats = {
            'selected_nodes': torch.empty(0, dtype=torch.long),
            'num_selected_nodes': 0,
            'selected_ratio': 0.0,
            'num_edges_before': int(edge_index.size(1)),
            'num_edges_after': int(edge_index.size(1)),
            'rewired_edges': 0,
            'max_rewired_per_node': 0,
            'broken_edges': 0,
            'added_edges': 0,
            'max_broken_per_node': 0,
            'max_added_per_node': 0,
            'mean_degree_before': 0.0,
            'mean_degree_after': 0.0,
            'degree_change': {},
            'degree_change_summary': {
                'mean': 0.0,
                'std': 0.0,
                'max_abs': 0.0,
            },
        }
        return data_ood, stats

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    undirected_no_loops, _ = remove_self_loops(edge_index)
    undirected = to_undirected(undirected_no_loops, num_nodes=num_nodes)
    undirected = coalesce(undirected, None, num_nodes, num_nodes)[0]
    u, v = undirected[0], undirected[1]
    if u.numel() == 0:
        data_ood = data.clone()
        data_ood.edge_attr = None
        stats = {
            'selected_nodes': torch.empty(0, dtype=torch.long),
            'num_selected_nodes': 0,
            'selected_ratio': 0.0,
            'num_edges_before': int(edge_index.size(1)),
            'num_edges_after': int(edge_index.size(1)),
            'rewired_edges': 0,
            'max_rewired_per_node': 0,
            'broken_edges': 0,
            'added_edges': 0,
            'max_broken_per_node': 0,
            'max_added_per_node': 0,
            'mean_degree_before': 0.0,
            'mean_degree_after': 0.0,
            'degree_change': {},
            'degree_change_summary': {
                'mean': 0.0,
                'std': 0.0,
                'max_abs': 0.0,
            },
        }
        return data_ood, stats

    # keep upper-triangular undirected representation (u < v) for rewiring.
    mask_upper = u < v
    u = u[mask_upper]
    v = v[mask_upper]
    num_undirected_edges = u.numel()

    node_candidates = torch.arange(num_nodes, device=device)
    select_mask = torch.rand(node_candidates.numel(), generator=generator, device=device) < Pood
    selected_nodes = node_candidates[select_mask]
    selected_nodes_cpu = selected_nodes.cpu()

    if selected_nodes.numel() == 0:
        edge_index_ood = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], dim=1)
        edge_index_ood = coalesce(edge_index_ood, None, num_nodes, num_nodes)[0]
        data_ood = data.clone()
        data_ood.edge_index = edge_index_ood
        data_ood.edge_attr = None
        stats = {
            'selected_nodes': selected_nodes_cpu,
            'num_selected_nodes': 0,
            'selected_ratio': 0.0,
            'num_edges_before': int(edge_index.size(1)),
            'num_edges_after': int(edge_index_ood.size(1)),
            'rewired_edges': 0,
            'max_rewired_per_node': 0,
            'broken_edges': 0,
            'added_edges': 0,
            'max_broken_per_node': 0,
            'max_added_per_node': 0,
            'mean_degree_before': 0.0,
            'mean_degree_after': 0.0,
            'degree_change': {},
            'degree_change_summary': {
                'mean': 0.0,
                'std': 0.0,
                'max_abs': 0.0,
            },
        }
        return data_ood, stats

    # Incidence index (CSR-like) for O(1) access to each node's incident edges.
    edge_ids = torch.arange(num_undirected_edges, device=device)
    inc_nodes = torch.cat([u, v], dim=0)
    inc_edge_ids = torch.cat([edge_ids, edge_ids], dim=0)
    inc_perm = inc_nodes.argsort()
    inc_nodes_sorted = inc_nodes[inc_perm]
    inc_edge_ids_sorted = inc_edge_ids[inc_perm]
    deg_before_all = torch.bincount(inc_nodes, minlength=num_nodes)

    remove_edge_ids = []
    add_u_list = []
    add_v_list = []
    broken_per_node = {}
    added_per_node = {}

    selected_nodes_list = selected_nodes.tolist()
    selected_starts = torch.searchsorted(inc_nodes_sorted, selected_nodes, right=False).tolist()
    selected_ends = torch.searchsorted(inc_nodes_sorted, selected_nodes, right=True).tolist()
    for node, start, end in zip(selected_nodes_list, selected_starts, selected_ends):
        incident_ids = inc_edge_ids_sorted[start:end]
        degree = int(incident_ids.numel())
        broken_per_node[node] = degree
        if degree > 0:
            remove_edge_ids.append(incident_ids)

        if degree == 0:
            added_per_node[node] = 0
            continue

        if Pcr <= 0.0:
            added_per_node[node] = 0
            continue

        # Original neighbors of this node (before removing edges)
        neighbors = torch.where(u[incident_ids] == node, v[incident_ids], u[incident_ids])

        # Reconnect to these original neighbors with probability Pcr
        sampled_mask = torch.rand(neighbors.numel(), generator=generator, device=device) < Pcr
        new_neighbors = neighbors[sampled_mask]
        if new_neighbors.numel() == 0:
            added_per_node[node] = 0
            continue

        # Ensure undirected upper-triangular representation (min, max)
        node_vec = torch.full((new_neighbors.numel(),), node, dtype=torch.long, device=device)
        add_u = torch.minimum(node_vec, new_neighbors)
        add_v = torch.maximum(node_vec, new_neighbors)

        add_u_list.append(add_u)
        add_v_list.append(add_v)
        added_per_node[node] = int(new_neighbors.numel())

    keep_mask = torch.ones(num_undirected_edges, dtype=torch.bool, device=device)
    total_broken = 0
    if remove_edge_ids:
        remove_ids = torch.unique(torch.cat(remove_edge_ids, dim=0))
        keep_mask[remove_ids] = False
        total_broken = int(remove_ids.numel())

    if add_u_list:
        add_u = torch.cat(add_u_list, dim=0)
        add_v = torch.cat(add_v_list, dim=0)
        u_new = torch.cat([u[keep_mask], add_u], dim=0)
        v_new = torch.cat([v[keep_mask], add_v], dim=0)
    else:
        u_new = u[keep_mask]
        v_new = v[keep_mask]

    undirected_new = torch.stack([u_new, v_new], dim=0)
    undirected_new = coalesce(undirected_new, None, num_nodes, num_nodes)[0]
    undirected_new = undirected_new[:, undirected_new[0] < undirected_new[1]]
    edge_index_ood = torch.cat([undirected_new, undirected_new.flip(0)], dim=1)
    edge_index_ood = coalesce(edge_index_ood, None, num_nodes, num_nodes)[0]

    # Degree stats on selected nodes (undirected degree).
    deg_after_all = torch.bincount(
        torch.cat([undirected_new[0], undirected_new[1]], dim=0),
        minlength=num_nodes,
    )
    selected_before = deg_before_all[selected_nodes]
    selected_after = deg_after_all[selected_nodes]
    selected_change = selected_after - selected_before

    degree_change = {
        int(n): int(dc)
        for n, dc in zip(selected_nodes_cpu.tolist(), selected_change.cpu().tolist())
    }
    mean_before = float(selected_before.float().mean().item()) if selected_before.numel() > 0 else 0.0
    mean_after = float(selected_after.float().mean().item()) if selected_after.numel() > 0 else 0.0
    change_mean = float(selected_change.float().mean().item()) if selected_change.numel() > 0 else 0.0
    change_std = float(selected_change.float().std(unbiased=False).item()) if selected_change.numel() > 0 else 0.0
    change_max_abs = float(selected_change.abs().max().item()) if selected_change.numel() > 0 else 0.0

    data_ood = data.clone()
    data_ood.edge_index = edge_index_ood
    data_ood.edge_attr = None

    stats = {
        'selected_nodes': selected_nodes_cpu,
        'num_selected_nodes': int(selected_nodes.numel()),
        'selected_ratio': float(selected_nodes.numel() / num_nodes),
        'num_edges_before': int(edge_index.size(1)),
        'num_edges_after': int(edge_index_ood.size(1)),
        # Backward-compatible fields:
        'rewired_edges': int(total_broken),
        'max_rewired_per_node': int(max(broken_per_node.values(), default=0)),
        # Node-level disconnect/reconnect stats:
        'broken_edges': int(total_broken),
        'added_edges': int(sum(added_per_node.values())),
        'max_broken_per_node': int(max(broken_per_node.values(), default=0)),
        'max_added_per_node': int(max(added_per_node.values(), default=0)),
        'mean_degree_before': mean_before,
        'mean_degree_after': mean_after,
        'degree_change': degree_change,
        'degree_change_summary': {
            'mean': change_mean,
            'std': change_std,
            'max_abs': change_max_abs,
        },
    }
    return data_ood, stats


def set_seed(seed_val=0):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_name):
    dataset_obj = PygNodePropPredDataset(name=dataset_name, root='/mnt/SGCN/dataset')
    evaluator   = Evaluator(name=dataset_name)
    split_idx   = dataset_obj.get_idx_split()
    train_idx   = split_idx['train']
    val_idx     = split_idx['valid']
    test_idx    = split_idx['test']
    data        = dataset_obj[0]
    return data, train_idx, val_idx, test_idx, evaluator


def preprocess(data, train_idx, n_classes):
    # Aggregate edge features to node features via sum of incoming edges
    x = scatter(data.edge_attr, data.edge_index[1], dim=0,
                dim_size=data.num_nodes, reduce='sum')
    data.x = x

    # Training labels as additional input features (others stay zero)
    data.train_labels_onehot = torch.zeros(data.num_nodes, n_classes)
    data.train_labels_onehot[train_idx, data.y[train_idx, 0].long()] = 1
    return data


def gen_model(n_node_feats, n_classes, use_labels, n_layers, n_hidden,
              dropout, input_drop, edge_drop, mpnn, jk):
    n_feats = (n_node_feats + n_classes) if use_labels else n_node_feats
    return GNN_PyG(
        n_feats,
        n_classes,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activation=F.relu,
        dropout=dropout,
        input_drop=input_drop,
        edge_drop=edge_drop,
        mpnn=mpnn,
        jk=jk,
    )


def add_labels(x, train_labels_onehot, idx, n_classes, device):
    """Concatenate one-hot training labels to node features for the given indices."""
    labels_onehot = torch.zeros([x.shape[0], n_classes], device=device)
    labels_onehot[idx] = train_labels_onehot[idx].to(device)
    return torch.cat([x, labels_onehot], dim=-1)

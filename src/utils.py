import random

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter

from .models import GNN_PyG


# ── obnb helpers ─────────────────────────────────────────────────────────────

class OBNBEvaluator:
    """Evaluator for obnb datasets.

    Mimics the OGB Evaluator interface so it can be used as a drop-in
    replacement inside ``run()``.

    Usage
    -----
    evaluator = OBNBEvaluator()
    result    = evaluator.eval({'y_pred': pred_tensor, 'y_true': true_tensor})
    rocauc    = result['rocauc']   # mean ROC-AUC across all tasks
    """

    def eval(self, input_dict):
        """Compute mean ROC-AUC across all tasks.

        Parameters
        ----------
        input_dict : dict
            Must contain:
              'y_pred' – Tensor or ndarray of shape (n_nodes, n_tasks).
                         Raw logits are accepted (rank-equivalent to probs).
              'y_true' – Tensor or ndarray of shape (n_nodes, n_tasks).
                         Binary labels.

        Returns
        -------
        dict with key 'rocauc' → float (mean across tasks that have >1 class).
        """
        from sklearn.metrics import roc_auc_score

        y_pred = input_dict['y_pred']
        y_true = input_dict['y_true']

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        scores = []
        for t in range(y_true.shape[1]):
            gt = y_true[:, t]
            pd = y_pred[:, t]
            # Skip tasks where only one class is present in this split.
            if len(np.unique(gt)) > 1:
                scores.append(roc_auc_score(gt, pd))

        mean_auc = float(np.mean(scores)) if scores else 0.0
        return {'rocauc': mean_auc}


def load_obnb_data(graph_name, label_name, encoder_type='one_hot_log_deg',
                   root='datasets', version='current'):
    """Load an obnb biomedical network benchmark dataset.

    Parameters
    ----------
    graph_name : str
        Name of the biological network, e.g. 'BioGRID', 'STRING', 'HumanNet'.
    label_name : str
        Name of the label set, e.g. 'DisGeNET', 'GOBP', 'GOMF'.
    encoder_type : str
        Node-feature encoding strategy:
          'one_hot_log_deg' – 32-dim one-hot encoded log degree (default).
          'adj'             – row of the dense adjacency matrix (n_nodes-dim).
    root : str
        Root directory for caching downloaded obnb data (default: 'datasets').
    version : str
        Data version: 'current' (archived), 'latest' (download from source),
        or a specific version string (default: 'current').

    Returns
    -------
    data       : torch_geometric.data.Data – full graph with node features/labels.
    train_idx  : LongTensor – training node indices.
    val_idx    : LongTensor – validation node indices.
    test_idx   : LongTensor – test node indices.
    evaluator  : OBNBEvaluator – compatible with run()'s evaluator interface.

    Notes
    -----
    * ``data.y`` has shape ``(num_nodes, num_tasks)``; ``num_tasks`` depends on
      the chosen network/label combination.
    * The study-bias holdout split is gene-level (based on PubMed citation
      count), so the same train/val/test assignment applies to all tasks.
    * ``use_labels`` is **not** supported for obnb datasets.  Keep
      ``USE_LABELS = False`` when using this loader.
    """
    # Lazy import: obnb is an optional dependency.  Importing at module level
    # would break the rest of the codebase for users who have not installed it.
    from obnb.dataset import OpenBiomedNetBench

    if encoder_type == 'adj':
        graph_as_feature    = True
        use_dense_graph     = True
        auto_generate_feature = None
    elif encoder_type == 'one_hot_log_deg':
        graph_as_feature    = False
        use_dense_graph     = False
        auto_generate_feature = 'OneHotLogDeg'
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type!r}. "
            "Choose from: 'one_hot_log_deg', 'adj'."
        )

    dataset = OpenBiomedNetBench(
        root=root,
        graph_name=graph_name,
        label_name=label_name,
        version=version,
        graph_as_feature=graph_as_feature,
        use_dense_graph=use_dense_graph,
        auto_generate_feature=auto_generate_feature,
    )
    data = dataset.to_pyg_data()

    # Derive global split indices from per-task masks.
    # obnb study-bias splits are gene-level (driven by PubMedCount), so every
    # task column contains the same binary assignment.  Column 0 is used as the
    # canonical global indicator.
    def _mask_to_idx(mask):
        if mask.dim() == 2:
            mask = mask[:, 0]
        return mask.nonzero(as_tuple=False).squeeze(1)

    train_idx = _mask_to_idx(data.train_mask)
    val_idx   = _mask_to_idx(data.val_mask)
    test_idx  = _mask_to_idx(data.test_mask)

    evaluator = OBNBEvaluator()
    return data, train_idx, val_idx, test_idx, evaluator


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

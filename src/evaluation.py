import torch
import pandas as pd


def _as_parent_index_tensors(go_parents, device):
    """Convert GO parent-child relations to child/parent index tensors."""
    if isinstance(go_parents, pd.DataFrame):
        if go_parents.empty:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        required_cols = {'child_idx', 'parent_idx'}
        missing_cols = required_cols.difference(go_parents.columns)
        if missing_cols:
            missing = ', '.join(sorted(missing_cols))
            raise ValueError(f"go_parents DataFrame is missing columns: {missing}")

        child_idx = torch.as_tensor(
            go_parents['child_idx'].to_numpy(), dtype=torch.long, device=device
        )
        parent_idx = torch.as_tensor(
            go_parents['parent_idx'].to_numpy(), dtype=torch.long, device=device
        )
        return child_idx, parent_idx

    if go_parents is None or len(go_parents) == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    parent_pairs = torch.as_tensor(go_parents, dtype=torch.long, device=device)
    if parent_pairs.ndim != 2 or parent_pairs.size(1) != 2:
        raise ValueError(
            "go_parents must be a DataFrame with child_idx/parent_idx columns "
            "or a sequence of (child_idx, parent_idx) pairs"
        )

    return parent_pairs[:, 0], parent_pairs[:, 1]


def compute_tpr_violation(probs, go_parents, node_mask=None, verbose=False):
    """
    Compute TPR (True Path Rule) Violation Rate.

    Args:
        probs: torch.Tensor, shape [N, C]
            Model prediction probabilities after sigmoid.
        go_parents:
            Either a pandas DataFrame with 'child_idx' and 'parent_idx'
            columns, or a list/sequence of (child_idx, parent_idx) pairs.
        node_mask: torch.Tensor, shape [N], optional
            Nodes to evaluate on, such as val/test mask or index tensor.
            If None, all nodes are evaluated.
        verbose: bool
            Whether to print summary statistics.

    Returns:
        violation_rate: float
        stats: dict
    """
    if not torch.is_tensor(probs):
        raise TypeError("probs must be a torch.Tensor")
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape [N, C], got {tuple(probs.shape)}")

    num_nodes, num_labels = probs.shape
    device = probs.device

    if node_mask is None:
        probs_eval = probs
    else:
        if not torch.is_tensor(node_mask):
            node_mask = torch.as_tensor(node_mask)
        node_mask = node_mask.to(device)

        if node_mask.ndim != 1:
            raise ValueError(f"node_mask must be one-dimensional, got shape {tuple(node_mask.shape)}")

        if node_mask.dtype == torch.bool:
            if node_mask.numel() != num_nodes:
                raise ValueError(
                    "Boolean node_mask must be one-dimensional with length "
                    f"{num_nodes}, got shape {tuple(node_mask.shape)}"
                )
            probs_eval = probs[node_mask]
        else:
            if torch.is_floating_point(node_mask):
                raise TypeError("node_mask must be a boolean mask or an integer index tensor")
            probs_eval = probs[node_mask.long()]

    nodes_evaluated = int(probs_eval.size(0))
    child_idx, parent_idx = _as_parent_index_tensors(go_parents, device)
    num_pairs = int(child_idx.numel())

    if num_pairs == 0 or nodes_evaluated == 0:
        stats = {
            'nodes_evaluated': nodes_evaluated,
            'parent_child_pairs': num_pairs,
            'total_comparisons': 0,
            'violation_count': 0,
            'violation_rate': 0.0,
        }
        if verbose:
            print("========== TPR Violation Analysis ==========")
            print(f"Nodes evaluated: {nodes_evaluated}")
            print(f"Parent-child pairs: {num_pairs}")
            print("Total comparisons: 0")
            print("Violations found: 0")
            print("Violation rate: 0.000000 (0.00%)")
            print("============================================")
        return 0.0, stats

    min_label_idx = int(torch.minimum(child_idx.min(), parent_idx.min()).item())
    max_label_idx = int(torch.maximum(child_idx.max(), parent_idx.max()).item())
    if min_label_idx < 0 or max_label_idx >= num_labels:
        raise ValueError(
            "go_parents contains label indices outside probs columns: "
            f"valid range is [0, {num_labels - 1}], got [{min_label_idx}, {max_label_idx}]"
        )

    child_probs = probs_eval[:, child_idx]
    parent_probs = probs_eval[:, parent_idx]
    violation_count = int((child_probs > parent_probs).sum().item())
    total_comparisons = nodes_evaluated * num_pairs
    violation_rate = violation_count / total_comparisons

    stats = {
        'nodes_evaluated': nodes_evaluated,
        'parent_child_pairs': num_pairs,
        'total_comparisons': int(total_comparisons),
        'violation_count': violation_count,
        'violation_rate': float(violation_rate),
    }

    if verbose:
        print("========== TPR Violation Analysis ==========")
        print(f"Nodes evaluated: {nodes_evaluated}")
        print(f"Parent-child pairs: {num_pairs}")
        print(f"Total comparisons: {total_comparisons:,}")
        print(f"Violations found: {violation_count:,}")
        print(f"Violation rate: {violation_rate:.6f} ({violation_rate * 100:.2f}%)")
        print("============================================")

    return float(violation_rate), stats

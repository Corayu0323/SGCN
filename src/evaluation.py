import argparse
import glob
import json
from pathlib import Path

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


def _load_tensor(path, key=None):
    """Load a tensor-like object from .pt/.pth, .npy, or csv/txt files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {'.pt', '.pth'}:
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, dict):
            if key is not None:
                if key not in obj:
                    raise KeyError(f"Key '{key}' not found in {path}")
                obj = obj[key]
            else:
                for candidate in ('probs', 'prob', 'pred', 'preds', 'y_pred', 'logits'):
                    if candidate in obj:
                        obj = obj[candidate]
                        break
                else:
                    keys = ', '.join(obj.keys())
                    raise ValueError(
                        f"{path} contains a dict. Pass --probs-key. Available keys: {keys}"
                    )
        return obj if torch.is_tensor(obj) else torch.as_tensor(obj)

    if suffix == '.npy':
        import numpy as np

        return torch.as_tensor(np.load(path))

    if suffix in {'.csv', '.txt'}:
        return torch.as_tensor(pd.read_csv(path, header=None).values)

    raise ValueError(f"Unsupported tensor file extension: {path.suffix}")


def _default_project_root():
    return Path(__file__).resolve().parents[1]


def _default_go_parents_path():
    return _default_project_root() / 'dataset' / 'ogbn_proteins' / 'mapping' / 'go_parents.csv'


def _find_latest_prediction():
    project_root = _default_project_root()
    candidates = []
    search_dirs = list((project_root / 'results').glob('*/preds'))
    search_dirs.append(project_root / 'output')

    for search_dir in search_dirs:
        for pattern in ('*.pt', '*.pth', '*.npy', '*.csv'):
            candidates.extend(glob.glob(str(search_dir / pattern)))
    if not candidates:
        return None
    return max(candidates, key=lambda p: Path(p).stat().st_mtime)


def _load_split_indices(split):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            "Loading --split requires ogb. Install ogb or pass --node-mask explicitly."
        ) from exc

    dataset = PygNodePropPredDataset(
        name='ogbn-proteins',
        root=str(_default_project_root() / 'dataset'),
    )
    split_idx = dataset.get_idx_split()
    return split_idx[split]


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compute GO True Path Rule violation rate from saved model predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--probs',
        default=None,
        help=(
            "Path to saved probabilities/logits (.pt, .pth, .npy, .csv). "
            "If omitted, the latest file in ../results/*/preds or ../output is used."
        ),
    )
    parser.add_argument(
        '--probs-key',
        default=None,
        help="Key to read when --probs points to a torch-saved dict.",
    )
    parser.add_argument(
        '--logits',
        action='store_true',
        help="Treat input as logits and apply sigmoid before computing violations.",
    )
    parser.add_argument(
        '--go-parents',
        default=str(_default_go_parents_path()),
        help="Path to go_parents.csv with child_idx and parent_idx columns.",
    )
    parser.add_argument(
        '--node-mask',
        default=None,
        help="Optional boolean mask or index tensor file (.pt, .pth, .npy, .csv, .txt).",
    )
    parser.add_argument(
        '--mask-key',
        default=None,
        help="Key to read when --node-mask points to a torch-saved dict.",
    )
    parser.add_argument(
        '--split',
        choices=('train', 'valid', 'test'),
        default=None,
        help="Evaluate on an OGB split. Ignored if --node-mask is provided.",
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Also print stats as one JSON object for logs/tables.",
    )
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    probs_path = args.probs or _find_latest_prediction()
    if probs_path is None:
        parser.print_help()
        print()
        print("No prediction file found in ../results/*/preds or ../output.")
        print("Set SAVE_PRED=True in the notebook/training config, run training, then rerun:")
        print("  python src/evaluation.py --split test")
        return 1

    probs = _load_tensor(probs_path, args.probs_key).float()
    if args.logits:
        probs = torch.sigmoid(probs)

    go_parents = pd.read_csv(args.go_parents)

    node_mask = None
    if args.node_mask is not None:
        node_mask = _load_tensor(args.node_mask, args.mask_key)
    elif args.split is not None:
        node_mask = _load_split_indices(args.split)

    print(f"Prediction file: {probs_path}")
    print(f"GO parent file: {args.go_parents}")
    print(f"Evaluation split: {args.split if args.node_mask is None else args.node_mask}")

    violation_rate, stats = compute_tpr_violation(
        probs=probs,
        go_parents=go_parents,
        node_mask=node_mask,
        verbose=True,
    )

    if args.json:
        print(json.dumps(stats, ensure_ascii=False))

    print(f"TPR Violation Rate: {violation_rate:.6f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

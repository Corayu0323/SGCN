from __future__ import annotations

import argparse
import csv
import glob
import gzip
from pathlib import Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_label_mapping_path() -> Path:
    return (
        _default_project_root()
        / "dataset"
        / "ogbn_proteins"
        / "mapping"
        / "labelidx2GO.csv.gz"
    )


def _find_latest_prediction() -> Path | None:
    project_root = _default_project_root()
    candidates = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend(
            glob.glob(str(project_root / "results" / "*" / "preds" / pattern))
        )
    if not candidates:
        return None
    return Path(max(candidates, key=lambda p: Path(p).stat().st_mtime))


def _load_tensor(path: str | Path, key: str | None = None) -> torch.Tensor:
    import torch

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pth"}:
        raise ValueError(f"Unsupported prediction format: {path.suffix}. Use .pt/.pth")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}")
            obj = obj[key]
        else:
            for candidate in ("probs", "prob", "pred", "preds", "y_pred", "logits"):
                if candidate in obj:
                    obj = obj[candidate]
                    break
            else:
                keys = ", ".join(obj.keys())
                raise ValueError(
                    f"{path} contains a dict. Pass --probs-key. Available keys: {keys}"
                )

    if torch.is_tensor(obj):
        return obj
    return torch.as_tensor(obj)


def _infer_run_id(path: Path) -> str:
    stem = path.stem
    return stem if stem else "unknown"


def _load_label_mapping(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {path}\n"
            "Expected ogbn-proteins mapping file 'labelidx2GO.csv.gz'."
        )

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return {}

    header = rows[0]
    data_rows = rows[1:]
    used_named_header = False

    idx_col = None
    go_col = None
    for i, c in enumerate(header):
        lc = c.strip().lower()
        if idx_col is None and ("idx" in lc or lc == "label"):
            idx_col = i
            used_named_header = True
        if go_col is None and "go" in lc:
            go_col = i
            used_named_header = True

    if idx_col is None:
        idx_col = 0 if len(header) >= 1 else None
    if go_col is None:
        go_col = 1 if len(header) >= 2 else None

    if idx_col is None or go_col is None:
        raise ValueError(
            f"Cannot parse label mapping columns from {path}. Header: {header}"
        )
    if not used_named_header:
        data_rows = rows

    mapping = {}
    for row in data_rows:
        if len(row) <= max(idx_col, go_col):
            continue
        idx_raw = row[idx_col].strip()
        go_raw = row[go_col].strip()
        if not idx_raw or not go_raw:
            continue
        mapping[int(idx_raw)] = go_raw
    return mapping


def _load_split_and_labels(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            "ogb is required for split loading. Please install ogb first."
        ) from exc

    dataset = PygNodePropPredDataset(
        name="ogbn-proteins",
        root=str(_default_project_root() / "dataset"),
    )
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    return split_idx[split], data.y


def _format_threshold_for_filename(threshold: float) -> str:
    return f"{threshold:g}"


def _build_detail_output_path(probs_path: Path, threshold: float) -> Path:
    threshold_str = _format_threshold_for_filename(threshold)
    return probs_path.parent / f"{probs_path.stem}_fpfn_threshold{threshold_str}.tsv"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline FP/FN error analysis for ogbn-proteins multi-label predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probs",
        default=None,
        help=(
            "Path to prediction probabilities (.pt/.pth). "
            "If omitted, the newest file in results/*/preds/*.pt|*.pth is used."
        ),
    )
    parser.add_argument(
        "--probs-key",
        default=None,
        help="Key to use when --probs points to a torch-saved dict.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="Dataset split for analysis.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize probabilities.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K labels to show for FP/FN summary.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Write per-node per-label FP/FN details to a TSV file.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20000,
        help="Maximum detail rows to write when --detail is enabled.",
    )
    parser.add_argument(
        "--label-mapping",
        default=str(_default_label_mapping_path()),
        help="Path to labelidx2GO mapping csv(.gz).",
    )
    return parser


def _topk_label_counts(mask: torch.Tensor, k: int) -> list[tuple[int, int]]:
    import torch

    counts = mask.sum(dim=0).to(torch.long)
    if counts.numel() == 0 or counts.max().item() == 0:
        return []
    k = min(k, counts.numel())
    values, indices = torch.topk(counts, k=k)
    pairs = []
    for label_idx, count in zip(indices.tolist(), values.tolist()):
        if count <= 0:
            continue
        pairs.append((int(label_idx), int(count)))
    return pairs


def _write_detail_file(
    output_path: Path,
    fp_mask: torch.Tensor,
    fn_mask: torch.Tensor,
    split_indices: torch.Tensor,
    probs_split: torch.Tensor,
    label_mapping: dict[int, str],
    max_items: int,
) -> tuple[int, bool]:
    import torch

    fp_pos = torch.nonzero(fp_mask, as_tuple=False)
    fn_pos = torch.nonzero(fn_mask, as_tuple=False)
    total_items = fp_pos.size(0) + fn_pos.size(0)
    limit = min(max_items, total_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for positions, err_type, true_v, pred_v in (
            (fp_pos, "FP", 0, 1),
            (fn_pos, "FN", 1, 0),
        ):
            for local_node_idx, label_idx in positions.tolist():
                if written >= limit:
                    break
                node_id = int(split_indices[local_node_idx].item())
                label_idx = int(label_idx)
                go_term = label_mapping.get(label_idx, "N/A")
                prob = float(probs_split[local_node_idx, label_idx].item())
                f.write(
                    f"node_id={node_id}\tlabel_idx={label_idx}\tgo_term={go_term}"
                    f"\ttype={err_type}\tprob={prob:.6f}\ttrue={true_v}\tpred={pred_v}\n"
                )
                written += 1
            if written >= limit:
                break

    truncated = total_items > max_items
    return written, truncated


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    import torch

    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("--threshold must be in [0, 1].")
    if args.max_items <= 0:
        raise ValueError("--max-items must be > 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0.")

    probs_path = Path(args.probs) if args.probs else _find_latest_prediction()
    if probs_path is None:
        parser.print_help()
        print()
        print("No prediction file found under results/*/preds/*.pt|*.pth")
        return 1
    if not probs_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {probs_path}")

    probs = _load_tensor(probs_path, args.probs_key).float()
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [num_nodes, num_labels], got {tuple(probs.shape)}")

    split_indices, y_true = _load_split_and_labels(args.split)
    y_true = torch.as_tensor(y_true).float()
    if y_true.ndim != 2:
        raise ValueError(f"Expected labels with shape [num_nodes, num_labels], got {tuple(y_true.shape)}")
    if probs.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch between probs {tuple(probs.shape)} and labels {tuple(y_true.shape)}"
        )

    split_indices = torch.as_tensor(split_indices).long()
    probs_split = probs[split_indices]
    y_true_split = y_true[split_indices]

    y_pred_split = probs_split > args.threshold
    y_true_bin = y_true_split > 0.5

    fp_mask = y_pred_split & (~y_true_bin)
    fn_mask = (~y_pred_split) & y_true_bin

    fp_total = int(fp_mask.sum().item())
    fn_total = int(fn_mask.sum().item())
    num_nodes = int(split_indices.numel())
    num_labels = int(probs.size(1))

    label_mapping_path = Path(args.label_mapping)
    label_mapping = _load_label_mapping(label_mapping_path)

    detail_output_path = None
    detail_written = 0
    truncated = False
    if args.detail:
        detail_output_path = _build_detail_output_path(probs_path, args.threshold)
        detail_written, truncated = _write_detail_file(
            output_path=detail_output_path,
            fp_mask=fp_mask,
            fn_mask=fn_mask,
            split_indices=split_indices,
            probs_split=probs_split,
            label_mapping=label_mapping,
            max_items=args.max_items,
        )

    run_id = _infer_run_id(probs_path)
    fp_topk = _topk_label_counts(fp_mask, args.top_k)
    fn_topk = _topk_label_counts(fn_mask, args.top_k)

    print("========== FP/FN Error Analysis ==========")
    print(f"run_id: {run_id}")
    print(f"probs_file: {probs_path}")
    print(f"split: {args.split}")
    print(f"threshold: {args.threshold}")
    print(f"nodes_in_split: {num_nodes}")
    print(f"num_labels: {num_labels}")
    print(f"total_fp: {fp_total}")
    print(f"total_fn: {fn_total}")
    print(f"detail_enabled: {args.detail}")
    print(f"detail_output: {detail_output_path if detail_output_path else 'N/A'}")
    if args.detail:
        print(f"detail_rows_written: {detail_written}")
        print(f"detail_truncated: {truncated}")
    print()
    print(f"Top-{args.top_k} FP labels:")
    if fp_topk:
        for label_idx, count in fp_topk:
            print(f"  label_idx={label_idx}\tgo_term={label_mapping.get(label_idx, 'N/A')}\tfp_count={count}")
    else:
        print("  (none)")
    print()
    print(f"Top-{args.top_k} FN labels:")
    if fn_topk:
        for label_idx, count in fn_topk:
            print(f"  label_idx={label_idx}\tgo_term={label_mapping.get(label_idx, 'N/A')}\tfn_count={count}")
    else:
        print("  (none)")
    print("===========================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Supervised clustering metrics, reported on a scale of 100.

The default metrics are ARI, NMI, FMI, homogeneity and completeness.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


DEFAULT_METRICS = (
    "ARI",
    "NMI",
    "FMI",
    "Homogeneity",
    "Completeness",
)
SUPPORTED_SUPERVISED_METRICS = DEFAULT_METRICS


def canonical_metric_name(name):
    """Return the canonical metric name used in output files."""
    return name


def _select_metrics(metrics, available):
    if metrics is None:
        raise ValueError("metrics must be explicitly specified.")

    selected = []
    for name in metrics:
        name = canonical_metric_name(name)
        if name not in available:
            raise ValueError(
                f"Unknown metric {name!r}. Available metrics: {tuple(available)}"
            )
        if name not in selected:
            selected.append(name)

    if not selected:
        raise ValueError("metrics must not be empty.")
    return tuple(selected)


def _check_labels(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if pd.isna(values).any():
        raise ValueError(f"{name} contains missing values.")
    return values


def _check_label_pair(gt, pred):
    gt = _check_labels(gt, "gt")
    pred = _check_labels(pred, "pred")
    if len(gt) != len(pred):
        raise ValueError(f"Length mismatch: gt={len(gt)}, pred={len(pred)}.")
    return gt, pred


def evaluate_supervised(
    gt,
    pred,
    *,
    metrics,
    scale=100.0,
):
    """Compute explicitly selected supervised clustering metrics."""
    gt, pred = _check_label_pair(gt, pred)
    selected = _select_metrics(metrics, SUPPORTED_SUPERVISED_METRICS)

    functions = {
        "ARI": adjusted_rand_score,
        "NMI": normalized_mutual_info_score,
        "FMI": fowlkes_mallows_score,
        "Homogeneity": homogeneity_score,
        "Completeness": completeness_score,
    }

    scores = {}
    for name in selected:
        value = functions[name](gt, pred)
        scores[name] = float(value) * scale
    return scores


def load_aligned_predictions(path, benchmark_barcodes):
    """Load predictions and return labels plus benchmark-order row indices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, dtype={"barcode": str})
    required = {"barcode", "predicted_label"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")
    if frame["barcode"].duplicated().any():
        raise ValueError(f"{path}: duplicated barcode")
    if frame["barcode"].isna().any():
        raise ValueError(f"{path}: barcode contains missing values")
    if frame["predicted_label"].isna().any():
        raise ValueError(f"{path}: predicted_label contains missing values")

    benchmark_barcodes = pd.Index(benchmark_barcodes, dtype=str)
    if not benchmark_barcodes.is_unique:
        raise ValueError("Dataset barcodes must be unique")
    result_barcodes = pd.Index(frame["barcode"].astype(str))

    missing = benchmark_barcodes.difference(result_barcodes)
    extra = result_barcodes.difference(benchmark_barcodes)
    if len(missing) or len(extra):
        raise ValueError(
            f"{path}: barcode mismatch; missing={len(missing)}, extra={len(extra)}"
        )

    positions = pd.Series(np.arange(len(frame)), index=result_barcodes)
    order = positions.loc[benchmark_barcodes].to_numpy(dtype=int)
    predictions = frame["predicted_label"].to_numpy()[order]
    return predictions, order

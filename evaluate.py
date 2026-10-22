#!/usr/bin/env python3
"""Evaluate CORCD predictions and summarize the five main-experiment seeds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corcd.data import BENCHMARK_SEEDS, DATASET_CONFIG, load_evaluation_data
from corcd.evaluation import (
    DEFAULT_METRICS,
    SUPPORTED_SUPERVISED_METRICS,
    canonical_metric_name,
    evaluate_supervised,
    load_aligned_predictions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("all", *DATASET_CONFIG), default="all",
        help="Dataset to evaluate; defaults to all six main datasets.",
    )
    parser.add_argument(
        "--seed", type=int, choices=BENCHMARK_SEEDS,
        help="Evaluate one seed only; omit to require all five seeds and summarize.",
    )
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument(
        "--result-dir", type=Path,
        help="Evaluate this single run directory; requires a specific --dataset and --seed.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        choices=SUPPORTED_SUPERVISED_METRICS,
        help="Main-experiment metrics to report, all scaled by 100.",
    )
    args = parser.parse_args(argv)
    if args.result_dir is not None and (args.dataset == "all" or args.seed is None):
        parser.error("--result-dir requires a specific --dataset and --seed")
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(result_dir, dataset, seed):
    """Require the exact successful training run, independent of old metrics."""
    info_path = result_dir / "run_info.json"
    prediction_path = result_dir / "predictions.csv"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected = {"method": "CORCD", "dataset": dataset, "seed": seed, "status": "success"}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"{info_path}: expected {key}={value!r}, got {info.get(key)!r}"
            )
    if not isinstance(info["seed"], int):
        raise ValueError(f"{info_path}: seed must be an integer")
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    return {
        "run_info_sha256": _sha256(info_path),
        "predictions_sha256": _sha256(prediction_path),
    }


def evaluate_run(data, result_dir, dataset, seed, metrics, fingerprint):
    """Align every predicted spot before applying the ground-truth label mask."""
    pred, _ = load_aligned_predictions(result_dir / "predictions.csv", data["obs_names"])
    mask = data["label_mask"]
    scores = evaluate_supervised(
        gt=data["y_true"], pred=pred[mask], metrics=metrics,
    )
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError(f"{dataset}/seed_{seed}: evaluation produced non-finite metrics")
    if validate_run(result_dir, dataset, seed) != fingerprint:
        raise RuntimeError(f"{result_dir}: training outputs changed during evaluation")
    return {
        "method": "CORCD",
        "dataset": dataset,
        "seed": seed,
        "requested_metrics": list(metrics),
        "n_spots": len(pred),
        "n_labeled_spots": int(mask.sum()),
        **fingerprint,
        "metrics": scores,
    }


def write_run_outputs(result_dir, payload):
    """Replace metrics with freshly calculated values for this exact run."""
    json_path = result_dir / "metrics.json"
    json_tmp = result_dir / "metrics.json.tmp"
    json_tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    json_tmp.replace(json_path)
    row = {
        "method": payload["method"], "dataset": payload["dataset"],
        "seed": payload["seed"], **payload["metrics"],
    }
    csv_tmp = result_dir / "metrics.csv.tmp"
    pd.DataFrame([row]).to_csv(csv_tmp, index=False)
    csv_tmp.replace(result_dir / "metrics.csv")


def write_summary(results_root, payloads, datasets, metrics):
    """Write only complete five-seed summaries; std uses ddof=1."""
    runs = pd.DataFrame([
        {"method": "CORCD", "dataset": p["dataset"], "seed": p["seed"], **p["metrics"]}
        for p in payloads
    ])
    if runs.duplicated(["dataset", "seed"]).any():
        raise ValueError("Duplicate dataset/seed in summary")
    for dataset in datasets:
        found = set(runs.loc[runs["dataset"] == dataset, "seed"])
        if found != set(BENCHMARK_SEEDS):
            raise ValueError(f"{dataset}: summary requires seeds {BENCHMARK_SEEDS}, got {sorted(found)}")
    records = []
    for dataset in datasets:
        group = runs.loc[runs["dataset"] == dataset]
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            records.append({
                "method": "CORCD", "dataset": dataset, "metric": metric,
                "n_seeds": len(BENCHMARK_SEEDS), "mean": mean, "std": std,
                "mean_std": f"{mean:.4f} ± {std:.4f}",
            })
    summary = pd.DataFrame(records)
    wide = summary.pivot(index=["method", "dataset", "n_seeds"], columns="metric", values="mean_std")
    wide = wide.reindex(columns=list(metrics)).reset_index()
    wide.columns.name = None
    output_dir = results_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("all_runs.csv", runs), ("summary_long.csv", summary), ("summary_wide.csv", wide)):
        temporary = output_dir / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output_dir / name)
    return output_dir


def main(argv=None):
    args = parse_args(argv)
    datasets = tuple(DATASET_CONFIG) if args.dataset == "all" else (args.dataset,)
    seeds = BENCHMARK_SEEDS if args.seed is None else (args.seed,)
    metrics = tuple(dict.fromkeys(canonical_metric_name(name) for name in args.metrics))
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    single_result_dir = args.result_dir.expanduser().resolve() if args.result_dir is not None else None

    # Require every selected run before evaluating or creating summary outputs.
    fingerprints = {}
    for dataset in datasets:
        for seed in seeds:
            result_dir = single_result_dir or results_root / dataset / f"seed_{seed}"
            fingerprints[dataset, seed] = validate_run(result_dir, dataset, seed)

    evaluated = []
    for dataset in datasets:
        data = load_evaluation_data(data_root, dataset)
        for seed in seeds:
            result_dir = single_result_dir or results_root / dataset / f"seed_{seed}"
            payload = evaluate_run(data, result_dir, dataset, seed, metrics, fingerprints[dataset, seed])
            evaluated.append((result_dir, payload))

    for result_dir, payload in evaluated:
        # Recheck all fingerprints before publishing any batch results.
        if validate_run(result_dir, payload["dataset"], payload["seed"]) != fingerprints[payload["dataset"], payload["seed"]]:
            raise RuntimeError(f"{result_dir}: training outputs changed during evaluation")
    for result_dir, payload in evaluated:
        write_run_outputs(result_dir, payload)
    if args.seed is None:
        summary_dir = write_summary(results_root, [p for _, p in evaluated], datasets, metrics)
        print(f"Five-seed summary saved to {summary_dir}")


if __name__ == "__main__":
    main()

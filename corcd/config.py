"""Resolve and validate CORCD experiment settings."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from .dataset_registry import BENCHMARK_SEEDS, DATASET_IDS, TRAINING_EPOCHS, GRAPH_SCALES, LOSS_WEIGHTS


# Shared implementation settings for the main experiments.
DEFAULT_CONFIG = {
    "preprocessing": {
        "rna_hvg": 3000,
        "rna_pca": 30,
        "adt_pca": 30,
        "atac_pca": 60
    },
    "model": {
        "emb_dim": 128,
        "encoder": "sage",
        "graph_fusion": "mean",
        "train_graph_encoder": True,
        "train_graph_decoder": True,
        "share_graph_encoder": True,
        "modality_fusion_mode": "linear"
    },
    "graph": {
        "scales": [
            4,
            8,
            12
        ],
        "symmetric": False
    },
    "loss": {
        "reconstruction": {
            "weight": 1.0,
            "modality_weights": {
                "RNA": 1.0,
                "ADT": 1.0,
                "ATAC": 1.0
            }
        },
        "contrastive": {
            "weight": 0.05,
            "max_pairs": 4096,
            "max_positives": 2,
            "farthest_ratio": 0.6,
            "margin": 0.5,
            "distance_batch_size": 128,
            "neighbor_k": 3
        },
        "distillation": {
            "weight": 0.05,
            "temperature": 0.2,
            "relation_k": 16,
            "max_nodes": 1024,
            "teacher_mode": "aggregated"
        }
    },
    "training": {
        "lr": 0.001,
        "weight_decay": 1e-05,
        "deterministic": True,
        "grad_clip_norm": 5.0
    },
    "clustering": {
        "method": "mclust",
        "use_pca": True,
        "n_comps": 20
    }
}


def load_config(dataset: str) -> dict[str, Any]:
    """Resolve shared settings and the published dataset defaults."""
    if dataset not in DATASET_IDS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASET_IDS)}")
    config = deepcopy(DEFAULT_CONFIG)
    config["training"]["epochs"] = TRAINING_EPOCHS[dataset]
    config["graph"]["scales"] = list(GRAPH_SCALES.get(dataset, config["graph"]["scales"]))
    for objective, weight in zip(("contrastive", "distillation"), LOSS_WEIGHTS[dataset]):
        config["loss"][objective]["weight"] = weight
    validate_config(config, dataset)
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable identifier for a resolved configuration."""
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _positive(value: Any, key: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a positive {'integer' if integer else 'number'}")
    if not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{key} must be a positive {'integer' if integer else 'number'}")


def _nonnegative(value: Any, key: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a nonnegative {'integer' if integer else 'number'}")
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{key} must be a nonnegative {'integer' if integer else 'number'}")


def validate_config(config: Mapping[str, Any], dataset: str | None = None) -> None:
    """Check the supported main-experiment method and essential numeric settings."""
    if dataset is not None and dataset not in DATASET_IDS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    required = {"preprocessing", "model", "graph", "loss", "training", "clustering"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(sorted(missing))}")
    for section in required:
        if not isinstance(config[section], Mapping):
            raise ValueError(f"{section} must be a mapping")
    model, graph, loss = config["model"], config["graph"], config["loss"]
    scales = graph.get("scales")
    if not isinstance(scales, list) or len(scales) < 3:
        raise ValueError("graph.scales must contain at least three distinct integers greater than 2")
    for scale in scales:
        _positive(scale, "graph.scales", integer=True)
    if len(set(scales)) != len(scales) or min(scales) <= 2:
        raise ValueError("graph.scales must contain at least three distinct integers greater than 2")
    if model.get("graph_fusion") != "mean":
        raise ValueError("CORCD main experiments use mean scale fusion")
    if model.get("modality_fusion_mode") != "linear":
        raise ValueError("CORCD main experiments require linear modality fusion")
    if model.get("encoder") != "sage":
        raise ValueError("CORCD main experiments require the sage encoder")
    for flag in ("train_graph_encoder", "train_graph_decoder", "share_graph_encoder"):
        if model.get(flag) is not True:
            raise ValueError(f"CORCD main experiments require model.{flag}=true")
    _positive(model.get("emb_dim"), "model.emb_dim", integer=True)
    objectives = {"reconstruction", "contrastive", "distillation"}
    if set(loss) != objectives:
        raise ValueError("loss must contain exactly reconstruction, contrastive, and distillation")
    for objective in objectives:
        if not isinstance(loss[objective], Mapping):
            raise ValueError(f"loss.{objective} must be a mapping")
        _positive(loss[objective].get("weight"), f"loss.{objective}.weight")
    reconstruction, contrastive, distillation = (loss[name] for name in ("reconstruction", "contrastive", "distillation"))
    modality_weights = reconstruction.get("modality_weights")
    if not isinstance(modality_weights, Mapping) or not modality_weights:
        raise ValueError("loss.reconstruction.modality_weights must be a nonempty mapping")
    for modality, weight in modality_weights.items():
        if modality not in {"RNA", "ADT", "ATAC"}:
            raise ValueError(f"Unknown reconstruction modality {modality!r}")
        _positive(weight, f"loss.reconstruction.modality_weights.{modality}")
    if distillation.get("teacher_mode") != "aggregated":
        raise ValueError("CORCD main experiments require the aggregated relation teacher")
    _positive(distillation.get("temperature"), "loss.distillation.temperature")
    allowed_contrastive = {
        "neighbor_k", "max_positives", "farthest_ratio",
        "margin", "max_pairs", "distance_batch_size", "weight",
    }
    unknown_contrastive = set(contrastive) - allowed_contrastive
    if unknown_contrastive:
        raise ValueError(f"Unsupported loss.contrastive fields: {', '.join(sorted(unknown_contrastive))}")
    for field in ("neighbor_k", "max_positives", "distance_batch_size"):
        _positive(contrastive.get(field), f"loss.contrastive.{field}", integer=True)
    _nonnegative(contrastive.get("max_pairs"), "loss.contrastive.max_pairs", integer=True)
    _positive(contrastive.get("margin"), "loss.contrastive.margin")
    _positive(contrastive.get("farthest_ratio"), "loss.contrastive.farthest_ratio")
    if contrastive["farthest_ratio"] > 1:
        raise ValueError("loss.contrastive.farthest_ratio must be between 0 (exclusive) and 1")
    for field in ("relation_k", "max_nodes"):
        _positive(distillation.get(field), f"loss.distillation.{field}", integer=True)
    for key, value in config["preprocessing"].items():
        _positive(value, f"preprocessing.{key}", integer=True)
    training = config["training"]
    _positive(training.get("epochs"), "training.epochs", integer=True)
    _positive(training.get("lr"), "training.lr")
    for field in ("weight_decay", "grad_clip_norm"):
        _nonnegative(training.get(field), f"training.{field}")
    if config["clustering"].get("method") != "mclust":
        raise ValueError("CORCD main experiments use mclust clustering")
    _positive(config["clustering"].get("n_comps"), "clustering.n_comps", integer=True)


# Public names used by the single-run and batch-run entry points.
validate = validate_config
experiment_config_hash = config_hash

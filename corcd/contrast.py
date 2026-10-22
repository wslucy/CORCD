"""Consensus-guided triplet sampling on reciprocal feature neighbors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import torch
import torch.nn.functional as F


@dataclass
class TripletSamples:
    anchors: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor
    modality_ids: torch.Tensor
    n_modalities: int


def contrastive_loss(
    z: torch.Tensor, samples: TripletSamples, *,
    margin: float = 0.5, max_pairs: int | None = None,
) -> torch.Tensor:
    """Sum per-modality mean triplet losses, without sample weights."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    total = z.sum() * 0.0
    for modality in range(samples.n_modalities):
        rows = torch.nonzero(samples.modality_ids == modality, as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        if max_pairs and max_pairs > 0 and rows.numel() > max_pairs:
            rows = rows[torch.randperm(rows.numel(), device=z.device)[:max_pairs]]
        anchors = z[samples.anchors[rows]]
        positives = z[samples.positives[rows]]
        negatives = z[samples.negatives[rows]]
        total = total + F.triplet_margin_loss(
            anchors, positives, negatives, margin=float(margin), p=2, reduction="mean"
        )
    return total


def spatial_knn_neighbors(spatial: np.ndarray, k: int) -> List[np.ndarray]:
    """Keep CORCD's existing KD neighborhood convention, handling tiny inputs."""
    from sklearn.neighbors import NearestNeighbors

    spatial = np.asarray(spatial, dtype=np.float32)
    n_nodes = int(spatial.shape[0])
    if n_nodes < 2:
        return [np.empty(0, dtype=np.int64) for _ in range(n_nodes)]
    k = max(1, min(int(k), n_nodes - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(spatial)
    # A query without X internally requests one extra neighbor. Preserve that
    # original query for normal datasets; use explicit X when N is too small.
    indices = nn.kneighbors(spatial if k + 1 >= n_nodes else None, return_distance=False)
    return [
        np.asarray([int(x) for x in row if int(x) != anchor][:k], dtype=np.int64)
        for anchor, row in enumerate(indices)
    ]


def _reciprocal_candidates(arrays: Sequence[np.ndarray], k: int, batch_size: int):
    """Reciprocal Euclidean neighbors on preprocessed modality features."""
    n_nodes = len(arrays[0])
    candidates = []
    for feature in arrays:
        nearest = []
        for start in range(0, n_nodes, batch_size):
            distances = cdist(feature[start:start + batch_size], feature, metric="euclidean")
            # Exclude self-pairs and exact-zero duplicate-feature distances.
            distances[distances == 0.0] = np.inf
            order = np.argsort(distances, axis=1, kind="stable")[:, :min(k, n_nodes)]
            for offset, row in enumerate(order):
                nearest.append(set(int(x) for x in row if np.isfinite(distances[offset, x])))
        modality = []
        for anchor, neighbors in enumerate(nearest):
            if not np.any(feature[anchor] != 0.0):
                modality.append(np.empty(0, dtype=np.int64))
            else:
                modality.append(np.asarray(
                    sorted(x for x in neighbors if anchor in nearest[x]), dtype=np.int64
                ))
        candidates.append(modality)
    return candidates


def build_consensus_samples(
    *, features: Sequence[torch.Tensor], spatial: np.ndarray,
    config: Dict, seed: int, device: torch.device,
) -> TripletSamples:
    """Rank reciprocal neighbors by cross-modality consensus and sample far negatives.

    Positives are reciprocal nearest neighbors within each modality. Consensus
    ranks this existing set; it does not weight the triplet loss.
    """
    if not features:
        raise ValueError("features cannot be empty")
    arrays = [x.detach().cpu().numpy().astype(np.float64, copy=False) for x in features]
    if any(x.ndim != 2 for x in arrays):
        raise ValueError("Features must be 2D matrices")
    n_nodes = len(arrays[0])
    if any(len(x) != n_nodes or not np.isfinite(x).all() for x in arrays):
        raise ValueError("All modalities must be finite matrices with the same spots")
    if len(spatial) != n_nodes:
        raise ValueError("Spatial coordinates and features must share spots")
    cfg = config["loss"]["contrastive"]
    k = int(cfg.get("neighbor_k", 3))
    max_pos = int(cfg.get("max_positives", 2))
    batch_size = int(cfg.get("distance_batch_size", 128))
    farthest_ratio = float(cfg.get("farthest_ratio", 0.6))
    if min(k, max_pos, batch_size) <= 0 or not 0.0 < farthest_ratio <= 1.0:
        raise ValueError("Neighbor counts must be positive and farthest_ratio must be in (0, 1]")
    rng = np.random.default_rng(int(seed))
    candidates = _reciprocal_candidates(arrays, k, batch_size)
    all_positives = [
        set(int(x) for modality in candidates for x in modality[anchor])
        for anchor in range(n_nodes)
    ]
    rows = []
    for start in range(0, n_nodes, batch_size):
        stop = min(start + batch_size, n_nodes)
        distances_by_modality = []
        consensus = np.zeros((stop - start, n_nodes), dtype=np.float64)
        for feature in arrays:
            distances = cdist(feature[start:stop], feature, metric="euclidean")
            distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
            distances_by_modality.append(distances)
            consensus += rankdata(distances, method="average", axis=1)
        # Average ranks first so exact score ties are preserved.
        consensus = np.clip(
            1.0 - (consensus / len(arrays) - 1.0) / max(n_nodes - 2, 1), 0.0, 1.0
        )
        for offset, anchor in enumerate(range(start, stop)):
            for modality, distances in enumerate(distances_by_modality):
                positives = candidates[modality][anchor]
                if not positives.size:
                    continue
                order = np.lexsort((positives, -consensus[offset, positives]))
                positives = positives[order[:max_pos]]
                source_distances = distances[offset]
                available = np.flatnonzero(np.isfinite(source_distances) & (source_distances > 0))
                count = max(1, int(available.size * farthest_ratio))
                order = np.lexsort((available, -source_distances[available]))
                # Exclude the complete positive union, including unselected positives.
                far_pool = np.asarray(
                    [x for x in available[order[:count]] if int(x) not in all_positives[anchor]],
                    dtype=np.int64,
                )
                if not far_pool.size:
                    continue
                for positive in positives:
                    negative = int(rng.choice(far_pool))
                    rows.append((anchor, int(positive), negative, modality))
    # Canonical order makes output independent of distance batching.
    rows.sort(key=lambda row: (row[3], row[0], row[1]))
    matrix = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    columns = [torch.as_tensor(matrix[:, col], dtype=torch.long, device=device) for col in range(4)]
    return TripletSamples(*columns, n_modalities=len(arrays))

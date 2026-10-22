"""Spatial nearest-neighbor graph construction for CORCD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphBundle:
    """Graph edges and weights indexed by modality and graph scale."""

    edges: List[List[torch.Tensor]]
    branch_names: List[str]
    metadata: Dict[str, object]
    edge_weights: List[List[torch.Tensor]] | None = None


def _as_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def knn_edge_index(
    values,
    k: int,
    *,
    symmetric: bool = False,
    include_self: bool = False,
) -> np.ndarray:
    values = _as_numpy(values).astype(np.float32, copy=False)
    n = values.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)
    k = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=k + (0 if include_self else 1))
    nn.fit(values)
    idx = nn.kneighbors(values, return_distance=False)
    if not include_self:
        idx = idx[:, 1:]

    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64)
    edges = np.stack([src, dst], axis=0)
    if symmetric and edges.shape[1] > 0:
        edges = np.concatenate([edges, edges[::-1]], axis=1)
    return unique_edge_index(edges)


def unique_edge_index(edge_index: np.ndarray) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    pairs = np.unique(edge_index.T, axis=0)
    return pairs.T.copy()


def build_multiscale_bundle(features, spatial, scales, symmetric, device) -> GraphBundle:
    """Build the same spatial graph scales for every modality."""
    clean = sorted({int(x) for x in scales})
    if len(clean) < 3 or min(clean) <= 2:
        raise ValueError(f"CORCD requires >=3 distinct scales and all k>2, got {clean}")
    edges, weights, counts = [], [], {}
    for k in clean:
        arr = knn_edge_index(spatial, k, symmetric=bool(symmetric))
        edge = torch.as_tensor(arr, dtype=torch.long, device=device)
        weight = torch.ones(edge.shape[1], dtype=torch.float32, device=device)
        edges.append(edge)
        weights.append(weight)
        counts[f"spatial_k{k}"] = int(edge.shape[1])
    return GraphBundle(
        edges=[[edge for edge in edges] for _ in features],
        branch_names=[f"spatial_k{k}" for k in clean],
        metadata={"mode": "manual_multi_scale_spatial", "scales": clean, "edge_counts": counts},
        edge_weights=[[weight for weight in weights] for _ in features],
    )

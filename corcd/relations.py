"""Multi-scale fusion and aggregated cross-omics relation distillation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import MultiScaleBackbone
from .contrast import spatial_knn_neighbors


class CORCDModel(MultiScaleBackbone):
    """GraphSAGE model with mean scale fusion and linear modality fusion."""

    def encode(self, features, graph_edges, graph_weights=None):
        _, modality_embeddings, branch_embeddings = super().encode(
            features, graph_edges, graph_weights
        )
        z = self.modality_fusion(torch.cat(modality_embeddings, dim=1))
        return z, modality_embeddings, branch_embeddings


def _candidate_tensor(spatial: np.ndarray, k: int, device: torch.device) -> torch.Tensor:
    rows = spatial_knn_neighbors(spatial, k)
    max_len = max((len(row) for row in rows), default=0)
    if max_len == 0:
        raise ValueError("No spatial neighbours available")
    arr = np.zeros((len(rows), max_len), dtype=np.int64)
    for idx, row in enumerate(rows):
        if len(row) == 0:
            arr[idx] = idx
        elif len(row) < max_len:
            arr[idx, :len(row)] = row
            arr[idx, len(row):] = row[-1]
        else:
            arr[idx] = row[:max_len]
    return torch.as_tensor(arr, dtype=torch.long, device=device)


class LocalScaleRelationKD:
    """Match student relations to the mean of scale-specific teacher relations."""

    def __init__(self, spatial: np.ndarray, relation_k: int = 16):
        self.spatial = spatial
        self.relation_k = int(relation_k)
        self._cache: Dict[str, torch.Tensor] = {}

    def __call__(self, student, branch_embeddings, *, temperature: float, max_nodes: int):
        if not branch_embeddings or not branch_embeddings[0]:
            return student.sum() * 0.0
        key = str(student.device)
        if key not in self._cache:
            self._cache[key] = _candidate_tensor(self.spatial, self.relation_k, student.device)
        candidates = self._cache[key]
        n_nodes = int(student.shape[0])
        if int(max_nodes) > 0 and n_nodes > int(max_nodes):
            anchors = torch.randperm(n_nodes, device=student.device)[:int(max_nodes)]
        else:
            anchors = torch.arange(n_nodes, device=student.device)
        selected = candidates[anchors]

        student_norm = F.normalize(student, p=2, dim=1)
        student_logits = (
            student_norm[anchors].unsqueeze(1) * student_norm[selected]
        ).sum(dim=-1) / float(temperature)
        log_student = F.log_softmax(student_logits, dim=1)
        n_scales = len(branch_embeddings[0])
        teacher_prob = torch.zeros_like(student_logits)
        with torch.no_grad():
            for scale_idx in range(n_scales):
                modality_views = [
                    F.normalize(modality[scale_idx].detach(), p=2, dim=1)
                    for modality in branch_embeddings
                ]
                teacher = torch.stack(modality_views, dim=0).mean(dim=0)
                teacher = F.normalize(teacher, p=2, dim=1)
                logits = (
                    teacher[anchors].unsqueeze(1) * teacher[selected]
                ).sum(dim=-1) / float(temperature)
                teacher_prob += F.softmax(logits, dim=1) / float(n_scales)
        return F.kl_div(log_student, teacher_prob, reduction="batchmean")

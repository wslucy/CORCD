"""Training with reconstruction, consensus contrast and relation distillation."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .contrast import build_consensus_samples, contrastive_loss
from .relations import CORCDModel, LocalScaleRelationKD


def _to_device_nested(values, device):
    if values is None:
        return None
    return [[x.to(device) for x in modality] for modality in values]


def train_corcd(
    *, features: Sequence[torch.Tensor], graph_bundle,
    config: Dict, spatial, device: torch.device,
):
    """Train CORCD and return the final embedding and completed epochs."""
    model_cfg = config["model"]
    train_cfg = config["training"]
    rec_cfg = config["loss"]["reconstruction"]
    contrast_cfg = config["loss"]["contrastive"]
    distill_cfg = config["loss"]["distillation"]

    model = CORCDModel(
        input_dims=[int(x.shape[1]) for x in features],
        emb_dim=int(model_cfg["emb_dim"]),
        n_graph_branches=len(graph_bundle.branch_names),
        encoder_name=str(model_cfg.get("encoder", "sage")),
        graph_fusion=str(model_cfg.get("graph_fusion", "mean")),
        train_graph_encoder=bool(model_cfg.get("train_graph_encoder", True)),
        train_graph_decoder=bool(model_cfg.get("train_graph_decoder", True)),
        share_graph_encoder=bool(model_cfg.get("share_graph_encoder", True)),
    ).to(device)
    features = [x.to(device) for x in features]
    graph_bundle.edges = _to_device_nested(graph_bundle.edges, device)
    if hasattr(graph_bundle, "edge_weights"):
        graph_bundle.edge_weights = _to_device_nested(graph_bundle.edge_weights, device)
    graph_weights = getattr(graph_bundle, "edge_weights", None)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    samples = build_consensus_samples(
        features=features, spatial=spatial, config=config,
        seed=int(train_cfg["seed"]), device=device,
    )
    relation_kd = LocalScaleRelationKD(spatial, int(distill_cfg.get("relation_k", 16)))
    pair_w = float(contrast_cfg["weight"])
    kd_w = float(distill_cfg["weight"])

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features, graph_bundle.edges, graph_weights)
        z = outputs["embedding"]
        modality_names = config.get("runtime", {}).get(
            "modalities", [str(i) for i in range(len(features))]
        )
        weight_map = rec_cfg.get("modality_weights", {}) or {}
        rec_weights = torch.as_tensor(
            [float(weight_map.get(str(name), 1.0)) for name in modality_names],
            dtype=z.dtype, device=z.device,
        )
        rec_weights = rec_weights / rec_weights.mean().clamp_min(1e-8)
        rec_loss = sum(
            rec_weights[idx] * F.mse_loss(feature, reconstruction)
            for idx, (feature, reconstruction) in enumerate(zip(features, outputs["reconstructions"]))
        )
        pair_loss = contrastive_loss(
            z, samples,
            margin=float(contrast_cfg["margin"]),
            max_pairs=int(contrast_cfg.get("max_pairs", 4096)),
        )
        kd_loss = relation_kd(
            z, outputs["branch_embeddings"],
            temperature=float(distill_cfg.get("temperature", 0.2)),
            max_nodes=int(distill_cfg.get("max_nodes", 1024)),
        )
        total_loss = float(rec_cfg.get("weight", 1.0)) * rec_loss + pair_w * pair_loss + kd_w * kd_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        total_loss.backward()
        grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_outputs = model(features, graph_bundle.edges, graph_weights)
    return final_outputs["embedding"], {"epochs_completed": int(epoch)}

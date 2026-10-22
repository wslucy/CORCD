"""The GraphSAGE backbone used by the CORCD main experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class SAGEConv_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class MultiScaleBackbone(nn.Module):
    """Shared GraphSAGE encoders, mean scale fusion, and linear modality fusion.

    Module construction order preserves the seeded initialization used in
    the main experiments.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        emb_dim: int,
        n_graph_branches: int,
        *,
        encoder_name: str = "sage",
        graph_fusion: str = "mean",
        train_graph_encoder: bool = True,
        train_graph_decoder: bool = True,
        share_graph_encoder: bool = True,
    ):
        super().__init__()
        if n_graph_branches < 1:
            raise ValueError("n_graph_branches must be >= 1")
        if encoder_name.lower() != "sage" or graph_fusion.lower() != "mean":
            raise ValueError("The CORCD backbone uses GraphSAGE and mean scale fusion")
        if not (train_graph_encoder and train_graph_decoder and share_graph_encoder):
            raise ValueError("CORCD requires trainable shared graph encoders and trainable decoders")
        self.input_dims = list(map(int, input_dims))
        self.emb_dim = int(emb_dim)
        self.n_graph_branches = int(n_graph_branches)
        self.encoder_name = encoder_name.lower()
        self.graph_fusion = graph_fusion.lower()
        self.train_graph_encoder = bool(train_graph_encoder)
        self.train_graph_decoder = bool(train_graph_decoder)
        self.share_graph_encoder = bool(share_graph_encoder)

        self.encoders = nn.ModuleList(
            [nn.ModuleList([SAGEConv_Encoder(in_dim, emb_dim)]) for in_dim in self.input_dims]
        )
        self.decoders = nn.ModuleList(
            [SAGEConv_Decoder(emb_dim, in_dim) for in_dim in self.input_dims]
        )
        self.branch_fusions = nn.ModuleList([nn.Identity() for _ in self.input_dims])
        self.modality_fusion = nn.Linear(len(self.input_dims) * emb_dim, emb_dim)
        self._set_graph_trainability()

    @property
    def uses_edge_weights(self) -> bool:
        return False

    def _set_graph_trainability(self) -> None:
        for parameter in self.encoders.parameters():
            parameter.requires_grad = self.train_graph_encoder
        for parameter in self.decoders.parameters():
            parameter.requires_grad = self.train_graph_decoder

    def trainable_parameter_summary(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable_parameters": int(trainable), "frozen_parameters": int(frozen)}

    def _apply_graph_module(self, module, x, edge_index, edge_weight=None):
        return module(x, edge_index)

    def encode(self, features, graph_edges, graph_weights=None):
        if len(features) != len(self.encoders):
            raise ValueError("Feature modality count differs from model modality count")
        if len(graph_edges) != len(features):
            raise ValueError("Graph modality count differs from feature modality count")
        if any(len(edges) != self.n_graph_branches for edges in graph_edges):
            raise ValueError("Every modality must provide the configured number of graph branches")
        if graph_weights is None:
            graph_weights = [[None for _ in edges] for edges in graph_edges]

        modality_embeddings: List[torch.Tensor] = []
        branch_embeddings: List[List[torch.Tensor]] = []
        for feature, modality_edges, modality_weights, modality_encoders in zip(
            features, graph_edges, graph_weights, self.encoders
        ):
            encoder = modality_encoders[0]
            branches = [
                self._apply_graph_module(encoder, feature, edge_index, edge_weight)
                for edge_index, edge_weight in zip(modality_edges, modality_weights)
            ]
            branch_embeddings.append(branches)
            modality_z = torch.stack(branches, dim=0).mean(dim=0)
            modality_embeddings.append(modality_z)

        z = self.modality_fusion(torch.cat(modality_embeddings, dim=1))
        return z, modality_embeddings, branch_embeddings

    def decode(self, z, graph_edges, graph_weights=None):
        if graph_weights is None:
            graph_weights = [[None for _ in edges] for edges in graph_edges]
        return [
            self._apply_graph_module(decoder, z, edges[0], weights[0])
            for decoder, edges, weights in zip(self.decoders, graph_edges, graph_weights)
        ]

    def forward(self, features, graph_edges, graph_weights=None):
        z, modality_embeddings, branch_embeddings = self.encode(features, graph_edges, graph_weights)
        reconstructions = self.decode(z, graph_edges, graph_weights)
        return {
            "embedding": z,
            "reconstructions": reconstructions,
            "modality_embeddings": modality_embeddings,
            "branch_embeddings": branch_embeddings,
        }

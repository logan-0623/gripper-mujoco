from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from interaction_vla.graph.schema import SceneGraph


@dataclass
class SceneBatch:
    """A dense, masked batch of fixed-capacity scene graphs."""

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_mask: Tensor
    edge_mask: Tensor

    def clone(self) -> "SceneBatch":
        return SceneBatch(
            node_features=self.node_features.clone(),
            edge_index=self.edge_index.clone(),
            edge_features=self.edge_features.clone(),
            node_mask=self.node_mask.clone(),
            edge_mask=self.edge_mask.clone(),
        )

    def to(self, device: torch.device | str) -> "SceneBatch":
        return SceneBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_mask=self.node_mask.to(device),
            edge_mask=self.edge_mask.to(device),
        )


def scene_graphs_to_batch(graphs: Iterable[SceneGraph]) -> SceneBatch:
    graphs = tuple(graphs)
    if not graphs:
        raise ValueError("at least one scene graph is required")
    for graph in graphs:
        graph.validate()
    reference_edges = graphs[0].edge_index
    if any(not np.array_equal(graph.edge_index, reference_edges) for graph in graphs[1:]):
        raise ValueError("all scene graphs in a batch must share edge ordering")
    return SceneBatch(
        node_features=torch.from_numpy(np.stack([graph.node_features for graph in graphs])).float(),
        edge_index=torch.from_numpy(reference_edges).long(),
        edge_features=torch.from_numpy(np.stack([graph.edge_features for graph in graphs])).float(),
        node_mask=torch.from_numpy(np.stack([graph.node_mask for graph in graphs])).bool(),
        edge_mask=torch.from_numpy(np.stack([graph.edge_mask for graph in graphs])).bool(),
    )


def permute_scene_batch(batch: SceneBatch, permutation: Tensor) -> SceneBatch:
    """Relabel nodes while preserving the graph represented by every edge."""

    permutation = permutation.to(device=batch.edge_index.device, dtype=torch.long)
    node_count = batch.node_features.shape[1]
    if permutation.shape != (node_count,) or not torch.equal(
        torch.sort(permutation).values, torch.arange(node_count, device=permutation.device)
    ):
        raise ValueError("permutation must contain every node index exactly once")

    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(node_count, device=permutation.device)
    return SceneBatch(
        node_features=batch.node_features[:, permutation],
        edge_index=inverse[batch.edge_index],
        edge_features=batch.edge_features.clone(),
        node_mask=batch.node_mask[:, permutation],
        edge_mask=batch.edge_mask.clone(),
    )


class FlatEncoder(nn.Module):
    """MLP baseline receiving exactly the same masked graph payload, flattened."""

    def __init__(
        self,
        max_nodes: int,
        max_edges: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        input_dim = (
            max_nodes * node_feature_dim
            + max_edges * edge_feature_dim
            + max_nodes
            + max_edges
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, batch: SceneBatch) -> Tensor:
        node_mask = batch.node_mask.unsqueeze(-1)
        edge_mask = batch.edge_mask.unsqueeze(-1)
        payload = torch.cat(
            (
                (batch.node_features * node_mask).flatten(start_dim=1),
                (batch.edge_features * edge_mask).flatten(start_dim=1),
                batch.node_mask.float(),
                batch.edge_mask.float(),
            ),
            dim=-1,
        )
        return self.network(payload)


def _two_layer_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class GraphEncoder(nn.Module):
    """Permutation-invariant relational encoder with masked message passing."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        message_rounds: int,
    ) -> None:
        super().__init__()
        if message_rounds < 1:
            raise ValueError("message_rounds must be positive")
        self.node_encoder = _two_layer_mlp(node_feature_dim, hidden_dim)
        self.edge_encoder = _two_layer_mlp(edge_feature_dim, hidden_dim)
        self.message_networks = nn.ModuleList(
            [_two_layer_mlp(3 * hidden_dim, hidden_dim) for _ in range(message_rounds)]
        )
        self.update_networks = nn.ModuleList(
            [_two_layer_mlp(2 * hidden_dim, hidden_dim) for _ in range(message_rounds)]
        )
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, batch: SceneBatch) -> Tensor:
        node_mask = batch.node_mask.unsqueeze(-1)
        edge_mask = batch.edge_mask.unsqueeze(-1)
        nodes = self.node_encoder(batch.node_features * node_mask) * node_mask
        edges = self.edge_encoder(batch.edge_features * edge_mask) * edge_mask
        source, destination = batch.edge_index

        for message_network, update_network in zip(
            self.message_networks, self.update_networks, strict=True
        ):
            messages = message_network(
                torch.cat((nodes[:, source], nodes[:, destination], edges), dim=-1)
            ) * edge_mask
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(1, destination, messages)
            degree = torch.zeros(
                nodes.shape[:2], dtype=nodes.dtype, device=nodes.device
            )
            degree.index_add_(1, destination, batch.edge_mask.to(nodes.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
            nodes = update_network(torch.cat((nodes, aggregate), dim=-1)) * node_mask

        pooled = nodes.sum(dim=1) / batch.node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.output(pooled)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_matched_encoders(
    *,
    max_nodes: int,
    max_edges: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    graph_hidden_dim: int,
    embedding_dim: int,
    message_rounds: int,
) -> tuple[FlatEncoder, GraphEncoder]:
    """Build encoders whose trainable parameter counts differ by at most 10%."""
    flat_input_dim = (
        max_nodes * node_feature_dim
        + max_edges * edge_feature_dim
        + max_nodes
        + max_edges
    )
    parameters_per_hidden_unit = flat_input_dim + 1 + embedding_dim
    maximum_graph_hidden_dim = max(graph_hidden_dim * 2, graph_hidden_dim + 32)
    candidate_graph_widths = sorted(
        range(1, maximum_graph_hidden_dim + 1),
        key=lambda width: (abs(width - graph_hidden_dim), width),
    )
    best: tuple[float, FlatEncoder, GraphEncoder] | None = None
    for resolved_graph_hidden_dim in candidate_graph_widths:
        graph = GraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=resolved_graph_hidden_dim,
            embedding_dim=embedding_dim,
            message_rounds=message_rounds,
        )
        graph_count = count_parameters(graph)
        ideal_hidden_dim = (graph_count - embedding_dim) / parameters_per_hidden_unit
        candidate_flat_widths = {
            max(1, int(np.floor(ideal_hidden_dim))),
            max(1, int(np.ceil(ideal_hidden_dim))),
        }
        for flat_hidden_dim in candidate_flat_widths:
            flat = FlatEncoder(
                max_nodes=max_nodes,
                max_edges=max_edges,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=flat_hidden_dim,
                embedding_dim=embedding_dim,
            )
            flat_count = count_parameters(flat)
            relative_difference = abs(flat_count - graph_count) / max(
                flat_count, graph_count
            )
            if best is None or relative_difference < best[0]:
                best = (relative_difference, flat, graph)
            if relative_difference <= 0.10:
                return flat, graph

    assert best is not None
    relative_difference, flat, graph = best
    raise ValueError(
        "could not match Flat and Graph encoder sizes within 10%; "
        f"flat={count_parameters(flat)}, graph={count_parameters(graph)}, "
        f"relative_difference={relative_difference:.3f}"
    )

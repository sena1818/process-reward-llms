from __future__ import annotations

import torch
import torch.nn.functional as F


def pointwise_binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def nodewise_pointwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    node_indices: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Mean candidate BCE inside each node, then mean across nodes."""

    if logits.ndim != 1 or labels.ndim != 1 or node_indices.ndim != 1:
        raise ValueError("logits, labels, and node_indices must be one-dimensional")
    if not (len(logits) == len(labels) == len(node_indices)):
        raise ValueError("nodewise pointwise tensors must have equal length")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )
    sums = torch.zeros(num_nodes, dtype=losses.dtype, device=losses.device)
    counts = torch.zeros(num_nodes, dtype=losses.dtype, device=losses.device)
    sums.index_add_(0, node_indices, losses)
    counts.index_add_(0, node_indices, torch.ones_like(losses))
    if torch.any(counts == 0):
        raise ValueError("Every node must contain at least one candidate")
    return (sums / counts).mean()


def pairwise_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = -F.logsigmoid(positive_scores - negative_scores)
    if weights is not None:
        losses = losses * weights
    return losses.mean()


def nodewise_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    node_offsets: torch.Tensor,
) -> torch.Tensor:
    """Mean all positive/negative comparisons per node, then mean nodes."""

    if logits.ndim != 1 or labels.ndim != 1 or node_offsets.ndim != 1:
        raise ValueError("logits, labels, and node_offsets must be one-dimensional")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have equal length")
    if len(node_offsets) < 2:
        raise ValueError("node_offsets must delimit at least one node")

    node_losses: list[torch.Tensor] = []
    for start_tensor, end_tensor in zip(node_offsets[:-1], node_offsets[1:]):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        node_scores = logits[start:end]
        node_labels = labels[start:end]
        positive_scores = node_scores[node_labels == 1]
        negative_scores = node_scores[node_labels == 0]
        if len(positive_scores) == 0 or len(negative_scores) == 0:
            raise ValueError("Every strict V0 node needs positive and negative candidates")
        pair_losses = F.softplus(
            negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)
        )
        node_losses.append(pair_losses.mean())
    return torch.stack(node_losses).mean()


def hybrid_loss(
    pointwise_loss: torch.Tensor,
    pairwise_loss: torch.Tensor,
    lambda_pair: float,
) -> torch.Tensor:
    return pointwise_loss + lambda_pair * pairwise_loss

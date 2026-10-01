"""Exp 2: RoBERTa with positive class weights and masked BCE."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.dl.roberta_baseline import RobertaBaseline


def compute_pos_weight(labels: Tensor, label_mask: Tensor) -> Tensor:
    """Compute negative/positive counts using only observed training targets.

    Labels with no positives or no negatives receive a neutral weight of 1
    so their loss stays finite and their observed targets remain trainable.
    """
    if labels.ndim != 2 or labels.shape != label_mask.shape:
        raise ValueError("labels and label_mask must have the same 2D shape")
    mask = label_mask.to(device=labels.device, dtype=torch.bool)
    observed = labels[mask]
    if not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Observed labels must be binary")
    positives = ((labels == 1) & mask).sum(dim=0).to(torch.float32)
    negatives = ((labels == 0) & mask).sum(dim=0).to(torch.float32)
    valid = (positives > 0) & (negatives > 0)
    return torch.where(valid, negatives / positives.clamp_min(1),
                       torch.ones_like(positives))


class RobertaWeightedBCE(RobertaBaseline):
    """Use the baseline CLS head with per-label positive BCE weights.

    ``pos_weight`` is a registered loss buffer: it follows the model device
    and is saved in the model state dictionary.
    """

    def __init__(self, pos_weight: Tensor, model_name: str = "roberta-base",
                 num_labels: int = 19, threshold: float = 0.5) -> None:
        weights = torch.as_tensor(pos_weight, dtype=torch.float32).detach().clone()
        if weights.shape != (num_labels,):
            raise ValueError("pos_weight must contain one weight per label")
        if not torch.all(torch.isfinite(weights) & (weights > 0)):
            raise ValueError("pos_weight must be finite and positive")
        super().__init__(model_name=model_name, num_labels=num_labels,
                         threshold=threshold)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")

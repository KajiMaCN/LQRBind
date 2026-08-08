from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LQRBindLoss(nn.Module):
    """Class-weighted binary cross-entropy used for released LQRBind training."""

    def __init__(self, positive_weight: float):
        super().__init__()
        self.positive_weight = float(positive_weight)

    def forward(self, logits: torch.Tensor, strong_labels: torch.Tensor) -> torch.Tensor:
        strong_labels = strong_labels.float()
        sample_weights = torch.ones_like(strong_labels)
        sample_weights = torch.where(
            strong_labels > 0,
            sample_weights * self.positive_weight,
            sample_weights,
        )
        elementwise_loss = F.binary_cross_entropy_with_logits(
            logits, strong_labels, reduction="none"
        )
        return (elementwise_loss * sample_weights).mean()

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: Literal["mean", "sum", "none"] = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        prob_pos = probs
        prob_neg = 1.0 - probs

        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(prob_pos.clamp(min=self.eps))
        log_neg = torch.log(prob_neg.clamp(min=self.eps))

        loss = targets * log_pos + (1.0 - targets) * log_neg

        pt = targets * prob_pos + (1.0 - targets) * prob_neg
        gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
        one_sided_weight = torch.pow(1.0 - pt, gamma)
        loss = loss * one_sided_weight

        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1.0 - targets)
            loss = loss * class_weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
